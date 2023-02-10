use std::collections::HashMap;
use std::hash::Hash;

/// Any type that can be represented as a Var.
pub trait VarRepresentable: Sized + Clone + Hash + Eq {
    fn to_var_repr(&self, count: usize) -> Var {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();

        let var_ir = VarHashIntermediateRepr::new(count, self);
        var_ir.hash(&mut hasher);

        let var_ptr = hasher.finish();

        Var(var_ptr)
    }
}

// default implementations
macro_rules! varrepresentable_impl {
    ($($typ:ty),*) => {
        $(impl VarRepresentable for $typ {})*
    };
}

varrepresentable_impl! {i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, char, String, &'static str}

/// Provides an intermediate representation used for hashing a VarRepresentable
/// type and it's corresponding hash.
#[derive(Hash)]
struct VarHashIntermediateRepr<'a, T: VarRepresentable> {
    count: usize,
    val: &'a T,
}

impl<'a, T: VarRepresentable> VarHashIntermediateRepr<'a, T> {
    #[must_use]
    fn new(count: usize, val: &'a T) -> Self {
        Self { count, val }
    }
}

/// Represents a unique Var derived from a hashed input and the variable count.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Var(u64);

/// A Term representing either a Value or Variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term<T> {
    Var(Var),
    Value(T),
}

/// A map representing potentially recursive Variable to Terminal mappings.
pub type TermMapping<T> = HashMap<Var, Term<T>>;

#[derive(Default, Clone, PartialEq, Eq)]
pub struct State<T> {
    variable_count: usize,

    term_mapping: TermMapping<T>,
}
impl<T: Default> State<T> {
    pub fn empty() -> Self {
        Self::default()
    }
}

impl<T> State<T> {
    #[must_use]
    pub fn new(variable_count: usize, term_mapping: TermMapping<T>) -> Self {
        Self {
            variable_count,
            term_mapping,
        }
    }
}

impl<T> AsRef<TermMapping<T>> for State<T> {
    fn as_ref(&self) -> &TermMapping<T> {
        &self.term_mapping
    }
}

fn extend_mapping<T, PM>(source: Var, term: Term<T>, prev_mapping: PM) -> TermMapping<T>
where
    T: VarRepresentable,
    PM: AsRef<TermMapping<T>>,
{
    let mut extended_mapping = prev_mapping.as_ref().clone();
    extended_mapping.insert(source, term);

    extended_mapping
}

pub trait Walkable<T> {
    fn walk(&self, term: &Term<T>) -> Term<T>;
}

impl<T> Walkable<T> for TermMapping<T>
where
    T: VarRepresentable,
{
    fn walk(&self, term: &Term<T>) -> Term<T> {
        // recurse down the terms until either a Value is encounter or no
        // further walking can occur.
        let mut current_term = term.clone();
        while let Term::Var(var) = &current_term {
            match self.get(var) {
                Some(next) => current_term = next.clone(),
                None => break,
            }
        }

        current_term
    }
}

pub fn walk<T, M>(mapping: &M, term: &Term<T>) -> Term<T>
where
    T: VarRepresentable,
    M: Walkable<T>,
{
    Walkable::walk(mapping, term)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_walk_until_expected_value() {
        let a = 'a'.to_var_repr(0);
        let b = 'b'.to_var_repr(0);

        let mapping: TermMapping<u8> = {
            let mut mapping = HashMap::new();
            mapping.insert(a, Term::Var(b));
            mapping.insert(b, Term::Value(2));
            mapping
        };

        // assert both values reify to the value of 2
        assert_eq!(walk(&mapping, &Term::Var(a)), Term::Value(2));
        assert_eq!(walk(&mapping, &Term::Var(b)), Term::Value(2));
    }
}
