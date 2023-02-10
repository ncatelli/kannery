use std::collections::HashMap;
use std::hash::Hash;

/// Any type that can be represented as a Var.
pub trait VarRepresentable: Sized + Clone + Hash + Eq {
    fn to_repr(&self, count: usize) -> Var {
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

varrepresentable_impl! {i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, String, &'static str}

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

trait Walkable<T> {
    fn walk(&self, term: &Term<T>) -> Term<T>;
}

impl<T: VarRepresentable, TM: AsRef<TermMapping<T>>> Walkable<T> for TM {
    fn walk(&self, term: &Term<T>) -> Term<T> {
        let mapping = self.as_ref();

        // recurse down the terms until either a Value is encounter or no
        // further walking can occur.
        let mut current_term = term.clone();
        while let Term::Var(var) = &current_term {
            match mapping.get(var) {
                Some(next) => current_term = next.clone(),
                None => break,
            }
        }

        current_term
    }
}

fn walk<T, M>(mapping: M, term: &Term<T>) -> Term<T>
where
    T: VarRepresentable,
    M: Walkable<T>,
{
    mapping.walk(term)
}

#[cfg(test)]
mod tests {
    use super::*;
}
