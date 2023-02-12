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

impl<T> Term<T> {
    /// Returns a boolean signifying if the type is a `Var` variant.
    pub fn is_var(&self) -> bool {
        matches!(self, Term::Var(_))
    }

    /// Returns a boolean signifying if the type is a `Value` variant.
    pub fn is_value(&self) -> bool {
        matches!(self, Term::Value(_))
    }
}

/// A map representing potentially recursive Variable to Terminal mappings.
pub type TermMapping<T> = HashMap<Var, Term<T>>;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
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

impl<T> AsMut<TermMapping<T>> for State<T> {
    fn as_mut(&mut self) -> &mut TermMapping<T> {
        &mut self.term_mapping
    }
}

pub trait Walkable<T>: Clone {
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

pub fn unify<T: VarRepresentable>(
    mapping: &TermMapping<T>,
    term1: &Term<T>,
    term2: &Term<T>,
) -> Option<TermMapping<T>> {
    let t1_target = walk(mapping, term1);
    let t2_target = walk(mapping, term2);
    let mut mapping = mapping.clone();

    match (t1_target, t2_target) {
        (Term::Var(v1), Term::Var(v2)) if v1 == v2 => Some(mapping),
        (Term::Var(v), t) | (t, Term::Var(v)) => {
            mapping.insert(v, t);
            Some(mapping)
        }
        (Term::Value(v1), Term::Value(v2)) if v1 == v2 => Some(mapping),
        _ => None,
    }
}

// Represents a calleable stream of states.
pub type Stream<T> = Vec<State<T>>;

pub trait Goal<T> {
    fn apply(&self, state: State<T>) -> Stream<T>;
}

impl<F, T> Goal<T> for F
where
    F: Fn(State<T>) -> Stream<T>,
{
    fn apply(&self, state: State<T>) -> Stream<T> {
        self(state)
    }
}

pub struct BoxedGoal<'a, T> {
    goal: Box<dyn Goal<T> + 'a>,
}

impl<'a, T> BoxedGoal<'a, T> {
    pub fn new<S>(state: S) -> Self
    where
        S: Goal<T> + 'a,
    {
        BoxedGoal {
            goal: Box::new(state),
        }
    }
}

impl<'a, T> Goal<T> for BoxedGoal<'a, T> {
    fn apply(&self, state: State<T>) -> Stream<T> {
        self.goal.apply(state)
    }
}

#[derive(Debug)]
pub struct Equal<T: VarRepresentable> {
    term1: Term<T>,
    term2: Term<T>,
}

impl<T: VarRepresentable> Equal<T> {
    pub fn new(term1: Term<T>, term2: Term<T>) -> Self {
        Self { term1, term2 }
    }
}

impl<T: VarRepresentable> Goal<T> for Equal<T> {
    fn apply(&self, state: State<T>) -> Stream<T> {
        let sub_mapping = state.as_ref();
        let unified_mapping = unify(sub_mapping, &self.term1, &self.term2);

        // Return an empty stream if the mapping is `None`.
        unified_mapping.map_or_else(Stream::new, |term_mapping| {
            vec![State::new(state.variable_count, term_mapping)]
        })
    }
}

pub fn eq<T: VarRepresentable>(term1: Term<T>, term2: Term<T>) -> impl Goal<T>
where
    Equal<T>: Goal<T>,
{
    Equal::new(term1, term2)
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

    #[test]
    fn should_unify_equal_values() {
        let a = 'a'.to_var_repr(0);
        let b = 'b'.to_var_repr(0);
        let c = 'c'.to_var_repr(0);

        let mapping: TermMapping<u8> = {
            let mut mapping = HashMap::new();
            mapping.insert(a, Term::Var(b));
            mapping.insert(b, Term::Value(1));
            mapping.insert(c, Term::Value(1));

            mapping
        };

        let goal = eq(Term::<u8>::Var(a), Term::<u8>::Var(c));

        let stream = goal.apply(State::new(0, mapping));

        println!("a={:?}\nb={:?}\nc={:?}", a, b, c);
        println!("{:?}", &stream);
    }
}
