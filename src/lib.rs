use std::collections::HashMap;
use std::hash::Hash;

/// Any type that can be represented as a `Var`.
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

/// Any type that can be represented as a `Term::Value`.
pub trait ValueRepresentable: Sized + Clone + Eq {}

impl<T: Sized + Clone + Eq> ValueRepresentable for T {}

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
pub enum Term<T: ValueRepresentable> {
    Var(Var),
    Value(T),
}

impl<T: ValueRepresentable> Term<T> {
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
pub type ReprMapping = HashMap<Var, String>;
pub type OccurrenceCounter = HashMap<String, usize>;

#[derive(Default, Clone, PartialEq, Eq)]
pub struct State<T: ValueRepresentable> {
    occurence_counter: OccurrenceCounter,
    term_mapping: TermMapping<T>,

    repr_mapping: ReprMapping,
}
impl<T: Default + ValueRepresentable> State<T> {
    pub fn empty() -> Self {
        Self::default()
    }
}

impl<T: ValueRepresentable> State<T> {
    #[must_use]
    pub fn new(
        occurence_counter: OccurrenceCounter,
        term_mapping: TermMapping<T>,
        repr_mapping: ReprMapping,
    ) -> Self {
        Self {
            occurence_counter,
            term_mapping,
            repr_mapping,
        }
    }
}

impl<T: ValueRepresentable> State<T> {
    pub fn insert<VAR: VarRepresentable + std::fmt::Display>(&mut self, key: VAR, term: Term<T>) {
        let repr = key.to_string();
        let occurrences = self.occurence_counter.get(&repr).copied().unwrap_or(0);
        let var = key.to_var_repr(occurrences);

        self.occurence_counter
            .entry(repr.clone())
            .and_modify(|count| *count += 1)
            .or_insert(1);
        self.repr_mapping.entry(var).or_insert(repr);
        self.term_mapping.insert(var, term);
    }
}

impl<T: ValueRepresentable> AsRef<TermMapping<T>> for State<T> {
    fn as_ref(&self) -> &TermMapping<T> {
        &self.term_mapping
    }
}

impl<T: ValueRepresentable> AsMut<TermMapping<T>> for State<T> {
    fn as_mut(&mut self) -> &mut TermMapping<T> {
        &mut self.term_mapping
    }
}

impl<T: ValueRepresentable + std::fmt::Debug> std::fmt::Debug for State<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "State<{}> ", std::any::type_name::<T>())?;
        let mut dm = f.debug_map();

        for k in self.term_mapping.keys() {
            match (self.repr_mapping.get(k), self.term_mapping.get(k)) {
                // key cannot be resolved and term is a var.
                (None, Some(Term::Var(var))) => match self.repr_mapping.get(var) {
                    Some(val_repr) => dm.entry(k, val_repr),
                    None => dm.entry(k, var),
                },
                // key can be resolved and term is a var.
                (Some(key_repr), Some(Term::Var(var))) => match self.repr_mapping.get(var) {
                    Some(val_repr) => dm.entry(key_repr, val_repr),
                    None => dm.entry(key_repr, var),
                },

                // key cannot be resolved and term is a value.
                (None, Some(t @ Term::Value(_))) => dm.entry(k, t),

                // key can be resolved and term is a value.
                (Some(repr), Some(t @ Term::Value(_))) => dm.entry(repr, t),

                // by nature of the k being pulled from the map, this state
                // should be unreachable and should panic if it is ever reached.
                (_, None) => unreachable!(),
            };
        }

        dm.finish()
    }
}

pub trait Walkable<T: ValueRepresentable>: Clone {
    fn walk(&self, term: &Term<T>) -> Term<T>;
}

impl<T: ValueRepresentable> Walkable<T> for TermMapping<T> {
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

pub trait Goal<T: ValueRepresentable> {
    fn apply(&self, state: State<T>) -> Stream<T>;
}

impl<F, T> Goal<T> for F
where
    T: ValueRepresentable,
    F: Fn(State<T>) -> Stream<T>,
{
    fn apply(&self, state: State<T>) -> Stream<T> {
        self(state)
    }
}

pub struct BoxedGoal<'a, T: ValueRepresentable> {
    goal: Box<dyn Goal<T> + 'a>,
}

impl<'a, T: ValueRepresentable> BoxedGoal<'a, T> {
    pub fn new<S>(state: S) -> Self
    where
        S: Goal<T> + 'a,
    {
        BoxedGoal {
            goal: Box::new(state),
        }
    }
}

impl<'a, T: ValueRepresentable> Goal<T> for BoxedGoal<'a, T> {
    fn apply(&self, state: State<T>) -> Stream<T> {
        self.goal.apply(state)
    }
}

#[derive(Debug)]
pub struct Fresh<T, F, G>
where
    T: ValueRepresentable,
    F: Fn(State<T>) -> State<T>,
    G: Goal<T>,
{
    _value_kind: std::marker::PhantomData<T>,
    var_decl_fn: F,
    goal: G,
}

impl<T, F, G> Fresh<T, F, G>
where
    T: ValueRepresentable,
    F: Fn(State<T>) -> State<T>,
    G: Goal<T>,
{
    pub fn new(var_decl_fn: F, goal: G) -> Self {
        Self {
            _value_kind: std::marker::PhantomData,
            var_decl_fn,
            goal,
        }
    }
}

impl<T, F, G> Goal<T> for Fresh<T, F, G>
where
    T: ValueRepresentable,
    F: Fn(State<T>) -> State<T>,
    G: Goal<T>,
{
    fn apply(&self, state: State<T>) -> Stream<T> {
        let state = (self.var_decl_fn)(state);

        (self.goal).apply(state)
    }
}

pub fn fresh<T>(var_decl_fn: impl Fn(State<T>) -> State<T>, goal: impl Goal<T>) -> impl Goal<T>
where
    T: ValueRepresentable,
{
    Fresh::new(var_decl_fn, goal)
}

#[derive(Debug)]
pub struct Equal<T: ValueRepresentable> {
    term1: Term<T>,
    term2: Term<T>,
}

impl<T: ValueRepresentable> Equal<T> {
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
            vec![State::new(
                state.occurence_counter,
                term_mapping,
                state.repr_mapping,
            )]
        })
    }
}

pub fn eq<T>(term1: Term<T>, term2: Term<T>) -> impl Goal<T>
where
    T: ValueRepresentable,
    Equal<T>: Goal<T>,
{
    Equal::new(term1, term2)
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! prepare_state {
        ($(($var:literal, $term:expr)),*) => {
            [
                $(
                ($var, $term),
                )*
            ]
            .into_iter()
            .fold(State::default(), |mut state, (var, term)| {
                state.insert(var, term);
                state
            })
        };
    }

    #[test]
    fn should_walk_until_expected_value() {
        let a = 'a'.to_var_repr(0);
        let b = 'b'.to_var_repr(0);

        let mapping: TermMapping<u8> = [(a, Term::Var(b)), (b, Term::Value(2))]
            .into_iter()
            .collect();

        // assert both values reify to the value of 2
        assert_eq!(walk(&mapping, &Term::Var(a)), Term::Value(2));
        assert_eq!(walk(&mapping, &Term::Var(b)), Term::Value(2));
    }

    #[test]
    fn should_unify_equal_values() {
        let a = 'a'.to_var_repr(0);
        let b = 'b'.to_var_repr(0);
        let c = 'c'.to_var_repr(0);
        let d = 'd'.to_var_repr(0);
        let state = prepare_state!(
            ('c', Term::Value(1)),
            ('d', Term::Value(2)),
            ('b', Term::Var(c)),
            ('a', Term::Var(b))
        );

        let goal = eq(Term::<u8>::Var(a), Term::<u8>::Var(b));
        let stream = goal.apply(state.clone());
        assert!(stream.len() == 1);
        assert_eq!(4, stream[0].as_ref().len(), "{:?}", stream[0]);
        assert_eq!(Term::Value(1), stream[0].as_ref().walk(&Term::Var(a)));

        let goal = eq(Term::<u8>::Var(a), Term::<u8>::Var(d));
        let stream = goal.apply(state);
        assert!(stream.is_empty());
    }
}
