use std::collections::HashMap;
use std::hash::Hash;

/// Any type that can be represented as a `Var`.
pub trait VarRepresentable: Sized + Clone + Hash + Eq {
    fn to_var_repr(&self, count: usize) -> Var {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();

        self.hash(&mut hasher);

        let var_repr = hasher.finish();

        Var::new(var_repr, count)
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

/// Represents a unique Var derived from a hashed input and the variable count.
#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Var {
    base: u64,
    count: usize,
}

impl Var {
    pub fn new(base: u64, count: usize) -> Self {
        Self { base, count }
    }

    pub fn as_base(&self) -> u64 {
        self.base
    }

    pub fn as_count(&self) -> usize {
        self.count
    }
}

/// A Term representing either a Value or Variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term<T: ValueRepresentable> {
    Var(Var),
    Value(T),
    // In place of a traditional cons list.
    Cons(Box<Term<T>>, Box<Term<T>>),
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

    /// Returns a boolean signifying if the type is a `Cons` variant.
    pub fn is_cons(&self) -> bool {
        matches!(self, Term::Cons(_, _))
    }
}

impl<T: ValueRepresentable> From<(Term<T>, Term<T>)> for Term<T> {
    fn from((head, tail): (Term<T>, Term<T>)) -> Self {
        cons(head, tail)
    }
}

/// Generate a cons list from a given head/tail value.
pub fn cons<T: ValueRepresentable>(head: Term<T>, tail: Term<T>) -> Term<T> {
    Term::Cons(Box::new(head), Box::new(tail))
}

/// A map representing potentially recursive Variable to Terminal mappings.
type TermMapping<T> = HashMap<Var, Term<T>>;

/// A map representing a Variable to it's string representation.
type ReprMapping = HashMap<Var, String>;

/// A map representing a Variable repr's occurrence count.
type OccurrenceCounter = HashMap<String, usize>;

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
    /// Declare and define a tracked `Var` for a given `Term<T>`.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::{State, Term};
    ///
    /// let mut state = State::<u8>::empty();
    /// state.define('a', Term::Value(1));
    /// ```
    pub fn define<VAR: VarRepresentable + std::fmt::Display>(
        &mut self,
        key: VAR,
        term: Term<T>,
    ) -> Var {
        let var = self.declare(key);
        self.term_mapping.insert(var, term);

        var
    }

    /// Dedeclare a tracked `Var` occurrence for a given key.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::State;
    ///
    /// let mut state = State::<u8>::empty();
    /// state.declare('a');
    /// ```
    pub fn declare<VAR: VarRepresentable + std::fmt::Display>(&mut self, key: VAR) -> Var {
        let repr = key.to_string();
        let occurrences = self.occurence_counter.get(&repr).copied().unwrap_or(0);
        let var = key.to_var_repr(occurrences);

        self.occurence_counter
            .entry(repr.clone())
            .and_modify(|count| *count += 1)
            .or_insert(1);
        self.repr_mapping.entry(var).or_insert(repr);

        var
    }

    /// Retrieves all occurrences, ordered by their occurrence of a given key.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::State;
    ///
    /// let state = {
    ///     let mut state = State::<u8>::empty();
    ///     for _ in 0..5 {
    ///         state.declare('a');
    ///     }
    ///
    ///     state
    /// };
    ///
    /// let vars = state.get_vars_by_key('a');
    /// assert!(vars.is_some());
    /// assert_eq!(vars.map(|v| v.len()), Some(5));
    /// ```
    pub fn get_vars_by_key<VAR: VarRepresentable + std::fmt::Display>(
        &self,
        key: VAR,
    ) -> Option<Vec<Var>> {
        let repr = key.to_string();
        let count = self.occurence_counter.get(&repr).copied()?;

        let vars = (0..count)
            .into_iter()
            .map(|count| key.to_var_repr(count))
            .collect();

        Some(vars)
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

                // key cannot be resolved and its value is a cons list.
                (None, Some(t @ Term::Cons(_, _))) => dm.entry(k, t),

                // key can be resolved and its term is a cons list.
                (Some(repr), Some(t @ Term::Cons(_, _))) => dm.entry(repr, t),

                // by nature of the k being pulled from the map, this state
                // should be unreachable and should panic if it is ever reached.
                (_, None) => unreachable!(),
            };
        }

        dm.finish()
    }
}

/// A trait for defining the behavior of traversing (_looking up_) the term
/// that a variable maps to.
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

/// Walk the terminal mapping returning the resolved term of a given variable, or itself.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let a = 'a'.to_var_repr(0);
///
/// let goal = fresh(|mut state| {
///     let a = state.declare('a');
///     let b = state.declare('b');
///     let c = state.declare('c');
///     let d = state.declare('d');
///
///     conjunction(
///         eq(Term::<u8>::Var(a), Term::<u8>::Var(b)),
///         conjunction(
///             eq(Term::<u8>::Var(b), Term::<u8>::Var(c)),
///             conjunction(
///                 eq(Term::<u8>::Var(c), Term::<u8>::Value(1)),
///                 eq(Term::<u8>::Var(d), Term::<u8>::Value(2)),
///             ),
///         ),
///     )
///     .apply(state)
/// });
///
/// let stream = goal.apply(State::empty());
/// assert_eq!(Term::Value(1), stream[0].as_ref().walk(&Term::Var(a)));
/// ```
pub fn walk<T, M>(mapping: &M, term: &Term<T>) -> Term<T>
where
    T: VarRepresentable,
    M: Walkable<T>,
{
    Walkable::walk(mapping, term)
}

/// A type for defining walking against a set of states.
pub trait DeepWalkable<T: ValueRepresentable>: Clone {
    fn deep_walk(&self, term: &Term<T>) -> Vec<Term<T>>;
}

impl<T: VarRepresentable> DeepWalkable<T> for Stream<T> {
    fn deep_walk(&self, term: &Term<T>) -> Vec<Term<T>> {
        self.iter()
            .map(|state| {
                let mapping = state.as_ref();
                Walkable::walk(mapping, term)
            })
            .collect()
    }
}

/// Resolve a variable against a set of states, returning all possible values.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let a = 'a'.to_var_repr(0);
///
/// let goal = fresh(|mut state| {
///     let a = state.declare('a');
///     let b = state.declare('b');
///     let c = state.declare('c');
///     let d = state.declare('d');
///
///     conjunction(
///         eq(Term::<u8>::Var(a), Term::<u8>::Var(b)),
///         conjunction(
///             eq(Term::<u8>::Var(b), Term::<u8>::Var(c)),
///             conjunction(
///                 eq(Term::<u8>::Var(c), Term::<u8>::Value(1)),
///                 eq(Term::<u8>::Var(d), Term::<u8>::Value(2)),
///             ),
///         ),
///     )
///     .apply(state)
/// });
///
/// let stream = goal.apply(State::empty());
/// assert_eq!(vec![Term::Value(1)], stream.deep_walk(&Term::Var(a)));
/// ```
pub fn deep_walk<T>(stream: &Stream<T>, term: &Term<T>) -> Vec<Term<T>>
where
    T: VarRepresentable,
{
    DeepWalkable::deep_walk(stream, term)
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
        (Term::Cons(lh, lt), Term::Cons(rh, rt)) => unify(&mapping, lh.as_ref(), rh.as_ref())
            .and_then(|mapping| unify(&mapping, lt.as_ref(), rt.as_ref())),
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
pub struct Fresh<T, F>
where
    T: ValueRepresentable,
    F: Fn(State<T>) -> Stream<T>,
{
    _value_kind: std::marker::PhantomData<T>,
    func: F,
}

impl<T, F> Fresh<T, F>
where
    T: ValueRepresentable,
    F: Fn(State<T>) -> Stream<T>,
{
    pub fn new(func: F) -> Self {
        Self {
            _value_kind: std::marker::PhantomData,
            func,
        }
    }
}

impl<T, F> Goal<T> for Fresh<T, F>
where
    T: ValueRepresentable,
    F: Fn(State<T>) -> Stream<T>,
{
    fn apply(&self, state: State<T>) -> Stream<T> {
        (self.func).apply(state)
    }
}

/// Declares a new variable.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let goal = fresh(|mut state| {
///     let a = state.declare('a');
///
///     eq(Term::<u8>::Var(a), Term::<u8>::Value(1)).apply(state)
/// });
///
/// let stream = goal.apply(State::empty());
/// assert!(stream.len() == 1);
/// assert_eq!(1, stream[0].as_ref().len(), "{:?}", stream[0]);
/// assert_eq!(
///     Term::Value(1),
///     stream[0].as_ref().walk(&Term::Var('a'.to_var_repr(0)))
/// );
/// ```
pub fn fresh<T, F>(func: F) -> impl Goal<T>
where
    T: ValueRepresentable,
    F: Fn(State<T>) -> Stream<T>,
{
    Fresh::new(func)
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

pub struct Disjunction<T, G1, G2>
where
    T: ValueRepresentable,
    G1: Goal<T>,
    G2: Goal<T>,
{
    _value_kind: std::marker::PhantomData<T>,
    goal1: G1,
    goal2: G2,
}

impl<T, G1, G2> Disjunction<T, G1, G2>
where
    T: ValueRepresentable,
    G1: Goal<T>,
    G2: Goal<T>,
{
    pub fn new(goal1: G1, goal2: G2) -> Self {
        Self {
            _value_kind: std::marker::PhantomData,
            goal1,
            goal2,
        }
    }
}

impl<T, G1, G2> Goal<T> for Disjunction<T, G1, G2>
where
    T: ValueRepresentable,
    G1: Goal<T>,
    G2: Goal<T>,
{
    fn apply(&self, state: State<T>) -> Stream<T> {
        let mut stream_head = self.goal1.apply(state.clone());
        let mut stream_tail = self.goal2.apply(state);

        stream_head.append(&mut stream_tail);

        stream_head
    }
}

/// Creates the disjunction of two goals. Returning all states that are valid
/// in either goal. Logically similar to an `or`.
///
/// # Examples
/// ```
/// use kannery::*;
///
/// let goal = fresh(|mut state| {
///     let a = state.declare('a');
///
///     disjunction(
///         eq(Term::<u8>::Var(a), Term::<u8>::Value(1)),
///         disjunction(
///             eq(Term::<u8>::Var(a), Term::<u8>::Value(2)),
///             eq(Term::<u8>::Var(a), Term::<u8>::Value(3)),
///         ),
///     )
///     .apply(state)
/// });
///
/// let stream = goal.apply(State::empty());
/// assert!(stream.len() == 3);
/// ```
pub fn disjunction<T>(goal1: impl Goal<T>, goal2: impl Goal<T>) -> impl Goal<T>
where
    T: ValueRepresentable,
{
    Disjunction::new(goal1, goal2)
}

pub struct Conjunction<T, G1, G2>
where
    T: ValueRepresentable,
    G1: Goal<T>,
    G2: Goal<T>,
{
    _value_kind: std::marker::PhantomData<T>,
    goal1: G1,
    goal2: G2,
}

impl<T, G1, G2> Conjunction<T, G1, G2>
where
    T: ValueRepresentable,
    G1: Goal<T>,
    G2: Goal<T>,
{
    pub fn new(goal1: G1, goal2: G2) -> Self {
        Self {
            _value_kind: std::marker::PhantomData,
            goal1,
            goal2,
        }
    }
}

impl<T, G1, G2> Goal<T> for Conjunction<T, G1, G2>
where
    T: ValueRepresentable,
    G1: Goal<T>,
    G2: Goal<T>,
{
    fn apply(&self, state: State<T>) -> Stream<T> {
        let stream = self.goal1.apply(state);

        stream
            .into_iter()
            .flat_map(|state| self.goal2.apply(state))
            .collect::<Vec<_>>()
    }
}

/// Creates the conjunction of two goals, flattening the state for all states
/// that are valid in both goals. Logically similar to an `and`.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let goal = fresh(|mut state| {
///     let a = state.declare('a');
///     let b = state.declare('b');
///     let c = state.declare('c');
///
///     conjunction(
///         eq(Term::<u8>::Var(a), Term::<u8>::Value(1)),
///         conjunction(
///             eq(Term::<u8>::Var(b), Term::<u8>::Value(2)),
///             eq(Term::<u8>::Var(c), Term::<u8>::Value(3)),
///         ),
///     )
///     .apply(state)
/// });
///
/// let stream = goal.apply(State::empty());
/// assert!(stream.len() == 1);
/// ```
pub fn conjunction<T>(goal1: impl Goal<T>, goal2: impl Goal<T>) -> impl Goal<T>
where
    T: ValueRepresentable,
{
    Conjunction::new(goal1, goal2)
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let goal = fresh(|mut state| {
            let a = state.declare('a');
            let b = state.declare('b');
            let c = state.declare('c');
            let d = state.declare('d');

            conjunction(
                eq(Term::<u8>::Var(a), Term::<u8>::Var(b)),
                conjunction(
                    eq(Term::<u8>::Var(b), Term::<u8>::Var(c)),
                    conjunction(
                        eq(Term::<u8>::Var(c), Term::<u8>::Value(1)),
                        eq(Term::<u8>::Var(d), Term::<u8>::Value(2)),
                    ),
                ),
            )
            .apply(state)
        });

        let stream = goal.apply(State::empty());
        assert!(stream.len() == 1);
        assert_eq!(4, stream[0].as_ref().len(), "{:?}", stream[0]);
        assert_eq!(Term::Value(1), stream[0].as_ref().walk(&Term::Var(a)));
    }

    #[test]
    fn should_evaluate_nested_fresh_calls() {
        let goal = fresh(|mut state| {
            let _a = state.declare('a');

            fresh(|mut state| {
                let a = state.declare('a');

                disjunction(
                    eq(Term::<u8>::Var(a), Term::<u8>::Value(1)),
                    disjunction(
                        eq(Term::<u8>::Var(a), Term::<u8>::Value(2)),
                        eq(Term::<u8>::Var(a), Term::<u8>::Value(3)),
                    ),
                )
                .apply(state)
            })
            .apply(state)
        });

        let stream = goal.apply(State::empty());
        assert!(stream.len() == 3);

        // should contain 2 vars for `'a'` in the first state.
        assert_eq!(
            stream[0].get_vars_by_key('a').map(|vars| vars.len()),
            Some(2)
        );
    }

    #[test]
    #[ignore = "not yet implemented"]
    fn should_return_multiple_relations() {
        let goal = fresh(|mut state| {
            let homer = state.declare("Homer");

            // children
            let bart = state.declare("Bart");
            let lisa = state.declare("Lisa");

            conjunction(
                eq(Term::Var(homer), Term::Var(bart)),
                conjunction(
                    eq(Term::Var(homer), Term::Var(lisa)),
                    conjunction(
                        fresh(move |state| eq(Term::Var(bart), Term::Value("Bart")).apply(state)),
                        fresh(move |state| eq(Term::Var(lisa), Term::Value("Lisa")).apply(state)),
                    ),
                ),
            )
            .apply(state)
        });

        let stream = goal.apply(State::<&'static str>::empty());

        let res = stream.deep_walk(&Term::Var("Homer".to_var_repr(0)));
        assert_eq!(res.len(), 2)
    }
}
