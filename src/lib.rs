use std::collections::HashMap;
use std::hash::Hash;

/// A helper macro for creating a `Rc`-wrapped `Term::Value` variant.
#[macro_export]
macro_rules! value_term {
    ($v:expr) => {
        $crate::Term::Value(std::rc::Rc::new($v))
    };
}

/// A helper macro for creating a `Term::Var` variant.
#[macro_export]
macro_rules! var_term {
    ($var:expr) => {
        $crate::Term::Var($var)
    };
}

/// A helper macro for creating a `Term::Cons` variant from two sub-`Term`s.
#[macro_export]
macro_rules! cons_term {
    ($head:expr, $tail:expr) => {
        $crate::Term::Cons(Box::new($head), Box::new($tail))
    };
}

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
    fn new(base: u64, count: usize) -> Self {
        Self { base, count }
    }
}

/// A Term representing either a Value or Variable or list of Terms.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Term<T: ValueRepresentable> {
    /// Term contains a variable.
    Var(Var),
    /// Term contains an `Rc`-wrapped value, to keep clones cheap.
    Value(std::rc::Rc<T>),
    // In place of a traditional cons list.
    Cons(Box<Term<T>>, Box<Term<T>>),
}

impl<T: ValueRepresentable> From<(Term<T>, Term<T>)> for Term<T> {
    fn from((head, tail): (Term<T>, Term<T>)) -> Self {
        cons_term!(head, tail)
    }
}

/// A map representing potentially recursive Variable to Terminal mappings.
type TermMapping<T> = HashMap<Var, Term<T>>;

/// A map representing a Variable to it's string representation.
type ReprMapping = HashMap<Var, String>;

/// A map representing a Variable repr's occurrence count.
type OccurrenceCounter = HashMap<String, usize>;

/// A state object for a given value that stores mappings of relationships
/// between `Term`s.
#[derive(Default, Clone, PartialEq, Eq)]
pub struct State<T: ValueRepresentable> {
    /// tracks the occurrence of a variable of a given representation.
    occurence_counter: OccurrenceCounter,
    /// Mappings of `Term`s relationships to eachother.
    term_mapping: TermMapping<T>,

    /// Mappings of `Var`s to their corresponding formattable representation.
    repr_mapping: ReprMapping,
}

impl<T: ValueRepresentable> State<T> {
    /// Returns an empty State object for a given type.
    pub fn empty() -> Self {
        Self {
            occurence_counter: HashMap::new(),
            term_mapping: HashMap::new(),
            repr_mapping: HashMap::new(),
        }
    }
}

impl<T: ValueRepresentable> State<T> {
    #[must_use]
    fn new(
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
    /// Declare a tracked `Var` occurrence for a given key.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::*;
    ///
    /// let mut state = State::<()>::empty();
    /// let _var = state.declare("x");
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
fn walk<T, M>(mapping: &M, term: &Term<T>) -> Term<T>
where
    T: VarRepresentable,
    M: Walkable<T>,
{
    Walkable::walk(mapping, term)
}

/// A type for defining walking against a set of states.
pub trait DeepWalkable<T: ValueRepresentable>: Clone {
    fn deep_walk(&self, term: &Term<T>) -> Term<T>;
}

impl<T: VarRepresentable> DeepWalkable<T> for TermMapping<T> {
    fn deep_walk(&self, term: &Term<T>) -> Term<T> {
        let term = self.walk(term);

        if let Term::Cons(head, tail) = term {
            let head_ref = self.walk(head.as_ref());
            let tail_ref = self.walk(tail.as_ref());

            cons_term!(head_ref, tail_ref)
        } else {
            term
        }
    }
}

pub trait Runnable<T: ValueRepresentable>: Clone {
    fn run(&self, term: &Term<T>) -> Vec<Term<T>>;
}

impl<T> Runnable<T> for Stream<T>
where
    T: VarRepresentable,
{
    fn run(&self, term: &Term<T>) -> Vec<Term<T>> {
        self.iter()
            .map(|state| {
                let mapping = state.as_ref();
                DeepWalkable::deep_walk(mapping, term)
            })
            .collect()
    }
}

/// Resolve a variable against a set of states, returning all possible values.
///
/// # Examples
pub fn run<T>(stream: &Stream<T>, term: &Term<T>) -> Vec<Term<T>>
where
    T: VarRepresentable,
{
    Runnable::run(stream, term)
}

pub fn unify<T, F>(
    mapping: &TermMapping<T>,
    term1: &Term<T>,
    term2: &Term<T>,
    condition_func: F,
) -> Option<TermMapping<T>>
where
    T: VarRepresentable,
    F: Fn(&Term<T>, &Term<T>) -> bool + Copy,
{
    let t1_target = walk(mapping, term1);
    let t2_target = walk(mapping, term2);
    let mut mapping = mapping.clone();

    match (t1_target, t2_target) {
        (Term::Var(v1), Term::Var(v2)) if v1 == v2 => Some(mapping),
        (Term::Var(v), t) | (t, Term::Var(v)) => {
            mapping.insert(v, t);
            Some(mapping)
        }
        (Term::Cons(lh, lt), Term::Cons(rh, rt)) => {
            unify(&mapping, lh.as_ref(), rh.as_ref(), condition_func)
                .and_then(|mapping| unify(&mapping, lt.as_ref(), rt.as_ref(), condition_func))
        }
        (t1, t2) if condition_func(&t1, &t2) => Some(mapping),
        _ => None,
    }
}

// Represents a calleable stream of states.
pub type Stream<T> = Vec<State<T>>;

/// The Goal trait represents the ability to apply an action to a set of
/// states that can be evaluated against those terms.
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

/// A `Goal` that allocates a new variable of a given representation for use
/// within a relationship mapping.
///
/// # Examples
/// ```
/// use kannery::*;
///
/// let _x_equals = Fresh::new("x", |x| {
///     eq(var_term!(x), value_term!(1))
/// });
/// ```
#[derive(Debug)]
pub struct Fresh<T, V, F, GO>
where
    T: ValueRepresentable,
    GO: Goal<T>,
    V: VarRepresentable,
    F: Fn(Var) -> GO,
{
    _value_kind: std::marker::PhantomData<T>,
    var: V,
    func: F,
}

impl<T, V, F, GO> Fresh<T, V, F, GO>
where
    T: ValueRepresentable,
    V: VarRepresentable,
    GO: Goal<T>,
    F: Fn(Var) -> GO,
{
    /// Declares a variable for a given id before passing it to the passed
    /// function for use in generating a `Goal`.
    ///
    /// # Examples
    /// ```
    /// use kannery::*;
    ///
    /// let _x_equals = Fresh::new("x", |x| {
    ///     eq(var_term!(x), value_term!(1))
    /// });
    /// ```
    pub fn new(var_id: V, func: F) -> Self {
        Self {
            _value_kind: std::marker::PhantomData,
            var: var_id,
            func,
        }
    }
}

impl<T, V, F, GO> Goal<T> for Fresh<T, V, F, GO>
where
    T: ValueRepresentable,
    V: VarRepresentable + std::fmt::Display,
    GO: Goal<T>,
    F: Fn(Var) -> GO,
{
    fn apply(&self, mut state: State<T>) -> Stream<T> {
        let var = state.declare(self.var.clone());
        (self.func)(var).apply(state)
    }
}

/// Instantiates a new variable for use in a goal.
///
/// # Examples
/// ```
/// use kannery::*;
///
/// let _x_equals = fresh("x", |x| {
///     eq(var_term!(x), value_term!(1))
/// });
/// ```
pub fn fresh<T, V, GO>(var: V, func: impl Fn(Var) -> GO) -> impl Goal<T>
where
    T: ValueRepresentable,
    V: VarRepresentable + std::fmt::Display,
    GO: Goal<T>,
{
    Fresh::new(var, func)
}

/// Declares a new variable for use in a goal.
///
/// A shorthand alias to [fresh].
///
/// # Examples
/// ```
/// use kannery::*;
///
/// let _x_equals = declare("x", |x| {
///     eq(var_term!(x), value_term!(1))
/// });
/// ```
pub fn declare<T, V, GO>(var: V, func: impl Fn(Var) -> GO) -> impl Goal<T>
where
    T: ValueRepresentable,
    V: VarRepresentable + std::fmt::Display,
    GO: Goal<T>,
{
    fresh(var, func)
}

/// A `Goal` that maps an equality relationship between two `Term`s.
///
/// # Examples
/// ```
/// use kannery::*;
///
/// let x_equals = fresh('x', |x| {
///     Equal::new(var_term!(x), value_term!(1))
/// });
/// let stream = x_equals.apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
///
/// assert_eq!(res.len(), 1);
/// assert_eq!([value_term!(1)].as_slice(), res.as_slice());
/// ```
#[derive(Debug)]
pub struct Equal<T: ValueRepresentable> {
    term1: Term<T>,
    term2: Term<T>,
}

impl<T: ValueRepresentable> Equal<T> {
    /// Instantiate a new `Equality` relationship between two terms.
    ///
    /// # Examples
    /// ```
    /// use kannery::*;
    ///
    /// let x_equals = fresh('x', |x| {
    ///     Equal::new(var_term!(x), value_term!(1))
    /// });
    /// let stream = x_equals.apply(State::<u8>::empty());
    /// let x_var = 'x'.to_var_repr(0);
    /// let res = stream.run(&var_term!(x_var));
    ///
    /// assert_eq!(res.len(), 1);
    /// assert_eq!([value_term!(1)].as_slice(), res.as_slice());
    /// ```
    pub fn new(term1: Term<T>, term2: Term<T>) -> Self {
        Self { term1, term2 }
    }
}

impl<T: VarRepresentable> Goal<T> for Equal<T> {
    fn apply(&self, state: State<T>) -> Stream<T> {
        let term1 = &self.term1;
        let term2 = &self.term2;

        let sub_mapping = state.as_ref();
        let unified_mapping = unify(sub_mapping, term1, term2, |lhs, rhs| lhs == rhs);

        // Return an empty stream if the mapping is `None`.
        if let Some(term_mapping) = unified_mapping {
            vec![State::new(
                state.occurence_counter,
                term_mapping,
                state.repr_mapping,
            )]
        } else {
            Stream::new()
        }
    }
}

/// Defines an equality relationship mapping between two `Term`s.
///
/// # Examples
/// ```
/// use kannery::*;
///
/// let x_equals = fresh('x', |x| {
///     equal(var_term!(x), value_term!(1))
/// });
/// let stream = x_equals.apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
///
/// assert_eq!(res.len(), 1);
/// assert_eq!([value_term!(1)].as_slice(), res.as_slice());
/// ```
pub fn equal<T>(term1: Term<T>, term2: Term<T>) -> impl Goal<T>
where
    T: ValueRepresentable,
    Equal<T>: Goal<T>,
{
    Equal::new(term1, term2)
}

/// Defines an equality relationship mapping between two `Term`s.
///
/// A shorthand alias to [equal].
///
/// # Examples
/// ```
/// use kannery::*;
///
/// let x_equals = fresh('x', |x| {
///     eq(var_term!(x), value_term!(1))
/// });
/// let stream = x_equals.apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
///
/// assert_eq!(res.len(), 1);
/// assert_eq!([value_term!(1)].as_slice(), res.as_slice());
/// ```
pub fn eq<T>(term1: Term<T>, term2: Term<T>) -> impl Goal<T>
where
    T: ValueRepresentable,
    Equal<T>: Goal<T>,
{
    equal(term1, term2)
}

/// A `Goal` that maps an less-than relationship between two `Term`s.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let x_is_less_than = |max: u8| {
///     fresh('x', move |x| {
///         conjunction(
///             disjunction(
///                 equal(var_term!(x), value_term!(1)),
///                 equal(var_term!(x), value_term!(2)),
///             ),
///             LessThan::new(var_term!(x), value_term!(max)),
///         )
///     })
/// };
/// let stream = x_is_less_than(2).apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
/// assert_eq!(res.len(), 1);
/// assert_eq!([value_term!(1)].as_slice(), res.as_slice());
/// ```
#[derive(Debug)]
pub struct LessThan<T: ValueRepresentable + PartialOrd> {
    term1: Term<T>,
    term2: Term<T>,
}

impl<T: ValueRepresentable + PartialOrd> LessThan<T> {
    /// Instantiate a new `<` relationship between two terms.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::*;
    ///
    /// let x_is_less_than = |max: u8| {
    ///     fresh('x', move |x| {
    ///         conjunction(
    ///             disjunction(
    ///                 equal(var_term!(x), value_term!(1)),
    ///                 equal(var_term!(x), value_term!(2)),
    ///             ),
    ///             LessThan::new(var_term!(x), value_term!(max)),
    ///         )
    ///     })
    /// };
    /// let stream = x_is_less_than(2).apply(State::<u8>::empty());
    /// let x_var = 'x'.to_var_repr(0);
    /// let res = stream.run(&var_term!(x_var));
    /// assert_eq!(res.len(), 1);
    /// assert_eq!([value_term!(1)].as_slice(), res.as_slice());
    /// ```
    pub fn new(term1: Term<T>, term2: Term<T>) -> Self {
        Self { term1, term2 }
    }
}

impl<T> Goal<T> for LessThan<T>
where
    T: VarRepresentable + PartialOrd,
{
    fn apply(&self, state: State<T>) -> Stream<T> {
        let term1 = &self.term1;
        let term2 = &self.term2;

        let sub_mapping = state.as_ref();
        let unified_mapping = unify(sub_mapping, term1, term2, |lhs, rhs| lhs < rhs);

        // Return an empty stream if the mapping is `None`.
        if let Some(term_mapping) = unified_mapping {
            vec![State::new(
                state.occurence_counter,
                term_mapping,
                state.repr_mapping,
            )]
        } else {
            Stream::new()
        }
    }
}

/// Defines an less-than `<` relationship mapping between two `Term`s.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let x_is_less_than = |max: u8| {
///     fresh('x', move |x| {
///         conjunction(
///             disjunction(
///                 equal(var_term!(x), value_term!(1)),
///                 equal(var_term!(x), value_term!(2)),
///             ),
///             less_than(var_term!(x), value_term!(max)),
///         )
///     })
/// };
/// let stream = x_is_less_than(2).apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
/// assert_eq!(res.len(), 1);
/// assert_eq!([value_term!(1)].as_slice(), res.as_slice());
/// ```
pub fn less_than<T>(term1: Term<T>, term2: Term<T>) -> impl Goal<T>
where
    T: ValueRepresentable + PartialOrd,
    LessThan<T>: Goal<T>,
{
    LessThan::new(term1, term2)
}

/// A `Goal` that maps an greater-than relationship between two `Term`s.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let x_is_greater_than = |min: u8| {
///     fresh('x', move |x| {
///         conjunction(
///             disjunction(
///                 equal(var_term!(x), value_term!(1)),
///                 equal(var_term!(x), value_term!(2)),
///             ),
///             GreaterThan::new(var_term!(x), value_term!(min)),
///         )
///     })
/// };
/// let stream = x_is_greater_than(1).apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
/// assert_eq!(res.len(), 1);
/// assert_eq!([value_term!(2)].as_slice(), res.as_slice());
/// ```
#[derive(Debug)]
pub struct GreaterThan<T: ValueRepresentable + PartialOrd> {
    term1: Term<T>,
    term2: Term<T>,
}

impl<T: ValueRepresentable + PartialOrd> GreaterThan<T> {
    /// Instantiate a new `>` relationship between two terms.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::*;
    ///
    /// let x_is_greater_than = |min: u8| {
    ///     fresh('x', move |x| {
    ///         conjunction(
    ///             disjunction(
    ///                 equal(var_term!(x), value_term!(1)),
    ///                 equal(var_term!(x), value_term!(2)),
    ///             ),
    ///             GreaterThan::new(var_term!(x), value_term!(min)),
    ///         )
    ///     })
    /// };
    /// let stream = x_is_greater_than(1).apply(State::<u8>::empty());
    /// let x_var = 'x'.to_var_repr(0);
    /// let res = stream.run(&var_term!(x_var));
    /// assert_eq!(res.len(), 1);
    /// assert_eq!([value_term!(2)].as_slice(), res.as_slice());
    /// ```
    pub fn new(term1: Term<T>, term2: Term<T>) -> Self {
        Self { term1, term2 }
    }
}

impl<T> Goal<T> for GreaterThan<T>
where
    T: VarRepresentable + PartialOrd,
{
    fn apply(&self, state: State<T>) -> Stream<T> {
        let term1 = &self.term1;
        let term2 = &self.term2;

        let sub_mapping = state.as_ref();
        let unified_mapping = unify(sub_mapping, term1, term2, |lhs, rhs| lhs > rhs);

        // Return an empty stream if the mapping is `None`.
        if let Some(term_mapping) = unified_mapping {
            vec![State::new(
                state.occurence_counter,
                term_mapping,
                state.repr_mapping,
            )]
        } else {
            Stream::new()
        }
    }
}

/// Defines an greater-than `>` relationship mapping between two `Term`s.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let x_is_greater_than = |min: u8| {
///     fresh('x', move |x| {
///         conjunction(
///             disjunction(
///                 equal(var_term!(x), value_term!(1)),
///                 equal(var_term!(x), value_term!(2)),
///             ),
///             greater_than(var_term!(x), value_term!(min)),
///         )
///     })
/// };
/// let stream = x_is_greater_than(1).apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
/// assert_eq!(res.len(), 1);
/// assert_eq!([value_term!(2)].as_slice(), res.as_slice());
/// ```
pub fn greater_than<T>(term1: Term<T>, term2: Term<T>) -> impl Goal<T>
where
    T: ValueRepresentable + PartialOrd,
    GreaterThan<T>: Goal<T>,
{
    GreaterThan::new(term1, term2)
}

/// A `Goal` that maps a disjunction relationship between two `Term`s.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let x_equals = fresh('x', |x| {
///     Disjunction::new(
///         equal(var_term!(x), value_term!(1)),
///         equal(var_term!(x), value_term!(2))
///     )
/// });
/// let stream = x_equals.apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
///
/// assert_eq!(res.len(), 2);
/// assert_eq!([value_term!(1), value_term!(2)].as_slice(), res.as_slice());
/// ```
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
    /// Instantiate a new `Disjunction` relationship between two terms.
    ///
    /// # Examples
    /// ```
    /// use kannery::*;
    ///
    /// let _x_equals = Disjunction::new(
    ///     equal(value_term!(1), value_term!(1)),
    ///     equal(value_term!(2), value_term!(2))
    /// );
    /// ```
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
///
/// ```
/// use kannery::*;
///
/// let x_equals = fresh('x', |x| {
///     disjunction(
///         equal(var_term!(x), value_term!(1)),
///         equal(var_term!(x), value_term!(2))
///     )
/// });
/// let stream = x_equals.apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
///
/// assert_eq!(res.len(), 2);
/// assert_eq!([value_term!(1), value_term!(2)].as_slice(), res.as_slice());
/// ```
pub fn disjunction<T>(goal1: impl Goal<T>, goal2: impl Goal<T>) -> impl Goal<T>
where
    T: ValueRepresentable,
{
    Disjunction::new(goal1, goal2)
}

/// Returns the states that are true of either of two goals. Equivalent to a
/// logical or.
///
/// A shorthand alias to [disjunction].
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let x_equals = declare('x', |x| {
///     either(
///         equal(var_term!(x), value_term!(1)),
///         equal(var_term!(x), value_term!(2))
///     )
/// });
/// let stream = x_equals.apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
///
/// assert_eq!(res.len(), 2);
/// assert_eq!([value_term!(1), value_term!(2)].as_slice(), res.as_slice());
/// ```
pub fn either<T>(goal1: impl Goal<T>, goal2: impl Goal<T>) -> impl Goal<T>
where
    T: ValueRepresentable,
{
    disjunction(goal1, goal2)
}

/// A `Goal` that maps a conjunction relationship between two `Term`s.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let x_equals = fresh('x', |x| {
///     fresh('y', move |y| {
///         Conjunction::new(
///             equal(var_term!(x), var_term!(y)),
///             equal(var_term!(y), value_term!(1))
///         )
///     })
/// });
/// let stream = x_equals.apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
///
/// assert_eq!(res.len(), 1);
/// assert_eq!([value_term!(1)].as_slice(), res.as_slice());
/// ```
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

/// A `Goal` that maps a conjunction relationship between two `Term`s.
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let x_equals = fresh('x', |x| {
///     fresh('y', move |y| {
///         conjunction(
///             equal(var_term!(x), var_term!(y)),
///             equal(var_term!(y), value_term!(1))
///         )
///     })
/// });
/// let stream = x_equals.apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
///
/// assert_eq!(res.len(), 1);
/// assert_eq!([value_term!(1)].as_slice(), res.as_slice());
/// ```
pub fn conjunction<T>(goal1: impl Goal<T>, goal2: impl Goal<T>) -> impl Goal<T>
where
    T: ValueRepresentable,
{
    Conjunction::new(goal1, goal2)
}

/// Returns the states that are true of both of two goals. Equivalent to a
/// logical and.
///
/// A shorthand alias to [conjunction].
///
/// # Examples
///
/// ```
/// use kannery::*;
///
/// let x_equals = declare('x', |x| {
///     declare('y', move |y| {
///         both(
///             equal(var_term!(x), var_term!(y)),
///             equal(var_term!(y), value_term!(1))
///         )
///     })
/// });
/// let stream = x_equals.apply(State::<u8>::empty());
/// let x_var = 'x'.to_var_repr(0);
/// let res = stream.run(&var_term!(x_var));
///
/// assert_eq!(res.len(), 1);
/// assert_eq!([value_term!(1)].as_slice(), res.as_slice());
/// ```
pub fn both<T>(goal1: impl Goal<T>, goal2: impl Goal<T>) -> impl Goal<T>
where
    T: ValueRepresentable,
{
    conjunction(goal1, goal2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_return_multiple_relations() {
        let parent_fn = |parent: Term<_>, child: Term<_>| {
            let homer = value_term!("Homer");
            let marge = value_term!("Marge");
            let bart = value_term!("Bart");
            let lisa = value_term!("Lisa");
            let abe = value_term!("Abe");
            let jackie = value_term!("Jackie");

            disjunction(
                eq(
                    cons_term!(parent.clone(), child.clone()),
                    cons_term!(homer.clone(), bart.clone()),
                ),
                disjunction(
                    eq(
                        cons_term!(parent.clone(), child.clone()),
                        cons_term!(homer.clone(), lisa.clone()),
                    ),
                    disjunction(
                        eq(
                            cons_term!(parent.clone(), child.clone()),
                            cons_term!(marge.clone(), bart),
                        ),
                        disjunction(
                            eq(
                                cons_term!(parent.clone(), child.clone()),
                                cons_term!(marge.clone(), lisa),
                            ),
                            disjunction(
                                eq(
                                    cons_term!(parent.clone(), child.clone()),
                                    cons_term!(abe, homer),
                                ),
                                eq(cons_term!(parent, child), cons_term!(jackie, marge)),
                            ),
                        ),
                    ),
                ),
            )
        };

        let sorted_value_strings = |res: Vec<Term<&str>>| {
            let mut elements = res
                .into_iter()
                .flat_map(|term| match term {
                    Term::Value(val) => Some(val.to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>();

            elements.sort();
            elements
        };

        let children_of_homer = fresh("child", |child| {
            parent_fn(value_term!("Homer"), var_term!(child))
        });
        let stream = children_of_homer.apply(State::empty());
        let child_var = "child".to_var_repr(0);
        let res = stream.run(&var_term!(child_var));

        assert_eq!(stream.len(), 2, "{:?}", res);
        let sorted_children = sorted_value_strings(res);
        assert_eq!(
            ["Bart".to_string(), "Lisa".to_string()].as_slice(),
            sorted_children.as_slice()
        );

        // map parent relationship
        let parents_of_lisa = fresh("parent", |parent| {
            parent_fn(var_term!(parent), value_term!("Lisa"))
        });
        let stream = parents_of_lisa.apply(State::empty());
        let parent_var = "parent".to_var_repr(0);
        let res = stream.run(&Term::Var(parent_var));

        assert_eq!(stream.len(), 2, "{:?}", res);
        let sorted_parents = sorted_value_strings(res);

        assert_eq!(
            ["Homer".to_string(), "Marge".to_string()].as_slice(),
            sorted_parents.as_slice()
        );
    }

    #[test]
    fn should_define_relations_without_fresh() {
        let parent_fn = |parent: Term<_>, child: Term<_>| {
            let homer = value_term!("Homer");
            let marge = value_term!("Marge");
            let bart = value_term!("Bart");
            let lisa = value_term!("Lisa");
            let abe = value_term!("Abe");
            let jackie = value_term!("Jackie");

            disjunction(
                eq(
                    cons_term!(parent.clone(), child.clone()),
                    cons_term!(homer.clone(), bart.clone()),
                ),
                disjunction(
                    eq(
                        cons_term!(parent.clone(), child.clone()),
                        cons_term!(homer.clone(), lisa.clone()),
                    ),
                    disjunction(
                        eq(
                            cons_term!(parent.clone(), child.clone()),
                            cons_term!(marge.clone(), bart),
                        ),
                        disjunction(
                            eq(
                                cons_term!(parent.clone(), child.clone()),
                                cons_term!(marge.clone(), lisa),
                            ),
                            disjunction(
                                eq(
                                    cons_term!(parent.clone(), child.clone()),
                                    cons_term!(abe, homer),
                                ),
                                eq(cons_term!(parent, child), cons_term!(jackie, marge)),
                            ),
                        ),
                    ),
                ),
            )
        };

        let sorted_value_strings = |res: Vec<Term<&str>>| {
            let mut elements = res
                .into_iter()
                .flat_map(|term| match term {
                    Term::Value(val) => Some(val.to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>();

            elements.sort();
            elements
        };

        let mut state = State::empty();
        let child = state.declare("child");
        let children_of_homer = || parent_fn(value_term!("Homer"), var_term!(child));
        let stream = children_of_homer().apply(state);
        let res = stream.run(&Term::Var(child));

        assert_eq!(stream.len(), 2, "{:?}", res);
        let sorted_children = sorted_value_strings(res);
        assert_eq!(
            ["Bart".to_string(), "Lisa".to_string()].as_slice(),
            sorted_children.as_slice()
        );

        // map parent relationship
        let mut state = State::empty();
        let parent = state.declare("parent");
        let parents_of_lisa = parent_fn(var_term!(parent), value_term!("Lisa"));
        let stream = parents_of_lisa.apply(state);
        let res = stream.run(&var_term!(parent));

        assert_eq!(stream.len(), 2, "{:?}", res);
        let sorted_parents = sorted_value_strings(res);

        assert_eq!(
            ["Homer".to_string(), "Marge".to_string()].as_slice(),
            sorted_parents.as_slice()
        );
    }
}
