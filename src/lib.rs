use std::collections::HashMap;
use std::hash::Hash;

#[macro_export]
macro_rules! value {
    ($v:expr) => {
        Term::Value($v)
    };
}

#[macro_export]
macro_rules! var {
    ($var:expr) => {
        Term::Var($var)
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
    /// Dedeclare a tracked `Var` occurrence for a given key.
    fn declare<VAR: VarRepresentable + std::fmt::Display>(&mut self, key: VAR) -> Var {
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
///
/// # Examples
///
pub fn walk<T, M>(mapping: &M, term: &Term<T>) -> Term<T>
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
            Term::Cons(Box::new(head_ref), Box::new(tail_ref))
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
///
pub fn run<T>(stream: &Stream<T>, term: &Term<T>) -> Vec<Term<T>>
where
    T: VarRepresentable,
{
    Runnable::run(stream, term)
}

fn unify<T: VarRepresentable>(
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
        (Term::Cons(lh, lt), Term::Cons(rh, rt)) => unify(&mapping, lh.as_ref(), rh.as_ref())
            .and_then(|mapping| unify(&mapping, lt.as_ref(), rt.as_ref())),
        (t1, t2) if t1 == t2 => Some(mapping),
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

pub fn fresh<T, V, GO>(var: V, func: impl Fn(Var) -> GO) -> impl Goal<T>
where
    T: ValueRepresentable,
    V: VarRepresentable + std::fmt::Display,
    GO: Goal<T>,
{
    Fresh::new(var, func)
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
        let term1 = &self.term1;
        let term2 = &self.term2;

        let sub_mapping = state.as_ref();
        let unified_mapping = unify(sub_mapping, term1, term2);

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
///
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
    fn should_return_multiple_relations() {
        let parent_fn = |parent: Term<_>, child: Term<_>| {
            disjunction(
                eq(
                    Term::Cons(Box::new(parent.clone()), Box::new(child.clone())),
                    Term::Cons(Box::new(value!("Homer")), Box::new(value!("Bart"))),
                ),
                disjunction(
                    eq(
                        Term::Cons(Box::new(parent.clone()), Box::new(child.clone())),
                        Term::Cons(Box::new(value!("Homer")), Box::new(value!("Lisa"))),
                    ),
                    disjunction(
                        eq(
                            Term::Cons(Box::new(parent.clone()), Box::new(child.clone())),
                            Term::Cons(Box::new(value!("Marge")), Box::new(value!("Bart"))),
                        ),
                        disjunction(
                            eq(
                                Term::Cons(Box::new(parent.clone()), Box::new(child.clone())),
                                Term::Cons(Box::new(value!("Marge")), Box::new(value!("Lisa"))),
                            ),
                            disjunction(
                                eq(
                                    Term::Cons(Box::new(parent.clone()), Box::new(child.clone())),
                                    Term::Cons(Box::new(value!("Abe")), Box::new(value!("Homer"))),
                                ),
                                eq(
                                    Term::Cons(Box::new(parent), Box::new(child)),
                                    Term::Cons(
                                        Box::new(value!("Jackie")),
                                        Box::new(value!("Marge")),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            )
        };

        let children_of_homer = || {
            fresh("child", move |child| {
                parent_fn(value!("Homer"), var!(child))
            })
        };
        let stream = children_of_homer().apply(State::empty());
        let child_var = "child".to_var_repr(0);
        let res = stream.run(&Term::Var(child_var));

        assert_eq!(stream.len(), 2, "{:?}", res);
        let sorted_children = {
            let mut children = res
                .into_iter()
                .flat_map(|term| match term {
                    Term::Value(val) => Some(val.to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>();

            children.sort();
            children
        };

        assert_eq!(
            ["Bart".to_string(), "Lisa".to_string()].as_slice(),
            sorted_children.as_slice()
        );
    }
}
