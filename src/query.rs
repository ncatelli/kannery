//! The query module provides an extension on top of the kanren engine for
//! generating queries and relations using a more simple builder pattern.
//!
//! # Examples
//!
//!  ```
//! use kannery::prelude::v1::*;
//! use kannery::*;
//!
//! let query = QueryBuilder::default()
//!     .with_var('a')
//!     .with_var('b')
//!     .with_term(Term::value(1_u8))
//!     .build(|((a, b), one)| {
//!         conjunction(
//!             conjunction(equal(b.clone(), one.clone()), equal(Term::value(1), one)),
//!             equal(a, b),
//!         )
//!     });
//!
//! let result = query.run();
//! let a_values = result.owned_values_of('a');
//! let b_values = result.owned_values_of('b');
//!
//! // assert all values of a == 1.
//! assert!(a_values.into_iter().all(|val| val == 1_u8));
//!
//! // assert all values of b == 1.
//! assert!(b_values.into_iter().all(|val| val == 1_u8))
//! ```

use std::collections::HashSet;
use std::rc;

use super::*;

/// A marker trait to identify non-`()` placeholder types.
pub trait IsNonEmptyUnpackable {}

impl<T: ValueRepresentable> IsNonEmptyUnpackable for Term<T> {}

/// A type that allows the conversion of joined pairs to an unpacked
/// representation for building a [Goal].
pub trait Unpackable<O> {
    type ValueKind;

    fn unpack(&self) -> O;
}

// Identity function
impl<T: ValueRepresentable> Unpackable<Term<T>> for Term<T> {
    type ValueKind = T;

    fn unpack(&self) -> Term<Self::ValueKind> {
        self.clone()
    }
}

impl<T: ValueRepresentable> Unpackable<Term<T>> for Var {
    type ValueKind = T;

    fn unpack(&self) -> Term<Self::ValueKind> {
        Term::var(*self)
    }
}

/// Represents a Variable for association to a [Query].
#[derive(Debug)]
pub struct AssociatedVar(Var);

impl AssociatedVar {
    /// Instantiate a new AssociatedVar.
    fn new(var: Var) -> Self {
        Self(var)
    }
}

impl IsNonEmptyUnpackable for AssociatedVar {}

impl<T> Unpackable<Term<T>> for AssociatedVar
where
    T: VarRepresentable,
{
    type ValueKind = T;

    fn unpack(&self) -> Term<T> {
        Term::var(self.0)
    }
}

/// Represents a value for association to a [Query].
#[derive(Debug)]
pub struct AssociatedValue<T: ValueRepresentable>(T);

impl<T: ValueRepresentable> AssociatedValue<T> {
    /// Instantiate a new AssociatedValue from a value representable kind.
    #[must_use]
    fn new(value: T) -> Self {
        Self(value)
    }
}

impl<T: ValueRepresentable> IsNonEmptyUnpackable for AssociatedValue<T> {}

impl<T> Unpackable<Term<T>> for AssociatedValue<T>
where
    T: ValueRepresentable,
{
    type ValueKind = T;

    fn unpack(&self) -> Term<T> {
        Term::value(self.0.clone())
    }
}

/// Join functions as an internal type for associating [Term]s passed to a [QueryBuilder].
#[derive(Debug)]
pub struct Join<P1, P2> {
    lvar: P1,
    rvar: P2,
}

impl<P1, P2> Join<P1, P2> {
    #[must_use]
    fn new(lvar: P1, rvar: P2) -> Self {
        Self { lvar, rvar }
    }
}

impl<T, P1, P2, O1, O2> Unpackable<(O1, O2)> for Join<P1, P2>
where
    T: ValueRepresentable,
    P1: Unpackable<O1, ValueKind = T>,
    P2: Unpackable<O2, ValueKind = T>,
{
    type ValueKind = T;

    fn unpack(&self) -> (O1, O2) {
        let lhs = self.lvar.unpack();
        let rhs = self.rvar.unpack();

        (lhs, rhs)
    }
}

impl<P1, P2> IsNonEmptyUnpackable for Join<P1, P2> {}

impl<P1, P2> From<Join<P1, P2>> for (P1, P2) {
    fn from(Join { lvar, rvar }: Join<P1, P2>) -> Self {
        (lvar, rvar)
    }
}

/// QueryBuilder provides methods and types for building a [Query] against a
/// relation from a set of variables.
#[derive(Debug, Clone)]
pub struct QueryBuilder<T, V>
where
    T: ValueRepresentable,
{
    associated_terms: V,
    state: State<T>,
}

impl<T, V> QueryBuilder<T, V>
where
    T: ValueRepresentable,
{
    /// Instantiates a new [QueryBuilder] from its consituent parts.
    #[must_use]
    pub fn new(vars: V, state: State<T>) -> Self {
        Self {
            associated_terms: vars,
            state,
        }
    }
}

impl<T> QueryBuilder<T, ()>
where
    T: ValueRepresentable,
{
    /// Takes a [Term] and allocates it for use in a [Goal] build.
    ///
    /// # Caller assumes
    /// If a `Term::Var` is passed, this variable has already been declared against
    /// the state. If not, use `Query::with_var` instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::prelude::v1::*;
    /// use kannery::*;
    ///
    /// let query = QueryBuilder::<u8, _>::default()
    ///     .with_var('a')
    ///     .with_term(Term::value(1));
    /// ```
    pub fn with_term(self, term: Term<T>) -> QueryBuilder<T, Term<T>> {
        let state = self.state;

        QueryBuilder::new(term, state)
    }

    /// Associates a new variable with the `QueryBuilder`.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::prelude::v1::*;
    /// use kannery::*;
    ///
    /// let query = QueryBuilder::<u8, _>::default()
    ///     .with_var('a')
    ///     .with_var('b');
    /// ```
    pub fn with_var<NV>(self, new_var_repr: NV) -> QueryBuilder<T, AssociatedVar>
    where
        NV: VarRepresentable + std::fmt::Display,
    {
        let mut state = self.state;

        let new_var = state.declare(new_var_repr);

        QueryBuilder::new(AssociatedVar::new(new_var), state)
    }

    /// Associates a new value [Term] with the [QueryBuilder].
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::prelude::v1::*;
    /// use kannery::*;
    ///
    /// let query = QueryBuilder::<u8, _>::default()
    ///     .with_var('a')
    ///     .with_value(1);
    /// ```
    pub fn with_value(self, new_val: T) -> QueryBuilder<T, AssociatedValue<T>> {
        let state = self.state;
        QueryBuilder::new(AssociatedValue::new(new_val), state)
    }
}

impl<T, V> QueryBuilder<T, V>
where
    T: ValueRepresentable,
    V: IsNonEmptyUnpackable,
{
    /// Takes a term and allocates it against the query.
    ///
    /// # Caller assumes
    ///
    /// If a `Term::Var` is passed, this variable has already been declared against
    /// the state. If not, use `Query::with_var` instead.
    pub fn with_term(self, term: Term<T>) -> QueryBuilder<T, Join<V, Term<T>>> {
        let state = self.state;
        let prev_terms = self.associated_terms;

        let joined_vars = Join::new(prev_terms, term);

        QueryBuilder::new(joined_vars, state)
    }

    /// Associates a new variable with the `QueryBuilder`.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::prelude::v1::*;
    /// use kannery::*;
    ///
    /// let query = QueryBuilder::<u8, _>::default()
    ///     .with_var('a')
    ///     .with_var('b');
    /// ```
    pub fn with_var<NV>(self, new_var_repr: NV) -> QueryBuilder<T, Join<V, AssociatedVar>>
    where
        NV: VarRepresentable + std::fmt::Display,
    {
        let mut state = self.state;
        let prev_vars = self.associated_terms;

        let new_var = state.declare(new_var_repr);
        let joined_vars = Join::new(prev_vars, AssociatedVar::new(new_var));

        QueryBuilder::new(joined_vars, state)
    }

    /// Associates a new value [Term] with the [QueryBuilder].
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::prelude::v1::*;
    /// use kannery::*;
    ///
    /// let query = QueryBuilder::<u8, _>::default()
    ///     .with_var('a')
    ///     .with_value(1);
    /// ```
    pub fn with_value(self, new_val: T) -> QueryBuilder<T, Join<V, AssociatedValue<T>>> {
        let state = self.state;
        let prev_terms = self.associated_terms;

        let new_associated_val = AssociatedValue::new(new_val);
        let joined_vars = Join::new(prev_terms, new_associated_val);

        QueryBuilder::new(joined_vars, state)
    }
}

impl<T> QueryBuilder<T, ()>
where
    T: ValueRepresentable,
{
    /// Constructs a [Query] from a passed relations.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::prelude::v1::*;
    /// use kannery::*;
    ///
    /// let query = QueryBuilder::<u8, _>::default()
    ///     .with_var('a')
    ///     .with_var('b')
    ///     .build(|(a, b)| equal(a, b));
    /// ```
    pub fn build<G, NGF>(self, new_goal: NGF) -> Query<T, (), G>
    where
        G: Goal<T>,
        NGF: Fn() -> G,
    {
        let state = self.state;

        let goal = new_goal();

        Query::new((), state, goal)
    }
}

impl<T, V> QueryBuilder<T, V>
where
    T: ValueRepresentable,
    V: IsNonEmptyUnpackable,
{
    /// Constructs a [Query] from a passed relations.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::prelude::v1::*;
    /// use kannery::*;
    ///
    /// let query = QueryBuilder::<u8, _>::default()
    ///     .with_var('a')
    ///     .with_var('b')
    ///     .build(|(a, b)| equal(a, b));
    /// ```
    pub fn build<G, UT, NGF>(self, new_goal: NGF) -> Query<T, UT, G>
    where
        G: Goal<T>,
        V: Unpackable<UT, ValueKind = T>,
        NGF: Fn(UT) -> G,
    {
        let state = self.state;

        let associated_terms = self.associated_terms;
        let goal = new_goal(associated_terms.unpack());

        Query::new(associated_terms.unpack(), state, goal)
    }
}

impl<T> Default for QueryBuilder<T, ()>
where
    T: ValueRepresentable,
{
    fn default() -> Self {
        Self {
            associated_terms: (),
            state: State::empty(),
        }
    }
}

/// Provides types for querying the results of a `Query` run for given
/// variables.
#[derive(Debug)]
pub struct QueryResult<T>
where
    T: ValueRepresentable,
{
    stream: Stream<T>,
}

impl<T> QueryResult<T>
where
    T: ValueRepresentable,
{
    #[must_use]
    fn new(stream: Stream<T>) -> Self {
        Self { stream }
    }

    /// Returns reference-counted values matching a given variable representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::prelude::v1::*;
    /// use kannery::*;
    ///
    /// let query = QueryBuilder::default()
    ///     .with_var('a')
    ///     .with_var('b')
    ///     .with_term(Term::value(1_u8))
    ///     .build(|((a, b), one)| {
    ///         conjunction(
    ///             conjunction(equal(b.clone(), one.clone()), equal(Term::value(1), one)),
    ///             equal(a, b),
    ///         )
    ///     });
    ///
    /// let result = query.run();
    /// let a_values = result.values_of('a');
    /// let b_values = result.values_of('b');
    ///
    /// // assert all values of a == 1.
    /// assert!(a_values.into_iter().all(|val| val.as_ref() == &1_u8));
    ///
    /// // assert all values of b == 1.
    /// assert!(b_values.into_iter().all(|val| val.as_ref() == &1_u8))
    /// ```
    pub fn values_of<V>(&self, var: V) -> HashSet<rc::Rc<T>>
    where
        V: VarRepresentable + std::fmt::Display,
        T: Hash,
    {
        let var_repr = var.to_string();
        let terms_matching_var = self
            .stream
            .iter()
            .filter_map(|state| {
                let count = state.occurence_counter.get(&var_repr).copied()?;

                let state_iter = (0..=(count))
                    .into_iter()
                    .flat_map(|occ_count| state.term_mapping.get(&var.to_var_repr(occ_count)));

                Some(state_iter)
            })
            .flatten();

        let values = terms_matching_var.filter_map(|term| match term {
            Term::Value(t) => Some(t.clone()),
            _ => None,
        });

        values.collect()
    }

    /// Returns owned values matching a given variable representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::prelude::v1::*;
    /// use kannery::*;
    ///
    /// let query = QueryBuilder::default()
    ///     .with_var('a')
    ///     .with_var('b')
    ///     .with_term(Term::value(1_u8))
    ///     .build(|((a, b), one)| {
    ///         conjunction(
    ///             conjunction(equal(b.clone(), one.clone()), equal(Term::value(1), one)),
    ///             equal(a, b),
    ///         )
    ///     });
    ///
    /// let result = query.run();
    /// let a_values = result.owned_values_of('a');
    /// let b_values = result.owned_values_of('b');
    ///
    /// // assert all values of a == 1.
    /// assert!(a_values.into_iter().all(|val| val == 1_u8));
    ///
    /// // assert all values of b == 1.
    /// assert!(b_values.into_iter().all(|val| val == 1_u8))
    /// ```
    pub fn owned_values_of<V>(&self, var: V) -> HashSet<T>
    where
        V: VarRepresentable + std::fmt::Display,
        T: Hash,
    {
        use std::borrow::Borrow;

        let var_repr = var.to_string();
        let terms_matching_var = self
            .stream
            .iter()
            .filter_map(|state| {
                let count = state.occurence_counter.get(&var_repr).copied()?;

                let state_iter = (0..=(count))
                    .into_iter()
                    .flat_map(|occ_count| state.term_mapping.get(&var.to_var_repr(occ_count)));

                Some(state_iter)
            })
            .flatten();

        let values = terms_matching_var.filter_map(|term| match term {
            Term::Value(t) => Some((Borrow::<T>::borrow(t)).clone()),
            _ => None,
        });

        values.collect()
    }

    pub fn into_stream(self) -> Stream<T> {
        self.stream
    }
}

impl<T> From<QueryResult<T>> for Stream<T>
where
    T: ValueRepresentable,
{
    fn from(result: QueryResult<T>) -> Self {
        result.into_stream()
    }
}

/// Query consists of an associated `State`, `Term`s and `Goal` that can then
/// be ran to generate a set of matching results.
#[derive(Debug, Clone)]
pub struct Query<T, V, G>
where
    T: ValueRepresentable,
    G: Goal<T>,
{
    _associated_terms: V,
    state: State<T>,

    goal: G,
}

impl<T, V, G> Query<T, V, G>
where
    T: ValueRepresentable,
    G: Goal<T>,
{
    /// Instantiate a new Query from its consituent parts.
    #[must_use]
    pub fn new(associated_terms: V, state: State<T>, goal_fn: G) -> Self {
        Self {
            _associated_terms: associated_terms,
            state,
            goal: goal_fn,
        }
    }

    /// Evaluate a query, returning the results.
    ///
    /// # Examples
    ///
    /// ```
    /// use kannery::prelude::v1::*;
    /// use kannery::*;
    ///
    /// let query = QueryBuilder::default()
    ///     .with_var('a')
    ///     .with_var('b')
    ///     .with_term(Term::value(1_u8))
    ///     .build(|((a, b), one)| {
    ///         conjunction(
    ///             conjunction(equal(b.clone(), one.clone()), equal(Term::value(1), one)),
    ///             equal(a, b),
    ///         )
    ///     });
    ///
    /// let result = query.run();
    /// let a_values = result.owned_values_of('a');
    /// let b_values = result.owned_values_of('b');
    ///
    /// // assert all values of a == 1.
    /// assert!(a_values.into_iter().all(|val| val == 1_u8));
    ///
    /// // assert all values of b == 1.
    /// assert!(b_values.into_iter().all(|val| val == 1_u8))
    /// ```
    pub fn run(self) -> QueryResult<T> {
        let state = self.state;
        let goal = self.goal;

        let stream = goal.apply(state);
        QueryResult::new(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_nest_joins() {
        let a = 'a'.to_var_repr(0);
        let b = 'b'.to_var_repr(0);
        let c = 0u8;

        let first_joined = Join::new(AssociatedVar(b), AssociatedValue(c));
        let joined = Join::new(AssociatedVar(a), first_joined);

        let (a2, (b2, c2)): (Term<u8>, (Term<u8>, Term<u8>)) = joined.unpack();
        assert!(matches!(a2, Term::Var(_)));
        assert!(matches!(b2, Term::Var(_)));
        assert_eq!(c2, Term::value(c));
    }
}
