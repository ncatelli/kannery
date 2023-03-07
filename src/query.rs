use super::*;

/// A marker trait to identify non-`()` placeholder types.
pub trait IsNonEmptyUnpackable {}

impl<T: ValueRepresentable> IsNonEmptyUnpackable for Term<T> {}

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

#[derive(Debug)]
pub struct Join<P1, P2> {
    lvar: P1,
    rvar: P2,
}

impl<P1, P2> Join<P1, P2> {
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
    pub fn with_var<NV>(self, new_var_repr: NV) -> QueryBuilder<T, Term<T>>
    where
        NV: VarRepresentable + std::fmt::Display,
    {
        let mut state = self.state;

        let new_var = state.declare(new_var_repr);

        QueryBuilder::new(Term::var(new_var), state)
    }

    /// Takes a term and allocates it against the query.
    ///
    /// # Caller assumes
    /// If a `Term::Var` is passed, this variable has already been declared against
    /// the state. If not, use `Query::with_var` instead.
    pub fn with_term(self, term: Term<T>) -> QueryBuilder<T, Term<T>> {
        let state = self.state;

        QueryBuilder::new(term, state)
    }
}

impl<T, V> QueryBuilder<T, V>
where
    T: ValueRepresentable,
    V: IsNonEmptyUnpackable,
{
    pub fn with_var<NV>(self, new_var_repr: NV) -> QueryBuilder<T, Join<V, Term<T>>>
    where
        NV: VarRepresentable + std::fmt::Display,
    {
        let mut state = self.state;
        let prev_vars = self.associated_terms;

        let new_var = state.declare(new_var_repr);
        let new_var_term = Term::var(new_var);
        let joined_vars = Join::new(prev_vars, new_var_term);

        QueryBuilder::new(joined_vars, state)
    }

    /// Takes a term and allocates it against the query.
    ///
    /// # Caller assumes
    /// If a `Term::Var` is passed, this variable has already been declared against
    /// the state. If not, use `Query::with_var` instead.
    pub fn with_term(self, term: Term<T>) -> QueryBuilder<T, Join<V, Term<T>>> {
        let state = self.state;
        let prev_terms = self.associated_terms;

        let joined_vars = Join::new(prev_terms, term);

        QueryBuilder::new(joined_vars, state)
    }
}

impl<T> QueryBuilder<T, ()>
where
    T: ValueRepresentable,
{
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
    pub fn new(associated_terms: V, state: State<T>, goal_fn: G) -> Self {
        Self {
            _associated_terms: associated_terms,
            state,
            goal: goal_fn,
        }
    }

    pub fn run(self) -> Stream<T> {
        let state = self.state;
        let goal = self.goal;

        goal.apply(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_nest_joins() {
        let a = 'a'.to_var_repr(0);
        let b = 'b'.to_var_repr(0);
        let c = Term::value(0u8);

        let first_joined = Join::new(Term::<u8>::var(b), c.clone());
        let joined = Join::new(a, first_joined);

        let (a2, (b2, c2)) = joined.unpack();
        assert!(matches!(a2, Term::Var(_)));
        assert!(matches!(b2, Term::Var(_)));
        assert_eq!(c2, c);
    }

    #[test]
    fn should_be_be_able_to_stack_on_query_builder() {
        let query = QueryBuilder::default()
            .with_var('a')
            .with_var('b')
            .with_term(Term::value(1_u8))
            .build(|((a, b), one)| {
                conjunction(
                    conjunction(equal(b.clone(), one.clone()), equal(Term::value(1), one)),
                    equal(a, b),
                )
            });

        let stream = query.run();
        assert_eq!(stream.len(), 1);
        assert_eq!(stream[0].term_mapping.len(), 2)
    }
}
