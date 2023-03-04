use super::*;

trait Unpackable<O> {
    fn unpack(&self) -> O;
}

// Identity function
impl<T: ValueRepresentable> Unpackable<Term<T>> for Term<T> {
    fn unpack(&self) -> Term<T> {
        self.clone()
    }
}

impl<T: ValueRepresentable> Unpackable<Term<T>> for Var {
    fn unpack(&self) -> Term<T> {
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

impl<P1, P2, O1, O2> Unpackable<(O1, O2)> for Join<P1, P2>
where
    P1: Unpackable<O1>,
    P2: Unpackable<O2>,
{
    fn unpack(&self) -> (O1, O2) {
        let lhs = self.lvar.unpack();
        let rhs = self.rvar.unpack();

        (lhs, rhs)
    }
}

trait Runnable<OV, T>
where
    T: ValueRepresentable,
{
    fn run(&self) -> (OV, Stream<T>);
}

pub struct Query<T, V, GF>
where
    T: ValueRepresentable,
{
    vars: V,
    state: State<T>,

    goal_fn: GF,
}

impl<T, V, GF> Query<T, V, GF>
where
    T: ValueRepresentable,
{
    pub fn new(vars: V, state: State<T>, goal_fn: GF) -> Self {
        Self {
            vars,
            goal_fn,
            state,
        }
    }

    pub fn with_vars<NV>(self, new_var_repr: NV) -> Query<T, Join<V, Var>, GF>
    where
        NV: VarRepresentable + std::fmt::Display,
    {
        let mut state = self.state;
        let prev_vars = self.vars;
        let goal = self.goal_fn;

        let new_var = state.declare(new_var_repr);
        let joined_vars = Join::new(prev_vars, new_var);

        Query::new(joined_vars, state, goal)
    }
}

impl<OV, T, V, GF, G> Runnable<OV, T> for Query<T, V, GF>
where
    T: ValueRepresentable,
    V: Unpackable<OV>,
    G: Goal<T>,
    GF: Fn(OV) -> G,
{
    fn run(&self) -> (OV, Stream<T>) {
        let vars = self.vars.unpack();
        let goal = (self.goal_fn)(vars);
        let state = self.state.clone();

        let stream = goal.apply(state);

        (self.vars.unpack(), stream)
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

        let (a2, (b2, c2)): (Term<u8>, (_, _)) = joined.unpack();
        assert!(matches!(a2, Term::Var(_)));
        assert!(matches!(b2, Term::Var(_)));
        assert_eq!(c2, c);
    }
}
