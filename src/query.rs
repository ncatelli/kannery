use super::*;

trait VarPackable<B> {
    fn pack(&self) -> B;
}

// functions as an identity function.
impl VarPackable<Var> for Var {
    fn pack(&self) -> Var {
        *self
    }
}

#[derive(Debug)]
pub struct Join<V1, V2> {
    lvar: V1,
    rvar: V2,
}

impl<V1, V2> Join<V1, V2> {
    fn new(lvar: V1, rvar: V2) -> Self {
        Self { lvar, rvar }
    }
}

impl<V1, V2, O1, O2> VarPackable<(O1, O2)> for Join<V1, V2>
where
    V1: VarPackable<O1>,
    V2: VarPackable<O2>,
{
    fn pack(&self) -> (O1, O2) {
        let lhs = self.lvar.pack();
        let rhs = self.rvar.pack();

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

    goal: GF,
}

impl<T, V, GF> Query<T, V, GF>
where
    T: ValueRepresentable,
{
    pub fn new(vars: V, state: State<T>, goal: GF) -> Self {
        Self { vars, goal, state }
    }

    pub fn with_vars<NV>(self, new_var_repr: NV) -> Query<T, Join<V, Var>, GF>
    where
        NV: VarRepresentable + std::fmt::Display,
    {
        let mut state = self.state;
        let prev_vars = self.vars;
        let goal = self.goal;

        let new_var = state.declare(new_var_repr);
        let joined_vars = Join::new(prev_vars, new_var);

        Query::new(joined_vars, state, goal)
    }
}

impl<OV, T, V, GF, G> Runnable<OV, T> for Query<T, V, GF>
where
    T: ValueRepresentable,
    V: VarPackable<OV>,
    G: Goal<T>,
    GF: Fn(OV) -> G,
{
    fn run(&self) -> (OV, Stream<T>) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_nest_joins() {
        let a = 'a'.to_var_repr(0);
        let b = 'b'.to_var_repr(0);
        let c = 'c'.to_var_repr(0);

        let first_joined = Join::new(b, c);
        let _joined = Join::new(a, first_joined);
    }
}
