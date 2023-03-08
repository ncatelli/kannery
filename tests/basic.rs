use kannery::*;

fn parent_child_relationship_goal(
    parent: Term<&'static str>,
    child: Term<&'static str>,
) -> impl Goal<&'static str> {
    let homer = Term::value("Homer");
    let marge = Term::value("Marge");
    let bart = Term::value("Bart");
    let lisa = Term::value("Lisa");
    let abe = Term::value("Abe");
    let jackie = Term::value("Jackie");

    either(
        equal(
            Term::cons(parent.clone(), child.clone()),
            Term::cons(homer.clone(), bart.clone()),
        ),
        either(
            equal(
                Term::cons(parent.clone(), child.clone()),
                Term::cons(homer.clone(), lisa.clone()),
            ),
            either(
                equal(
                    Term::cons(parent.clone(), child.clone()),
                    Term::cons(marge.clone(), bart),
                ),
                either(
                    equal(
                        Term::cons(parent.clone(), child.clone()),
                        Term::cons(marge.clone(), lisa),
                    ),
                    either(
                        equal(
                            Term::cons(parent.clone(), child.clone()),
                            Term::cons(abe, homer),
                        ),
                        equal(Term::cons(parent, child), Term::cons(jackie, marge)),
                    ),
                ),
            ),
        ),
    )
}

fn sort_values(res: Vec<Term<&str>>) -> Vec<String> {
    let mut elements = res
        .into_iter()
        .flat_map(|term| match term {
            Term::Value(val) => Some(val.to_string()),
            _ => None,
        })
        .collect::<Vec<_>>();

    elements.sort();
    elements
}

#[test]
fn should_return_multiple_relations() {
    let children_of_homer = fresh("child", |child| {
        parent_child_relationship_goal(Term::value("Homer"), Term::var(child))
    });
    let stream = children_of_homer.apply(State::empty());
    let child_var = "child".to_var_repr(0);
    let res = stream.run(&Term::var(child_var));

    assert_eq!(stream.len(), 2, "{:?}", res);
    let sorted_children = sort_values(res);
    assert_eq!(
        ["Bart".to_string(), "Lisa".to_string()].as_slice(),
        sorted_children.as_slice()
    );

    // map parent relationship
    let parents_of_lisa = fresh("parent", |parent| {
        parent_child_relationship_goal(Term::var(parent), Term::value("Lisa"))
    });
    let stream = parents_of_lisa.apply(State::empty());
    let parent_var = "parent".to_var_repr(0);
    let res = stream.run(&Term::Var(parent_var));

    assert_eq!(stream.len(), 2, "{:?}", res);
    let sorted_parents = sort_values(res);

    assert_eq!(
        ["Homer".to_string(), "Marge".to_string()].as_slice(),
        sorted_parents.as_slice()
    );
}

#[test]
fn should_define_relations_without_fresh() {
    let mut state = State::empty();
    let child = state.declare("child");
    let children_of_homer = parent_child_relationship_goal(Term::value("Homer"), Term::var(child));
    let stream = children_of_homer.apply(state);
    let res = stream.run(&Term::Var(child));

    assert_eq!(stream.len(), 2, "{:?}", res);
    let sorted_children = sort_values(res);
    assert_eq!(
        ["Bart".to_string(), "Lisa".to_string()].as_slice(),
        sorted_children.as_slice()
    );

    // map parent relationship
    let mut state = State::empty();
    let parent = state.declare("parent");
    let parents_of_lisa = parent_child_relationship_goal(Term::var(parent), Term::value("Lisa"));
    let stream = parents_of_lisa.apply(state);
    let res = stream.run(&Term::var(parent));

    assert_eq!(stream.len(), 2, "{:?}", res);
    let sorted_parents = sort_values(res);

    assert_eq!(
        ["Homer".to_string(), "Marge".to_string()].as_slice(),
        sorted_parents.as_slice()
    );
}

#[test]
fn should_build_query_with_query_dsl() {
    use kannery::query::*;

    let child_of_homer = QueryBuilder::default()
        .with_var("child")
        .build(|child| parent_child_relationship_goal(Term::value("Homer"), child));

    let res = child_of_homer.run();
    let sorted_children = {
        let mut children: Vec<_> = res.owned_values_of("child").into_iter().collect();
        children.sort();
        children
    };

    assert_eq!(
        ["Bart".to_string(), "Lisa".to_string()].as_slice(),
        sorted_children.as_slice()
    );

    // map parent relationship
    let parents_of_lisa = QueryBuilder::default()
        .with_var("parent")
        .build(|parent| parent_child_relationship_goal(parent, Term::value("Lisa")));

    let res = parents_of_lisa.run();
    let sorted_parents = {
        let mut parents: Vec<_> = res.owned_values_of("parent").into_iter().collect();
        parents.sort();
        parents
    };

    assert_eq!(
        ["Homer".to_string(), "Marge".to_string()].as_slice(),
        sorted_parents.as_slice()
    );
}
