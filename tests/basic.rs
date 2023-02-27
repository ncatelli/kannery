use kannery::*;

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
    let parent_fn = |parent: Term<_>, child: Term<_>| {
        let homer = value_term!("Homer");
        let marge = value_term!("Marge");
        let bart = value_term!("Bart");
        let lisa = value_term!("Lisa");
        let abe = value_term!("Abe");
        let jackie = value_term!("Jackie");

        either(
            equal(
                cons_term!(parent.clone(), child.clone()),
                cons_term!(homer.clone(), bart.clone()),
            ),
            either(
                equal(
                    cons_term!(parent.clone(), child.clone()),
                    cons_term!(homer.clone(), lisa.clone()),
                ),
                either(
                    equal(
                        cons_term!(parent.clone(), child.clone()),
                        cons_term!(marge.clone(), bart),
                    ),
                    either(
                        equal(
                            cons_term!(parent.clone(), child.clone()),
                            cons_term!(marge.clone(), lisa),
                        ),
                        either(
                            equal(
                                cons_term!(parent.clone(), child.clone()),
                                cons_term!(abe, homer),
                            ),
                            equal(cons_term!(parent, child), cons_term!(jackie, marge)),
                        ),
                    ),
                ),
            ),
        )
    };

    let children_of_homer = fresh("child", |child| {
        parent_fn(value_term!("Homer"), var_term!(child))
    });
    let stream = children_of_homer.apply(State::empty());
    let child_var = "child".to_var_repr(0);
    let res = stream.run(&var_term!(child_var));

    assert_eq!(stream.len(), 2, "{:?}", res);
    let sorted_children = sort_values(res);
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
    let sorted_parents = sort_values(res);

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

        either(
            equal(
                cons_term!(parent.clone(), child.clone()),
                cons_term!(homer.clone(), bart.clone()),
            ),
            either(
                equal(
                    cons_term!(parent.clone(), child.clone()),
                    cons_term!(homer.clone(), lisa.clone()),
                ),
                either(
                    equal(
                        cons_term!(parent.clone(), child.clone()),
                        cons_term!(marge.clone(), bart),
                    ),
                    either(
                        equal(
                            cons_term!(parent.clone(), child.clone()),
                            cons_term!(marge.clone(), lisa),
                        ),
                        either(
                            equal(
                                cons_term!(parent.clone(), child.clone()),
                                cons_term!(abe, homer),
                            ),
                            equal(cons_term!(parent, child), cons_term!(jackie, marge)),
                        ),
                    ),
                ),
            ),
        )
    };

    let mut state = State::empty();
    let child = state.declare("child");
    let children_of_homer = || parent_fn(value_term!("Homer"), var_term!(child));
    let stream = children_of_homer().apply(state);
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
    let parents_of_lisa = parent_fn(var_term!(parent), value_term!("Lisa"));
    let stream = parents_of_lisa.apply(state);
    let res = stream.run(&var_term!(parent));

    assert_eq!(stream.len(), 2, "{:?}", res);
    let sorted_parents = sort_values(res);

    assert_eq!(
        ["Homer".to_string(), "Marge".to_string()].as_slice(),
        sorted_parents.as_slice()
    );
}
