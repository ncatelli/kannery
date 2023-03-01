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
    };

    let children_of_homer = fresh("child", |child| {
        parent_fn(Term::value("Homer"), Term::var(child))
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
        parent_fn(Term::var(parent), Term::value("Lisa"))
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
    };

    let mut state = State::empty();
    let child = state.declare("child");
    let children_of_homer = || parent_fn(Term::value("Homer"), Term::var(child));
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
    let parents_of_lisa = parent_fn(Term::var(parent), Term::value("Lisa"));
    let stream = parents_of_lisa.apply(state);
    let res = stream.run(&Term::var(parent));

    assert_eq!(stream.len(), 2, "{:?}", res);
    let sorted_parents = sort_values(res);

    assert_eq!(
        ["Homer".to_string(), "Marge".to_string()].as_slice(),
        sorted_parents.as_slice()
    );
}
