use criterion::{criterion_group, criterion_main, Criterion};
use kannery::*;

/// A simple relationship mapping.
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

fn boxed_parent_child_relationship_goal(
    parent: Term<&'static str>,
    child: Term<&'static str>,
) -> impl Goal<&'static str> {
    let homer = Term::value("Homer");
    let marge = Term::value("Marge");
    let bart = Term::value("Bart");
    let lisa = Term::value("Lisa");
    let abe = Term::value("Abe");
    let jackie = Term::value("Jackie");

    any([
        equal(
            Term::cons(parent.clone(), child.clone()),
            Term::cons(homer.clone(), bart.clone()),
        ),
        equal(
            Term::cons(parent.clone(), child.clone()),
            Term::cons(homer.clone(), lisa.clone()),
        ),
        equal(
            Term::cons(parent.clone(), child.clone()),
            Term::cons(marge.clone(), bart),
        ),
        equal(
            Term::cons(parent.clone(), child.clone()),
            Term::cons(marge.clone(), lisa),
        ),
        equal(
            Term::cons(parent.clone(), child.clone()),
            Term::cons(abe, homer),
        ),
        equal(Term::cons(parent, child), Term::cons(jackie, marge)),
    ])
}

fn relation_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("building relations");

    group.bench_function("construction of static relation", |b| {
        b.iter(|| {
            let child_of_homer = fresh("child", |child| {
                parent_child_relationship_goal(Term::value("Homer"), Term::var(child))
            });

            let stream = child_of_homer.apply(State::empty());
            let child_var = "child".to_var_repr(0);
            let children = stream.run(&Term::Var(child_var));

            assert_eq!(children.len(), 2);
        });
    });

    group.bench_function("construction of relation from compound goal", |b| {
        b.iter(|| {
            let child_of_homer = fresh("child", |child| {
                parent_child_relationship_goal(Term::value("Homer"), Term::var(child))
            });

            let stream = child_of_homer.apply(State::empty());
            let child_var = "child".to_var_repr(0);
            let children = stream.run(&Term::Var(child_var));

            assert_eq!(children.len(), 2);
        });
    });
}

fn query_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("query construction");

    group.bench_function("construction of static query", |b| {
        b.iter(|| {
            let child_of_homer = QueryBuilder::default()
                .with_value("Homer")
                .with_var("child")
                .build(|(parent, child)| parent_child_relationship_goal(parent, child));

            let res = child_of_homer.run();
            let children = res.owned_values_of("child");

            assert_eq!(children.len(), 2);
        });
    });

    group.bench_function("construction of query from compound goal", |b| {
        b.iter(|| {
            let child_of_homer = QueryBuilder::default()
                .with_value("Homer")
                .with_var("child")
                .build(|(parent, child)| boxed_parent_child_relationship_goal(parent, child));

            let res = child_of_homer.run();
            let children = res.owned_values_of("child");

            assert_eq!(children.len(), 2);
        });
    });
}

criterion_group!(benches, relation_construction, query_construction);
criterion_main!(benches);
