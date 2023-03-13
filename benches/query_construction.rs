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

fn evaluate_simple_query(c: &mut Criterion) {
    c.bench_function("construction of query", |b| {
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
}

criterion_group!(benches, evaluate_simple_query);
criterion_main!(benches);
