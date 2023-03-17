# kannery
## General
A small µKanren-inspired utility, containing an additional query-builder DSL.

## Usage
### Testing
Tests and documentation are provided primarily via docstring examples but a few tests are additionally provided through standard rust unit testing and can be run via:

```
cargo test
```

#### Benchmark tests
Additionally benchmark tests are provided via `criterion` and can be run via:

```
cargo bench
```

### Example Query

```rust
use kannery::prelude::v1::*;
use kannery::*;

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

let result = query.run();
let a_values = result.owned_values_of('a');
let b_values = result.owned_values_of('b');

// assert all values of a == 1.
assert!(a_values.into_iter().all(|val| val == 1_u8));

// assert all values of b == 1.
assert!(b_values.into_iter().all(|val| val == 1_u8))
```
