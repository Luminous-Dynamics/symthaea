![Crates.io](https://img.shields.io/crates/l/must_future)
![Crates.io](https://img.shields.io/crates/v/must_future)

# must_future

BoxFutures cannot be marked `#[must_use]` because they are just type
definitions. This newtype struct wraps a BoxFuture with something that
can be marked `#[must_use]`.

## Will Not Compile:

```compile_fail
#![deny(unused_must_use)]

use futures::future::FutureExt;

#[tokio::main]
async fn main() {
    fn get_future() -> must_future::MustBoxFuture<'static, ()> {
        async { }.boxed().into()
    }

    get_future(); // unused `must_future::MustBoxFuture` that must be used
}
```
