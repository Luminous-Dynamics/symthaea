//! Generic service over storage and authorization abstractions.

use symthaea_ceb_domain::{Authorizer, User, UserId};
use symthaea_ceb_store::Store;

pub struct UserService<S, A>
where
    S: Store,
    A: Authorizer,
{
    store: S,
    authorizer: A,
}

impl<S, A> UserService<S, A>
where
    S: Store,
    A: Authorizer,
{
    pub fn new(store: S, authorizer: A) -> Self {
        Self { store, authorizer }
    }

    pub fn get_user(&self, id: UserId) -> Option<User> {
        let user = self.store.get_user(id)?;
        self.authorizer.can_read(&user).then_some(user)
    }
}
