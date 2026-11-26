//! Profile query API.

use symthaea_ceb_domain::{Authorizer, User, UserId};
use symthaea_ceb_service::UserService;
use symthaea_ceb_store::Store;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    pub user: User,
}

pub fn get_profile<S, A>(service: &UserService<S, A>, id: UserId) -> Option<Profile>
where
    S: Store,
    A: Authorizer,
{
    service.get_user(id).map(|user| Profile { user })
}
