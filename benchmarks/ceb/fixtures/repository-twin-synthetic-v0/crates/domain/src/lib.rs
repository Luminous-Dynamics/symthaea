//! Domain records and authorization contract.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UserId(pub u64);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct User {
    pub id: UserId,
    pub display_name: String,
}

pub trait Authorizer {
    fn can_read(&self, user: &User) -> bool;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultAuthorizer;

impl Authorizer for DefaultAuthorizer {
    fn can_read(&self, user: &User) -> bool {
        user.id.0 != 0
    }
}
