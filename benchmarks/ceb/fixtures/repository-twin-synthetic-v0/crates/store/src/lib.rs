//! Storage abstraction and deterministic in-memory implementation.

use std::collections::BTreeMap;
use symthaea_ceb_domain::{User, UserId};

pub trait Store {
    fn get_user(&self, id: UserId) -> Option<User>;
}

#[derive(Debug, Default, Clone)]
pub struct MemoryStore {
    users: BTreeMap<UserId, User>,
}

impl MemoryStore {
    pub fn insert(&mut self, user: User) {
        self.users.insert(user.id, user);
    }
}

impl Store for MemoryStore {
    fn get_user(&self, id: UserId) -> Option<User> {
        self.users.get(&id).cloned()
    }
}
