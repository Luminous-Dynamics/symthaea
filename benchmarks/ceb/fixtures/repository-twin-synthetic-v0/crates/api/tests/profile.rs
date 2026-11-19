use symthaea_ceb_api::get_profile;
use symthaea_ceb_domain::{Authorizer, User, UserId};
use symthaea_ceb_service::UserService;
use symthaea_ceb_store::{MemoryStore, Store};

struct AllowAll;

impl Authorizer for AllowAll {
    fn can_read(&self, _user: &User) -> bool {
        true
    }
}

#[test]
fn profile_query_returns_the_matching_user() {
    let mut store = MemoryStore::default();
    let expected = User {
        id: UserId(7),
        display_name: "Ada".to_string(),
    };
    store.insert(expected.clone());

    let service = UserService::new(store, AllowAll);
    let result = get_profile(&service, UserId(7)).expect("profile");

    assert_eq!(result.user, expected);
}

#[test]
fn profile_query_returns_none_for_unknown_user() {
    let service = UserService::new(MemoryStore::default(), AllowAll);
    assert!(get_profile(&service, UserId(999)).is_none());
}

// Keep the trait import in this integration test as an explicit evidence point:
// the test uses a generic API whose service parameter is constrained by Store.
fn _store_contract<S: Store>(_store: &S) {}
