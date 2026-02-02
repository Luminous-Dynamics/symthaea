#![no_main]

use libfuzzer_sys::fuzz_target;
use mycelix_zk_tax::Jurisdiction;
use std::str::FromStr;

fuzz_target!(|data: &[u8]| {
    // Try to parse arbitrary bytes as jurisdiction code
    if let Ok(s) = std::str::from_utf8(data) {
        // Parsing should never panic
        let result = Jurisdiction::from_str(s);

        // If parsing succeeds, the jurisdiction should be valid
        if let Ok(jurisdiction) = result {
            // Getting properties should never panic
            let code = jurisdiction.code();
            let name = jurisdiction.name();
            let authority = jurisdiction.authority();
            let currency = jurisdiction.currency();
            let statuses = jurisdiction.valid_filing_statuses();

            // Code should be non-empty
            assert!(!code.is_empty());
            assert!(!name.is_empty());
            assert!(!authority.is_empty());
            assert!(!currency.is_empty());
            assert!(!statuses.is_empty());

            // Roundtrip: code -> jurisdiction -> code
            let roundtrip = Jurisdiction::from_str(code);
            assert!(roundtrip.is_ok());
            assert_eq!(roundtrip.unwrap(), jurisdiction);
        }
    }

    // Also try from_code
    if let Ok(s) = std::str::from_utf8(data) {
        if let Some(jurisdiction) = Jurisdiction::from_code(s) {
            assert_eq!(jurisdiction.code(), s.to_uppercase());
        }
    }
});
