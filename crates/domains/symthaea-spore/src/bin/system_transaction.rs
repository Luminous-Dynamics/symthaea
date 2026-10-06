        std::fs::write(&path, format!("{first}\n{second}\n")).unwrap();
        std::fs::set_permissions(
            &path,
            std::os::unix::fs::PermissionsExt::from_mode(0o600),
        )
        .unwrap();
        let ledger = TransactionLedger::open_at(&path).unwrap();
        let error = ledger.load().expect_err("transaction ID collision must fail closed");
        assert!(error.contains("reuses transaction_id"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ledger_rejects_oversized_total_file_on_load() {
        let name = random_suffix();
        let path = std::env::temp_dir()
            .join(format!("symthaea-transaction-ledger-total-oversized-{name}.jsonl"));
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(MAX_JOURNAL_BYTES + 1).unwrap();
        drop(file);

        let ledger = TransactionLedger::open_at(&path).unwrap();
        let error = ledger
            .load()
            .expect_err("oversized total journal must fail closed");
        assert!(error.contains("total limit"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ledger_rejects_oversized_event_on_load() {
        use std::os::unix::fs::PermissionsExt;

        let name = random_operation_id().unwrap();
        let path = std::env::temp_dir()
            .join(format!("symthaea-transaction-ledger-oversized-{name}.jsonl"));
        let oversized = "x".repeat(MAX_JOURNAL_EVENT_BYTES + 1);
        std::fs::write(&path, format!("{oversized}
")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

        let ledger = TransactionLedger::open_at(&path).unwrap();
        let error = ledger.load().expect_err("oversized event must fail closed");
        assert!(error.contains("exceeds"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn ledger_rejects_unterminated_oversized_event_without_unbounded_read() {
        let name = random_operation_id().unwrap();
        let path = std::env::temp_dir()
            .join(format!("symthaea-transaction-ledger-unterminated-oversized-{name}.jsonl"));
        let oversized = "x".repeat(MAX_JOURNAL_EVENT_BYTES + 1);
        std::fs::write(&path, oversized).unwrap();
        std::fs::set_permissions(
            &path,
            std::os::unix::fs::PermissionsExt::from_mode(0o600),