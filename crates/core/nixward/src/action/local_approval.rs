        .unwrap()
    }

    fn request(intent: &NixActionIntentV1, nonce_byte: u8) -> PendingNixApprovalRequestV1 {
        PendingNixApprovalRequestV1::from_intent(
            intent,
            "daemon-incarnation:1",
            "nixos-rebuild switch --flake .#workstation",
            "same-uid-process-v1",
            ms(1_000),
            ms(2_000),
            [nonce_byte; 32],
        )
        .unwrap()
    }

    #[test]
    fn request_identity_is_deterministic_and_binds_display() {
        let intent = intent("machine:workstation", "generation:42");
        let a = request(&intent, 1);
        let mut b = request(&intent, 1);