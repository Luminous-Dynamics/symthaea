        )
        .expect("explicit BLAKE3 provenance should permit lexical binding");

        assert_eq!(plan.content_binding, ContentBindingStatus::LexicallyBound);
        assert_eq!(plan.lexical_provenance.as_deref(), Some(provenance.as_str()));
        assert!(
            plan.grounding_surface()
                .contains(&format!(r#""lexical_provenance":"{}""#, provenance))
        );
    }
