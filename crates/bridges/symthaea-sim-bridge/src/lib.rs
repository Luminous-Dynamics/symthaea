            CouplingMode::OneWay,
        )
        .with_stage(producer)
        .with_stage(consumer)
        .with_typed_connection(TypedPhysicalConnection::new(
            "producer",
            "length",
            "consumer",
            "length",
        ));

        assert!(request.validate_typed_topology().is_err());
    }

    #[test]
    fn explicit_typed_topology_allows_fan_out_to_distinct_destinations() {
        let length = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        );
        let producer = CoupledSimulationStage::new(
            "producer",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
        )
        .typed_produces([TypedPhysicalPort::new("length", length.clone())]);
        let consumer_a = CoupledSimulationStage::new(
            "consumer-a",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_consumes([TypedPhysicalPort::new("length", length.clone())]);
        let consumer_b = CoupledSimulationStage::new(
            "consumer-b",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_consumes([TypedPhysicalPort::new("length", length)]);

        let request = MultiPhysicsRequest::new(
            "fan-out",
            "preserve legitimate one-to-many topology",
            CouplingMode::OneWay,
        )
        .with_stage(producer)
        .with_stage(consumer_a)
        .with_stage(consumer_b)
        .with_typed_connection(TypedPhysicalConnection::new(
            "producer", "length", "consumer-a", "length",
        ))
        .with_typed_connection(TypedPhysicalConnection::new(
            "producer", "length", "consumer-b", "length",
        ));

        assert!(request.validate_typed_topology().is_ok());
    }

    #[test]
    fn explicit_typed_topology_rejects_duplicate_stage_ids() {
        let producer = CoupledSimulationStage::new(
            "duplicate",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
        );
        let consumer = CoupledSimulationStage::new(
            "duplicate",
            EngineeringDomain::Electrical,
            SolverKind::Circuit,
        );

        let request = MultiPhysicsRequest::new(
            "duplicate-stage",
            "reject ambiguous stage identity",
            CouplingMode::OneWay,
        )
        .with_stage(producer)
        .with_stage(consumer);

        assert!(request.validate_typed_topology().is_err());
    }

    #[test]
    fn explicit_typed_topology_rejects_duplicate_connections() {
        let length = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        );
        let producer = CoupledSimulationStage::new(
            "producer",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
        )
        .typed_produces([TypedPhysicalPort::new("length", length.clone())]);
        let consumer = CoupledSimulationStage::new(
            "consumer",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_consumes([TypedPhysicalPort::new("length", length)]);

        let edge = TypedPhysicalConnection::new(
            "producer", "length", "consumer", "length",
        );
        let request = MultiPhysicsRequest::new(
            "duplicate-edge",
            "reject duplicate typed edge identity",
            CouplingMode::OneWay,
        )
        .with_stage(producer)
        .with_stage(consumer)
        .with_typed_connection(edge.clone())
        .with_typed_connection(edge);

        assert!(request.validate_typed_topology().is_err());
    }

    #[test]
    fn explicit_typed_topology_rejects_multiple_producers_for_one_input() {
        let length = PhysicalType::with_kind(
            symthaea_types::QuantityKind::Length,
            symthaea_types::PhysicalDimension::LENGTH,
        );
        let producer_a = CoupledSimulationStage::new(
            "producer-a",
            EngineeringDomain::Mechanical,
            SolverKind::FiniteElement,
        )
        .typed_produces([TypedPhysicalPort::new("length", length.clone())]);
        let producer_b = CoupledSimulationStage::new(
            "producer-b",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_produces([TypedPhysicalPort::new("length", length.clone())]);
        let consumer = CoupledSimulationStage::new(
            "consumer",
            EngineeringDomain::Mechanical,
            SolverKind::MultibodyDynamics,
        )
        .typed_consumes([TypedPhysicalPort::new("length", length)]);

        let request = MultiPhysicsRequest::new(
            "ambiguous-destination",
            "reject many-to-one typed endpoint",
            CouplingMode::OneWay,
        )
        .with_stage(producer_a)
        .with_stage(producer_b)
        .with_stage(consumer)