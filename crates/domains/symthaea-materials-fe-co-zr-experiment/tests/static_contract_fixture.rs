// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::io::Cursor;
use symthaea_materials_authorized_audit_binding::bind_authorized_targets_to_historical_run;
use symthaea_materials_historical_extraction::{
    CompactHistoricalCorpus, NormalizedOqmdRecord, bind_extraction_receipt,
    oqmd_v17_fe_co_zr_protocol,
};
use symthaea_materials_historical_import::{
    HistoricalDatabaseEngine, HistoricalImportProfile, bind_successful_import,
};
use symthaea_materials_history_tool::verify_historical_run;
use symthaea_materials_schema_inventory::{
    ColumnInventory, IndexColumnInventory, IndexInventory, MySqlSchemaInventory,
    TableInventory,
};
use symthaea_materials_snapshot_acquisition::{
    AcquisitionRoute, HistoricalSnapshotAcquisitionReceipt, hash_snapshot_stream,
};
use symthaea_materials_target_set::{
    AuthorizedHistoricalTarget, AuthorizedHistoricalTargetSet, AuthorizedQuantitativeLabel,
};

fn hex(ch: char) -> String {
    ch.to_string().repeat(64)
}

#[test]
fn pure_fixture_reaches_authorized_historical_audit_binding() {
    let protocol = oqmd_v17_fe_co_zr_protocol();
    let snapshot = b"oqmd-v1.7-static-contract-fixture";
    let identity = hash_snapshot_stream(Cursor::new(snapshot)).unwrap();
    let acquisition = HistoricalSnapshotAcquisitionReceipt {
        schema_version: 1,
        protocol_sha256: protocol.protocol_sha256().unwrap(),
        provider: protocol.provider.clone(),
        database_version: protocol.database_version.clone(),
        dump_filename: protocol.dump_filename.clone(),
        source_license: protocol.source_license.clone(),
        acquired_at_utc: "2026-09-20T00:00:00Z".to_string(),
        compressed_snapshot_sha256: identity.sha256.clone(),
        compressed_snapshot_bytes: identity.bytes,
        acquisition_tool_sha256: hex('a'),
        execution_environment_sha256: hex('b'),
        transfer_log_sha256: hex('c'),
        route: AcquisitionRoute::LocalMirror {
            mirror_locator: "static-contract-fixture".to_string(),
            mirror_manifest_sha256: hex('d'),
        },
    };
    let profile = HistoricalImportProfile {
        schema_version: 1,
        protocol_sha256: protocol.protocol_sha256().unwrap(),
        database_engine: HistoricalDatabaseEngine::MySql,
        server_artifact_sha256: hex('1'),
        client_artifact_sha256: hex('2'),
        decompressor_artifact_sha256: hex('3'),
        import_environment_manifest_sha256: hex('4'),
        server_version: "fixture-server".to_string(),
        client_version: "fixture-client".to_string(),
        character_set_server: "utf8mb4".to_string(),
        collation_server: "utf8mb4_bin".to_string(),
        sql_mode: vec!["STRICT_TRANS_TABLES".to_string()],
        lower_case_table_names: 0,
        time_zone: "+00:00".to_string(),
        max_allowed_packet_bytes: 64 * 1024 * 1024,
        innodb_strict_mode: true,
        import_command_sha256: hex('5'),
    };
    let inventory = MySqlSchemaInventory {
        schema_version: 1,
        database_name: "oqmd_v17".to_string(),
        tables: vec![TableInventory {
            name: "entries".to_string(),
            engine: "InnoDB".to_string(),
            show_create_table_sha256: hex('6'),
            exact_row_count: 1,
            columns: vec![ColumnInventory {
                ordinal_position: 1,
                name: "id".to_string(),
                column_type: "bigint".to_string(),
                nullable: false,
                default_repr: None,
                collation: None,
                extra: String::new(),
            }],
            indexes: vec![IndexInventory {
                name: "PRIMARY".to_string(),
                unique: true,
                index_type: "BTREE".to_string(),
                columns: vec![IndexColumnInventory {
                    sequence: 1,
                    column_name: "id".to_string(),
                    sub_part: None,
                    descending: false,
                }],
            }],
            foreign_keys: Vec::new(),
        }],
    };
    let import = bind_successful_import(
        &protocol,
        &profile,
        &acquisition,
        &hex('7'),
        &hex('8'),
        &inventory.inventory_sha256().unwrap(),
        &inventory.row_count_projection_sha256().unwrap(),
        &hex('9'),
    )
    .unwrap();

    let composition_sha = hex('e');
    let structure_sha = hex('f');
    let corpus = CompactHistoricalCorpus::from_records(
        &protocol,
        vec![NormalizedOqmdRecord {
            entry_id: 1,
            name: "Fe".to_string(),
            element_set: vec!["Fe".to_string()],
            composition_sha256: composition_sha.clone(),
            structure_sha256: Some(structure_sha.clone()),
            duplicate_entry_id: None,
            spacegroup: None,
            prototype: None,
            natoms: 1,
            ntypes: 1,
            delta_e_ev_atom: Some("0".to_string()),
            stability_ev_atom: Some("0".to_string()),
            band_gap_ev: None,
            calculation_label: Some("static".to_string()),
            fit: Some("standard".to_string()),
            icsd_id: None,
            property_condition_signature: "oqmd-v1.7-standard-static".to_string(),
        }],
    )
    .unwrap();
    let import_sha = import
        .receipt_sha256(&protocol, &profile, &acquisition)
        .unwrap();
    let extraction = bind_extraction_receipt(
        &protocol,
        &corpus,
        &identity.sha256,
        &profile.import_environment_manifest_sha256,
        &inventory.inventory_sha256().unwrap(),
        &import_sha,
        &hex('a'),
        &hex('b'),
        &hex('c'),
        &hex('d'),
        &hex('1'),
    )
    .unwrap();

    // This target's exact structure exists historically, but its K1 label does not.
    // The fixture should therefore classify a label holdout rather than a novel structure.
    let authorized = AuthorizedHistoricalTargetSet {
        schema_version: 1,
        disclosure_manifest_sha256: hex('2'),
        source_artifact_sha256: hex('3'),
        disclosure_date: "2026-02-09".to_string(),
        targets: vec![AuthorizedHistoricalTarget {
            target_id: "fixture-fe-k1".to_string(),
            composition_sha256: composition_sha,
            structure_sha256: structure_sha,
            structure_artifact_sha256: hex('4'),
            structure_fact_id: "structure-fixture".to_string(),
            properties: vec![AuthorizedQuantitativeLabel {
                disclosure_fact_id: "property-fixture".to_string(),
                property_id: "k1_mj_m3".to_string(),
                value: "1.1".to_string(),
                unit: "MJ/m^3".to_string(),
                condition_signature: "0K|SOC".to_string(),
                method_artifact_sha256: hex('5'),
            }],
        }],
    };
    authorized.validate().unwrap();
    let targets = authorized.to_audit_targets().unwrap();

    let verified = verify_historical_run(
        &hex('9'),
        &protocol,
        Cursor::new(snapshot),
        &acquisition,
        &profile,
        &import,
        &inventory,
        &extraction,
        &corpus,
        &targets,
    )
    .unwrap();
    let binding = bind_authorized_targets_to_historical_run(&verified, &authorized).unwrap();

    assert_eq!(verified.bundle.target_count, 1);
    assert_eq!(verified.bundle.compact_corpus_record_count, 1);
    assert_eq!(binding.target_count, 1);
    assert_eq!(binding.binding_sha256().unwrap().len(), 64);
    assert_eq!(verified.verified_run_sha256().unwrap().len(), 64);
}
