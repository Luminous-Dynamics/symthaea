use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::process::ExitCode;

use sym_rsi_asb_adapter_reference::api_contract::{
    validate_catalog_submission, CampaignCardV1, CampaignStatusV1, CatalogResponseV1, JobResponseV1,
};
use sym_rsi_asb_adapter_reference::parse_and_validate;

fn validate_api_fixtures() -> Result<serde_json::Value, String> {
    let campaign = CampaignCardV1::parse(include_str!("../fixtures/campaign-card-v1.json"))
        .map_err(|error| format!("campaign fixture: {error}"))?;
    campaign.validate().map_err(|error| format!("campaign contract: {error}"))?;

    let status = CampaignStatusV1::parse(include_str!("../fixtures/status-v1.json"))
        .map_err(|error| format!("status fixture: {error}"))?;
    status
        .validate_against(&campaign)
        .map_err(|error| format!("status contract: {error}"))?;

    let catalog = CatalogResponseV1::parse(include_str!("../fixtures/catalog-stage-1-v1.json"))
        .map_err(|error| format!("catalog fixture: {error}"))?;
    let catalog_ids = catalog
        .validate_against(&campaign, 1)
        .map_err(|error| format!("catalog contract: {error}"))?;

    let selected = vec!["d-01".to_owned(), "d-02".to_owned()];
    validate_catalog_submission(
        &campaign,
        1,
        &selected,
        &catalog_ids,
        &BTreeSet::new(),
        status.budget.remaining,
    )
    .map_err(|error| format!("submission contract: {error}"))?;

    let completed = JobResponseV1::parse(include_str!("../fixtures/completed-job-v1.json"))
        .map_err(|error| format!("completed job fixture: {error}"))?;
    let advances_stage = completed
        .advances_stage(&campaign)
        .map_err(|error| format!("completed job contract: {error}"))?;

    let pinned_cards = [
        (
            "nuclease-active-learning",
            include_str!("../fixtures/asb_lab_v1/nuclease_campaign_card.json"),
        ),
        (
            "propylene-active-learning",
            include_str!("../fixtures/asb_lab_v1/propylene_campaign_card.json"),
        ),
    ];
    let mut pinned_campaign_reports = Vec::new();
    for (expected_id, contents) in pinned_cards {
        let pinned = CampaignCardV1::parse(contents)
            .map_err(|error| format!("{expected_id} pinned campaign card: {error}"))?;
        pinned.validate().map_err(|error| format!("{expected_id} pinned campaign contract: {error}"))?;
        let expected_budget = match expected_id {
            "nuclease-active-learning" => 384,
            "propylene-active-learning" => 60,
            _ => unreachable!("fixed pinned campaign fixture"),
        };
        if !pinned.campaign_id.ends_with(expected_id) || pinned.budget.total != expected_budget {
            return Err(format!("{expected_id}: pinned campaign identity/budget mismatch"));
        }
        pinned_campaign_reports.push(serde_json::json!({
            "campaign": expected_id,
            "campaign_id": pinned.campaign_id,
            "total_budget": pinned.budget.total,
            "stages": pinned.stages.len(),
            "shared_pool": pinned.is_shared_pool_catalog()
        }));
    }

    Ok(serde_json::json!({
        "valid": true,
        "campaign_id": campaign.campaign_id,
        "api_version": campaign.api_version,
        "catalog_designs": catalog_ids.len(),
        "selected_batch_size": selected.len(),
        "completed_job_advances_stage": advances_stage,
        "pinned_upstream_campaign_cards": pinned_campaign_reports,
        "claim_limit": "Fixture contract conformance only; no HTTP request, oracle access, campaign run or benchmark result."
    }))
}

fn main() -> ExitCode {
    let Some(argument) = env::args().nth(1) else {
        eprintln!("usage: sym-rsi-asb-adapter-reference <manifest.json|--api-fixtures>");
        return ExitCode::from(64);
    };

    if argument == "--api-fixtures" {
        return match validate_api_fixtures() {
            Ok(report) => {
                println!("{}", serde_json::to_string_pretty(&report).expect("serializable report"));
                ExitCode::SUCCESS
            }
            Err(error) => {
                eprintln!("invalid Lab API v1 fixture contract: {error}");
                ExitCode::from(2)
            }
        };
    }

    let contents = match fs::read_to_string(&argument) {
        Ok(contents) => contents,
        Err(error) => {
            eprintln!("cannot read {argument}: {error}");
            return ExitCode::from(66);
        }
    };

    match parse_and_validate(&contents) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("serializable validation report")
            );
            if report.valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            }
        }
        Err(error) => {
            eprintln!("invalid compatibility manifest JSON: {error}");
            ExitCode::from(65)
        }
    }
}
