        let n = self.test_samples as f64;
        if n <= 0.0 {
            return Err(RelationalPredictionError::InvalidSplit);
        }

        let mean_absolute_error = absolute_error / n;
        let mean_squared_error = squared_error / n;
        let tolerance = 1e-12;

        if (mean_absolute_error - self.mean_absolute_error).abs() > tolerance
            || (mean_squared_error - self.mean_squared_error).abs() > tolerance
        {
            return Err(RelationalPredictionError::InvalidSplit);
        }

        Ok(())
    }
}

/// One complete held-out evidence packet.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutRelationalPredictionEvidence {
    pub provenance: RelationalPredictionProvenance,
    pub summary: HeldOutRelationalPredictionSummary,
    pub records: Vec<PredictionEvidenceRecord>,
}

/// Rolling-origin packet retaining a complete trace at each origin.
#[derive(Debug, Clone, PartialEq)]
pub struct RollingOriginRelationalPredictionEvidence {
    pub provenance: RelationalPredictionProvenance,
    pub config: RollingOriginRelationalPredictionConfig,
    pub observed: RollingOriginRelationalPredictionSummary,
    pub origins: Vec<HeldOutRelationalPredictionEvidence>,
}

/// Caller-attested provenance for an empirical qualification run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationalPredictionProvenance {
    pub protocol_id: String,
    pub source_data_sha256: String,
    pub software_commit_sha: String,
}

impl RelationalPredictionProvenance {
    pub fn new(
        protocol_id: impl Into<String>,
        source_data_sha256: impl Into<String>,
        software_commit_sha: impl Into<String>,
    ) -> Result<Self, RelationalPredictionError> {
        let provenance = Self {
            protocol_id: protocol_id.into(),
            source_data_sha256: source_data_sha256.into(),
            software_commit_sha: software_commit_sha.into(),
        };

        if provenance.protocol_id.trim().is_empty() {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "protocol_id",
            ));
        }
        if !is_hex_digest(&provenance.source_data_sha256, 64) {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "source_data_sha256",
            ));
        }
        if !is_hex_digest(&provenance.software_commit_sha, 40) {
            return Err(RelationalPredictionError::InvalidEvidenceProvenance(
                "software_commit_sha",
            ));
        }

        Ok(provenance)
    }
}

/// Exact held-out prediction trace for one feature family.
///
/// This retains fitted preprocessing/model parameters and every held-out
/// Held-out comparison across the required baselines and the relational model.
///
/// No score is interpreted as a consciousness, relationship, value, or
/// causality measure. The only claim this structure supports is predictive
/// comparison under the supplied split and target definition.
#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutRelationalPredictionSummary {
    pub train_samples: usize,
    pub test_samples: usize,
    pub gap_samples: usize,
    pub minimum_outcome_horizon: f64,
    pub maximum_outcome_horizon: f64,
    pub persistence_baseline: PredictionScore,
    pub isolated_agents: PredictionScore,
    pub common_driver: PredictionScore,
    pub synchrony_only: PredictionScore,
    pub non_relational_context: PredictionScore,
    pub relational_augmented: PredictionScore,
    pub relational_profile: PredictionScore,
    pub status: EvidenceStatus,
}

impl HeldOutRelationalPredictionEvidence {
    pub fn validate(&self) -> Result<(), RelationalPredictionError> {
        Self::validate_provenance(&self.provenance)?;
        if self.records.len() != PredictionFeatureSet::all().len() {
            return Err(RelationalPredictionError::InvalidSplit);
        }

        for record in &self.records {
            record.validate_trace()?;
            if record.score() != self.summary.score(record.feature_set) {
                return Err(RelationalPredictionError::InvalidSplit);
            }
        }

        Ok(())
    }

    pub fn to_json(&self) -> Result<String, RelationalPredictionError> {
        self.validate()?;

        let scores = PredictionFeatureSet::all()
            .into_iter()
            .map(|feature_set| {
                let score = self.summary.score(feature_set);
                serde_json::json!({
                    "feature_set": feature_set_name(feature_set),
                    "parameter_count": score.parameter_count,
                    "train_samples": score.train_samples,
                    "test_samples": score.test_samples,
                    "mean_absolute_error": score.mean_absolute_error,
                    "mean_squared_error": score.mean_squared_error
                })
            })
            .collect::<Vec<_>>();

        let records = self.records
            .iter()
            .map(prediction_evidence_record_json)
            .collect::<Vec<_>>();

        Ok(serde_json::json!({
            "schema": "relational-prediction-evidence/v1",
            "provenance": {
                "protocol_id": &self.provenance.protocol_id,
                "source_data_sha256": &self.provenance.source_data_sha256,
                "software_commit_sha": &self.provenance.software_commit_sha
            },
            "split": {
                "train_samples": self.summary.train_samples,
                "test_samples": self.summary.test_samples,
                "gap_samples": self.summary.gap_samples,
                "minimum_outcome_horizon": self.summary.minimum_outcome_horizon,
                "maximum_outcome_horizon": self.summary.maximum_outcome_horizon
            },
            "scores": scores,
            "records": records
        }).to_string())
    }

    fn validate_provenance(