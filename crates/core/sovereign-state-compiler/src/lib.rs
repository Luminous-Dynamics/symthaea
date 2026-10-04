        &self.authorization_digest
    }

    pub fn authority_id(&self) -> &str {
        &self.authorized.authorization.authority_id
    }

    pub fn nonce(&self) -> &str {
        &self.authorized.authorization.nonce
    }

    pub fn consumed_at_ms(&self) -> u64 {
        self.consumed_at_ms
    }

    /// Admit this consumed authorization for execution at `now_ms`.
    ///
    /// Revalidation here closes the gap between durable nonce consumption and
    /// mutation start: an authorization that expires after consumption is
    /// safely burned rather than executed late.
    pub fn admit_execution(
        self,
        now_ms: u64,
    ) -> Result<ExecutionAuthorization<'a>, PlanValidationError> {
        if now_ms < self.consumed_at_ms {
            return Err(PlanValidationError::AuthorizationConsumptionFromFuture {
                consumed_at_ms: self.consumed_at_ms,
                now_ms,
            });
        }
        self.authorized.validate(now_ms)?;
        Ok(ExecutionAuthorization {
            authorized: self.authorized,
            authorization_digest: self.authorization_digest,
            consumed_at_ms: self.consumed_at_ms,
            started_at_ms: now_ms,
        })
    }
}
/// Opaque execution admission for one exact, already-consumed authorization.
///
/// This handle carries the validated execution-start timestamp and exact
/// authorization digest recorded by the durable consumption boundary. It is
/// deliberately non-cloneable and non-serializable.
#[derive(Debug)]
pub struct ExecutionAuthorization<'a> {
    authorized: &'a AuthorizedDeploymentPlan,
    authorization_digest: ContentDigest,
    consumed_at_ms: u64,
    started_at_ms: u64,
}

/// Source of a fresh target snapshot for execution-time preflight.
///
/// Implementations are target-specific and are responsible for obtaining the
/// snapshot from the live target rather than replaying portable evidence.
pub trait ExecutionPreflightObserver {
    type Error: std::error::Error + Send + Sync + 'static;

    fn observe(&mut self) -> Result<TargetSnapshot, Self::Error>;
}

#[derive(Debug, Error)]
pub enum ExecutionPreflightError<E: std::error::Error + Send + Sync + 'static> {
    #[error("execution preflight observation failed: {0}")]
    Observation(E),
    #[error("execution preflight canonical serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("execution preflight target identity does not match authorization")]
    TargetIdentityMismatch,
    #[error("execution preflight target platform does not match authorization")]
    TargetPlatformMismatch,
    #[error("execution preflight target profile does not match authorization")]
    TargetProfileMismatch,
    #[error("execution preflight observation digest does not match authorized pre-state")]
    ObservationDigestMismatch,
    #[error("execution preflight is future-dated: observed at {observed_at_ms} ms but admission time is {now_ms} ms")]
    ObservationFutureDated { observed_at_ms: u64, now_ms: u64 },
    #[error("execution preflight was captured before execution admission")]
    ObservationBeforeExecution {
        observed_at_ms: u64,
        started_at_ms: u64,
    },
    #[error("execution preflight target snapshot is stale: age {age_ms} ms exceeds maximum {max_age_ms} ms")]
    SnapshotStale { age_ms: u64, max_age_ms: u64 },
    #[error("execution preflight is missing an authorized target resource")]
    MissingAuthorizedResource(ResourceRef),
    #[error("execution preflight observation contains an invalid resource")]
    InvalidResource,
    #[error("execution preflight observation digest is incomplete")]
    InvalidObservationDigest,
    #[error("execution preflight target profile is incomplete")]
    InvalidTargetProfile,
}

#[derive(Debug)]
pub struct PreflightedExecutionAuthorization<'a> {
    execution: ExecutionAuthorization<'a>,
    observation_digest: ContentDigest,
    observed_at_ms: u64,
}

impl<'a> PreflightedExecutionAuthorization<'a> {
    pub fn authorized_plan(&self) -> &'a AuthorizedDeploymentPlan {
        self.execution.authorized_plan()
    }

    pub fn authorization_digest(&self) -> &ContentDigest {
        self.execution.authorization_digest()
    }

    pub fn consumed_at_ms(&self) -> u64 {
        self.execution.consumed_at_ms()
    }

    pub fn started_at_ms(&self) -> u64 {
        self.execution.started_at_ms()
    }

    pub fn preflight_observation_digest(&self) -> &ContentDigest {
        &self.observation_digest
    }

    pub fn preflight_observed_at_ms(&self) -> u64 {
        self.observed_at_ms
    }
}

impl<'a> ExecutionAuthorization<'a> {
    pub fn authorized_plan(&self) -> &'a AuthorizedDeploymentPlan {
        self.authorized
    }

    pub fn authorization_digest(&self) -> &ContentDigest {
        &self.authorization_digest
    }

    pub fn consumed_at_ms(&self) -> u64 {
        self.consumed_at_ms
    }

    pub fn started_at_ms(&self) -> u64 {
        self.started_at_ms
    }

    /// Bind the execution authorization to a fresh target observation captured
    /// after execution admission and before the real effect boundary.
    ///
    /// The observer is deliberately supplied by the target-specific layer:
    /// SSC requires a fresh observation contract without knowing how a concrete
    /// platform reads its live state.
    pub fn preflight<O: ExecutionPreflightObserver>(
        self,
        observer: &mut O,
        now_ms: u64,
    ) -> Result<PreflightedExecutionAuthorization<'a>, ExecutionPreflightError<O::Error>> {
        let snapshot = observer
            .observe()
            .map_err(ExecutionPreflightError::Observation)?;

        if snapshot.profile.identity.0.is_empty() || snapshot.profile.platform.is_empty() {
            return Err(ExecutionPreflightError::InvalidTargetProfile);
        }
        if !has_concrete_digest(Some(&snapshot.observation_digest)) {
            return Err(ExecutionPreflightError::InvalidObservationDigest);
        }
        if snapshot
            .resources
            .iter()
            .any(|resource| resource.kind.is_empty() || !has_concrete_digest(Some(&resource.identity)))
        {
            return Err(ExecutionPreflightError::InvalidResource);
        }
        if snapshot.profile.identity != self.authorized.plan.target_snapshot.profile.identity {
            return Err(ExecutionPreflightError::TargetIdentityMismatch);
        }
        if snapshot.profile.platform != self.authorized.plan.target_snapshot.profile.platform {
            return Err(ExecutionPreflightError::TargetPlatformMismatch);
        }

        let authorized_profile_digest = self
            .authorized
            .plan
            .target_snapshot
            .profile
            .digest()
            .map_err(ExecutionPreflightError::Observation)?;
        let observed_profile_digest = snapshot