    pub unit: String,
    /// Unique D-Bus owner of org.freedesktop.systemd1 for this job epoch.
    ///
    /// Unique names are connection-scoped and never change owner, so retaining
    /// this value prevents a durable receipt from collapsing two systemd
    /// manager incarnations that happen to reuse other job identifiers.
    pub manager_owner: String,
    /// Job object path returned by systemd.
    pub object_path: String,
    /// Monotonic timestamp when the observer received the matching JobRemoved signal.
    /// This is local observation-order evidence, not a systemd-provided event timestamp.
    #[serde(default)]
    pub removed_at_monotonic_us: Option<u64>,
    /// systemd JobRemoved result. Only the exact `done` value is accepted as
    /// successful evidence; unknown future vocabulary remains recordable but
    /// cannot satisfy the proof predicate.
    pub result: String,
}

impl NixSystemdJobEvidenceV1 {
    pub fn validate_shape(&self) -> Result<(), NixPostStateErrorV1> {
        if self.id == 0 {
            return Err(NixPostStateErrorV1::InvalidJobId);
        }
        NixServiceOperationV1::new(self.unit.clone(), NixServiceOperationKindV1::Start)
            .map_err(|_| NixPostStateErrorV1::InvalidJobUnit)?;
        if self.unit != self.unit.trim() {
            return Err(NixPostStateErrorV1::InvalidJobUnit);
        }
        validate_unique_manager_owner(&self.manager_owner)?;
        require_nonempty(&self.object_path, "systemd job object path")?;
        if !self
            .object_path
            .starts_with("/org/freedesktop/systemd1/job/")
        {