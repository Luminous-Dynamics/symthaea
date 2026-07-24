# Maintenance and Diagnostic Isolation

Series 45 treats servicing as a separate, non-actuating lifecycle. A maintenance
session is accepted only when the physical service key is present, the power
stage is isolated, the emergency-stop loop has been verified, deployment and
calibration identities match, and the permit is fresh and authenticated.

Capabilities are explicit. Read-only diagnostics, redacted trace export, and
non-actuating self-test do not imply permission to stage calibration or software.
Mutating operations require a distinct independent approver. A maintenance
session always reports `actuation_inhibited = true` and
`power_enable_permitted = false`.

A permit cannot clear a latched physical inspection, arm the device, or bypass
the independent safety controller. Diagnostic data retention and privacy policy
remain deployment responsibilities.
