------------------- MODULE MESH_HANDSHAKE_LIFECYCLE -------------------
EXTENDS FiniteSets, TLC

CONSTANTS
  Profiles,
  ClassicalProfile,
  HybridProfile,
  RequireHybrid,
  DowngradeAuthorized,
  BackendUsable,
  AdmissionMode

ASSUME
  /\ Profiles = {ClassicalProfile, HybridProfile}
  /\ ClassicalProfile # HybridProfile
  /\ RequireHybrid \in BOOLEAN
  /\ DowngradeAuthorized \in BOOLEAN
  /\ BackendUsable \in BOOLEAN
  /\ AdmissionMode \in {"strict", "one-sided", "stub-backend", "downgrade",
                          "identity-unbound", "transcript-unbound"}

VARIABLES
  phase,
  chosenProfile,
  peerIdentityBound,
  transcriptBound,
  candidateSecretPresent,
  initiatorConfirmed,
  responderConfirmed,
  sessionEstablished,
  failedClosed

vars ==
  << phase,
     chosenProfile,
     peerIdentityBound,
     transcriptBound,
     candidateSecretPresent,
     initiatorConfirmed,
     responderConfirmed,
     sessionEstablished,
     failedClosed >>

ProfileAllowed(profile) ==
  \/ profile = HybridProfile
  \/ /\ ~RequireHybrid
     /\ DowngradeAuthorized
     /\ profile = ClassicalProfile

Init ==
  /\ phase = "Idle"
  /\ chosenProfile = "none"
  /\ peerIdentityBound = FALSE
  /\ transcriptBound = FALSE
  /\ candidateSecretPresent = FALSE
  /\ initiatorConfirmed = FALSE
  /\ responderConfirmed = FALSE
  /\ sessionEstablished = FALSE
  /\ failedClosed = FALSE

Start ==
  /\ phase = "Idle"
  /\ phase' = "Negotiating"
  /\ UNCHANGED << chosenProfile,
                  peerIdentityBound,
                  transcriptBound,
                  candidateSecretPresent,
                  initiatorConfirmed,
                  responderConfirmed,
                  sessionEstablished,
                  failedClosed >>

SelectProfile(profile) ==
  /\ phase = "Negotiating"
  /\ profile \in Profiles
  /\ \/ AdmissionMode = "downgrade"
     \/ ProfileAllowed(profile)
  /\ chosenProfile' = profile
  /\ UNCHANGED << phase,
                  peerIdentityBound,
                  transcriptBound,
                  candidateSecretPresent,
                  initiatorConfirmed,
                  responderConfirmed,
                  sessionEstablished,
                  failedClosed >>

BindIdentityAndTranscript ==
  /\ phase = "Negotiating"
  /\ chosenProfile \in Profiles
  /\ \/ AdmissionMode = "downgrade"
     \/ ProfileAllowed(chosenProfile)
  /\ peerIdentityBound' = (AdmissionMode # "identity-unbound")
  /\ transcriptBound' = (AdmissionMode # "transcript-unbound")
  /\ candidateSecretPresent' = (BackendUsable
                               \/ AdmissionMode = "stub-backend")
  /\ phase' = "Candidate"
  /\ UNCHANGED << chosenProfile,
                  initiatorConfirmed,
                  responderConfirmed,
                  sessionEstablished,
                  failedClosed >>

ConfirmInitiator ==
  /\ phase = "Candidate"
  /\ candidateSecretPresent
  /\ (peerIdentityBound \/ AdmissionMode = "identity-unbound")
  /\ (transcriptBound \/ AdmissionMode = "transcript-unbound")
  /\ \/ BackendUsable
     \/ AdmissionMode = "stub-backend"
  /\ initiatorConfirmed' = TRUE
  /\ UNCHANGED << phase,
                  chosenProfile,
                  peerIdentityBound,
                  transcriptBound,
                  candidateSecretPresent,
                  responderConfirmed,
                  sessionEstablished,
                  failedClosed >>

ConfirmResponder ==
  /\ phase = "Candidate"
  /\ candidateSecretPresent
  /\ (peerIdentityBound \/ AdmissionMode = "identity-unbound")
  /\ (transcriptBound \/ AdmissionMode = "transcript-unbound")
  /\ \/ BackendUsable
     \/ AdmissionMode = "stub-backend"
  /\ responderConfirmed' = TRUE
  /\ UNCHANGED << phase,
                  chosenProfile,
                  peerIdentityBound,
                  transcriptBound,
                  candidateSecretPresent,
                  initiatorConfirmed,
                  sessionEstablished,
                  failedClosed >>

Establish ==
  /\ phase = "Candidate"
  /\ candidateSecretPresent
  /\ IF AdmissionMode = "strict"
        THEN /\ BackendUsable
             /\ peerIdentityBound
             /\ transcriptBound
             /\ initiatorConfirmed
             /\ responderConfirmed
             /\ ProfileAllowed(chosenProfile)
        ELSE IF AdmissionMode = "one-sided"
             THEN /\ peerIdentityBound
                  /\ transcriptBound
                  /\ initiatorConfirmed
             ELSE IF AdmissionMode = "stub-backend"
                  THEN TRUE
                  ELSE IF AdmissionMode = "identity-unbound"
                       THEN /\ transcriptBound
                            /\ initiatorConfirmed
                            /\ responderConfirmed
                            /\ ProfileAllowed(chosenProfile)
                       ELSE IF AdmissionMode = "transcript-unbound"
                            THEN /\ peerIdentityBound
                                 /\ initiatorConfirmed
                                 /\ responderConfirmed
                                 /\ ProfileAllowed(chosenProfile)
                            ELSE /\ BackendUsable
                                 /\ peerIdentityBound
                                 /\ transcriptBound
                                 /\ initiatorConfirmed
                                 /\ responderConfirmed
  /\ sessionEstablished' = TRUE
  /\ phase' = "Established"
  /\ UNCHANGED << chosenProfile,
                  peerIdentityBound,
                  transcriptBound,
                  candidateSecretPresent,
                  initiatorConfirmed,
                  responderConfirmed,
                  failedClosed >>

Abort ==
  /\ phase \in {"Negotiating", "Candidate"}
  /\ phase' = "Closed"
  /\ candidateSecretPresent' = FALSE
  /\ initiatorConfirmed' = FALSE
  /\ responderConfirmed' = FALSE
  /\ failedClosed' = TRUE
  /\ UNCHANGED << chosenProfile,
                  peerIdentityBound,
                  transcriptBound,
                  sessionEstablished >>

\* Completed/failed sessions are valid terminal protocol states, not deadlocks.
Terminal ==
  /\ phase \in {"Established", "Closed"}
  /\ UNCHANGED vars

Next ==
  \/ Start
  \/ \E profile \in Profiles : SelectProfile(profile)
  \/ BindIdentityAndTranscript
  \/ ConfirmInitiator
  \/ ConfirmResponder
  \/ Establish
  \/ Abort
  \/ Terminal

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ phase \in {"Idle", "Negotiating", "Candidate", "Established", "Closed"}
  /\ chosenProfile \in Profiles \cup {"none"}
  /\ peerIdentityBound \in BOOLEAN
  /\ transcriptBound \in BOOLEAN
  /\ candidateSecretPresent \in BOOLEAN
  /\ initiatorConfirmed \in BOOLEAN
  /\ responderConfirmed \in BOOLEAN
  /\ sessionEstablished \in BOOLEAN
  /\ failedClosed \in BOOLEAN

EstablishedSound ==
  ~sessionEstablished
  \/ /\ candidateSecretPresent
     /\ BackendUsable
     /\ peerIdentityBound
     /\ transcriptBound
     /\ initiatorConfirmed
     /\ responderConfirmed
     /\ ProfileAllowed(chosenProfile)
     /\ phase = "Established"

FailedStateErasesCandidate ==
  ~failedClosed \/ ~candidateSecretPresent

UnavailableBackendNeverEstablishes ==
  BackendUsable \/ ~sessionEstablished

=============================================================================
