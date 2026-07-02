// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ProofLog {
    struct Attestation {
        string proofHash;
        uint256 consensusScore;
        uint256 timestamp;
        address submitter;
    }

    mapping(string => Attestation[]) private attestationsByProof;

    event AttestationRecorded(string proofHash, uint256 consensusScore, address submitter);

    function recordAttestation(string memory proofHash, uint256 consensusScore) external {
        Attestation memory att = Attestation({
            proofHash: proofHash,
            consensusScore: consensusScore,
            timestamp: block.timestamp,
            submitter: msg.sender
        });
        attestationsByProof[proofHash].push(att);
        emit AttestationRecorded(proofHash, consensusScore, msg.sender);
    }

    function getAttestations(string memory proofHash) external view returns (Attestation[] memory) {
        return attestationsByProof[proofHash];
    }
}
