// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "./TaxBracketVerifier.sol";

/**
 * @title TaxBracketVerifierTest
 * @notice Foundry tests for the TaxBracketVerifier contract
 */
contract TaxBracketVerifierTest is Test {
    TaxBracketVerifier public verifier;
    address public user1 = address(0x1);
    address public user2 = address(0x2);

    // Known test values (from Rust SDK tests)
    uint64 constant US_2024_BRACKET_2_LOWER = 47_150;
    uint64 constant US_2024_BRACKET_2_UPPER = 100_525;
    uint32 constant TAX_YEAR_2024 = 2024;

    function setUp() public {
        verifier = new TaxBracketVerifier();
    }

    // =============================================================================
    // COMMITMENT TESTS
    // =============================================================================

    function test_ComputeCommitment_Deterministic() public view {
        bytes32 c1 = verifier.computeCommitment(US_2024_BRACKET_2_LOWER, US_2024_BRACKET_2_UPPER, TAX_YEAR_2024);
        bytes32 c2 = verifier.computeCommitment(US_2024_BRACKET_2_LOWER, US_2024_BRACKET_2_UPPER, TAX_YEAR_2024);
        assertEq(c1, c2, "Commitments should be deterministic");
    }

    function test_ComputeCommitment_DiffersByYear() public view {
        bytes32 c2024 = verifier.computeCommitment(US_2024_BRACKET_2_LOWER, US_2024_BRACKET_2_UPPER, 2024);
        bytes32 c2025 = verifier.computeCommitment(US_2024_BRACKET_2_LOWER, US_2024_BRACKET_2_UPPER, 2025);
        assertTrue(c2024 != c2025, "Commitments should differ by year");
    }

    function test_ComputeCommitment_DiffersByBounds() public view {
        bytes32 c1 = verifier.computeCommitment(0, 11_600, TAX_YEAR_2024);
        bytes32 c2 = verifier.computeCommitment(11_600, 47_150, TAX_YEAR_2024);
        assertTrue(c1 != c2, "Commitments should differ by bounds");
    }

    // =============================================================================
    // PROOF VERIFICATION TESTS
    // =============================================================================

    function test_VerifyProof_DevMode() public {
        bytes32 commitment = verifier.computeCommitment(US_2024_BRACKET_2_LOWER, US_2024_BRACKET_2_UPPER, TAX_YEAR_2024);

        TaxBracketVerifier.ProofInput memory input = TaxBracketVerifier.ProofInput({
            jurisdiction: TaxBracketVerifier.Jurisdiction.US,
            taxYear: TAX_YEAR_2024,
            bracketIndex: 2,
            rateBps: 2200,
            bracketLower: US_2024_BRACKET_2_LOWER,
            bracketUpper: US_2024_BRACKET_2_UPPER,
            commitment: commitment,
            isDevMode: true,
            risc0Receipt: ""
        });

        vm.prank(user1);
        bool success = verifier.verifyProof(input);
        assertTrue(success, "Dev mode proof should verify");

        // Check stored proof
        (TaxBracketVerifier.VerifiedProof memory proof, bool found) = verifier.getLatestProof(
            user1,
            TaxBracketVerifier.Jurisdiction.US,
            TAX_YEAR_2024
        );
        assertTrue(found, "Proof should be found");
        assertEq(proof.bracketIndex, 2);
        assertEq(proof.rateBps, 2200);
    }

    function test_VerifyProof_InvalidCommitment_Reverts() public {
        bytes32 wrongCommitment = bytes32(uint256(0x123));

        TaxBracketVerifier.ProofInput memory input = TaxBracketVerifier.ProofInput({
            jurisdiction: TaxBracketVerifier.Jurisdiction.US,
            taxYear: TAX_YEAR_2024,
            bracketIndex: 2,
            rateBps: 2200,
            bracketLower: US_2024_BRACKET_2_LOWER,
            bracketUpper: US_2024_BRACKET_2_UPPER,
            commitment: wrongCommitment,
            isDevMode: true,
            risc0Receipt: ""
        });

        bytes32 expectedCommitment = verifier.computeCommitment(US_2024_BRACKET_2_LOWER, US_2024_BRACKET_2_UPPER, TAX_YEAR_2024);

        vm.prank(user1);
        vm.expectRevert(abi.encodeWithSelector(
            TaxBracketVerifier.InvalidCommitment.selector,
            expectedCommitment,
            wrongCommitment
        ));
        verifier.verifyProof(input);
    }

    function test_VerifyProof_InvalidBounds_Reverts() public {
        TaxBracketVerifier.ProofInput memory input = TaxBracketVerifier.ProofInput({
            jurisdiction: TaxBracketVerifier.Jurisdiction.US,
            taxYear: TAX_YEAR_2024,
            bracketIndex: 0,
            rateBps: 1000,
            bracketLower: 100_000,  // Lower > Upper (invalid)
            bracketUpper: 50_000,
            commitment: bytes32(0),
            isDevMode: true,
            risc0Receipt: ""
        });

        vm.prank(user1);
        vm.expectRevert(abi.encodeWithSelector(
            TaxBracketVerifier.InvalidBracketBounds.selector,
            100_000,
            50_000
        ));
        verifier.verifyProof(input);
    }

    function test_VerifyProof_InvalidYear_Reverts() public {
        TaxBracketVerifier.ProofInput memory input = TaxBracketVerifier.ProofInput({
            jurisdiction: TaxBracketVerifier.Jurisdiction.US,
            taxYear: 2019,  // Too old
            bracketIndex: 0,
            rateBps: 1000,
            bracketLower: 0,
            bracketUpper: 11_600,
            commitment: bytes32(0),
            isDevMode: true,
            risc0Receipt: ""
        });

        vm.prank(user1);
        vm.expectRevert(abi.encodeWithSelector(
            TaxBracketVerifier.InvalidTaxYear.selector,
            2019
        ));
        verifier.verifyProof(input);
    }

    function test_VerifyProof_Duplicate_Reverts() public {
        bytes32 commitment = verifier.computeCommitment(US_2024_BRACKET_2_LOWER, US_2024_BRACKET_2_UPPER, TAX_YEAR_2024);

        TaxBracketVerifier.ProofInput memory input = TaxBracketVerifier.ProofInput({
            jurisdiction: TaxBracketVerifier.Jurisdiction.US,
            taxYear: TAX_YEAR_2024,
            bracketIndex: 2,
            rateBps: 2200,
            bracketLower: US_2024_BRACKET_2_LOWER,
            bracketUpper: US_2024_BRACKET_2_UPPER,
            commitment: commitment,
            isDevMode: true,
            risc0Receipt: ""
        });

        vm.prank(user1);
        verifier.verifyProof(input);

        // Try to verify same proof again (from different user)
        vm.prank(user2);
        vm.expectRevert(abi.encodeWithSelector(
            TaxBracketVerifier.ProofAlreadyVerified.selector,
            commitment
        ));
        verifier.verifyProof(input);
    }

    // =============================================================================
    // QUERY TESTS
    // =============================================================================

    function test_HasProofInRange() public {
        // Submit proof for bracket 2 (22%)
        bytes32 commitment = verifier.computeCommitment(US_2024_BRACKET_2_LOWER, US_2024_BRACKET_2_UPPER, TAX_YEAR_2024);

        TaxBracketVerifier.ProofInput memory input = TaxBracketVerifier.ProofInput({
            jurisdiction: TaxBracketVerifier.Jurisdiction.US,
            taxYear: TAX_YEAR_2024,
            bracketIndex: 2,
            rateBps: 2200,
            bracketLower: US_2024_BRACKET_2_LOWER,
            bracketUpper: US_2024_BRACKET_2_UPPER,
            commitment: commitment,
            isDevMode: true,
            risc0Receipt: ""
        });

        vm.prank(user1);
        verifier.verifyProof(input);

        // Should be in range [0, 3]
        assertTrue(
            verifier.hasProofInRange(user1, TaxBracketVerifier.Jurisdiction.US, TAX_YEAR_2024, 0, 3),
            "Should have proof in range 0-3"
        );

        // Should be in range [2, 2]
        assertTrue(
            verifier.hasProofInRange(user1, TaxBracketVerifier.Jurisdiction.US, TAX_YEAR_2024, 2, 2),
            "Should have proof in exact bracket"
        );

        // Should NOT be in range [3, 6]
        assertFalse(
            verifier.hasProofInRange(user1, TaxBracketVerifier.Jurisdiction.US, TAX_YEAR_2024, 3, 6),
            "Should not have proof in range 3-6"
        );

        // User2 should have no proofs
        assertFalse(
            verifier.hasProofInRange(user2, TaxBracketVerifier.Jurisdiction.US, TAX_YEAR_2024, 0, 6),
            "User2 should have no proofs"
        );
    }

    // =============================================================================
    // MULTI-JURISDICTION TESTS
    // =============================================================================

    function test_MultipleJurisdictions() public {
        // US proof
        bytes32 usCommitment = verifier.computeCommitment(US_2024_BRACKET_2_LOWER, US_2024_BRACKET_2_UPPER, TAX_YEAR_2024);
        vm.prank(user1);
        verifier.verifyProof(TaxBracketVerifier.ProofInput({
            jurisdiction: TaxBracketVerifier.Jurisdiction.US,
            taxYear: TAX_YEAR_2024,
            bracketIndex: 2,
            rateBps: 2200,
            bracketLower: US_2024_BRACKET_2_LOWER,
            bracketUpper: US_2024_BRACKET_2_UPPER,
            commitment: usCommitment,
            isDevMode: true,
            risc0Receipt: ""
        }));

        // UK proof (different bounds)
        uint64 ukLower = 50_270;
        uint64 ukUpper = 125_140;
        bytes32 ukCommitment = verifier.computeCommitment(ukLower, ukUpper, TAX_YEAR_2024);
        vm.prank(user1);
        verifier.verifyProof(TaxBracketVerifier.ProofInput({
            jurisdiction: TaxBracketVerifier.Jurisdiction.UK,
            taxYear: TAX_YEAR_2024,
            bracketIndex: 2,
            rateBps: 4000,
            bracketLower: ukLower,
            bracketUpper: ukUpper,
            commitment: ukCommitment,
            isDevMode: true,
            risc0Receipt: ""
        }));

        // Check both exist
        TaxBracketVerifier.VerifiedProof[] memory proofs = verifier.getProofsForAddress(user1);
        assertEq(proofs.length, 2, "Should have 2 proofs");

        // Check each can be queried
        assertTrue(
            verifier.hasProofInRange(user1, TaxBracketVerifier.Jurisdiction.US, TAX_YEAR_2024, 2, 2),
            "Should have US proof"
        );
        assertTrue(
            verifier.hasProofInRange(user1, TaxBracketVerifier.Jurisdiction.UK, TAX_YEAR_2024, 2, 2),
            "Should have UK proof"
        );
    }

    // =============================================================================
    // HELPER FUNCTION TESTS
    // =============================================================================

    function test_JurisdictionName_AllG20() public view {
        // Americas
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.US), "United States");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.CA), "Canada");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.MX), "Mexico");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.BR), "Brazil");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.AR), "Argentina");
        // Europe
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.UK), "United Kingdom");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.DE), "Germany");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.FR), "France");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.IT), "Italy");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.RU), "Russia");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.TR), "Turkey");
        // Asia-Pacific
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.JP), "Japan");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.CN), "China");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.IN_), "India");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.KR), "South Korea");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.ID_), "Indonesia");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.AU), "Australia");
        // Middle East & Africa
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.SA), "Saudi Arabia");
        assertEq(verifier.jurisdictionName(TaxBracketVerifier.Jurisdiction.ZA), "South Africa");
    }

    function test_SaudiArabiaZeroTax() public {
        // Saudi Arabia has 0% income tax
        bytes32 commitment = verifier.computeCommitment(0, type(uint64).max, TAX_YEAR_2024);

        vm.prank(user1);
        verifier.verifyProof(TaxBracketVerifier.ProofInput({
            jurisdiction: TaxBracketVerifier.Jurisdiction.SA,
            taxYear: TAX_YEAR_2024,
            bracketIndex: 0,
            rateBps: 0,  // 0% tax
            bracketLower: 0,
            bracketUpper: type(uint64).max,
            commitment: commitment,
            isDevMode: true,
            risc0Receipt: ""
        }));

        assertTrue(
            verifier.hasProofInRange(user1, TaxBracketVerifier.Jurisdiction.SA, TAX_YEAR_2024, 0, 0),
            "Should have Saudi Arabia 0% tax proof"
        );
    }

    function test_ChinaProof() public {
        // China 20% bracket: 144,000 - 300,000 CNY
        uint64 cnLower = 144_000;
        uint64 cnUpper = 300_000;
        bytes32 commitment = verifier.computeCommitment(cnLower, cnUpper, TAX_YEAR_2024);

        vm.prank(user1);
        verifier.verifyProof(TaxBracketVerifier.ProofInput({
            jurisdiction: TaxBracketVerifier.Jurisdiction.CN,
            taxYear: TAX_YEAR_2024,
            bracketIndex: 2,
            rateBps: 2000,  // 20%
            bracketLower: cnLower,
            bracketUpper: cnUpper,
            commitment: commitment,
            isDevMode: true,
            risc0Receipt: ""
        }));

        (TaxBracketVerifier.VerifiedProof memory proof, bool found) = verifier.getLatestProof(
            user1,
            TaxBracketVerifier.Jurisdiction.CN,
            TAX_YEAR_2024
        );
        assertTrue(found, "China proof should be found");
        assertEq(proof.rateBps, 2000);
    }

    function test_RateToPercent() public view {
        assertEq(verifier.rateToPercent(2200), "22.0");
        assertEq(verifier.rateToPercent(1000), "10.0");
        assertEq(verifier.rateToPercent(3700), "37.0");
        assertEq(verifier.rateToPercent(750), "7.5");
    }

    // =============================================================================
    // ADMIN TESTS
    // =============================================================================

    function test_SetRisc0Verifier() public {
        address newVerifier = address(0x999);
        verifier.setRisc0Verifier(newVerifier);
        assertEq(verifier.risc0Verifier(), newVerifier);
    }

    function test_SetRisc0Verifier_OnlyOwner() public {
        vm.prank(user1);
        vm.expectRevert(TaxBracketVerifier.OnlyOwner.selector);
        verifier.setRisc0Verifier(address(0x999));
    }

    function test_TransferOwnership() public {
        verifier.transferOwnership(user1);
        assertEq(verifier.owner(), user1);

        vm.prank(user1);
        verifier.setRisc0Verifier(address(0x999));
        assertEq(verifier.risc0Verifier(), address(0x999));
    }
}
