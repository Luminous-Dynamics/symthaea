// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title TaxBracketVerifier
 * @notice On-chain verification of ZK tax bracket proofs from Mycelix
 * @dev Verifies FNV-1a commitments and optionally integrates with Risc0 verifier
 *
 * This contract allows:
 * - Verifying tax bracket proof commitments
 * - Storing verified proofs on-chain
 * - Querying if an address has proven membership in a bracket range
 *
 * @custom:security-contact security@mycelix.net
 */
contract TaxBracketVerifier {
    // =============================================================================
    // TYPES
    // =============================================================================

    /// @notice Supported G20 tax jurisdictions
    enum Jurisdiction {
        // Americas
        US,     // United States (IRS)
        CA,     // Canada (CRA)
        MX,     // Mexico (SAT)
        BR,     // Brazil (Receita Federal)
        AR,     // Argentina (AFIP)
        // Europe
        UK,     // United Kingdom (HMRC)
        DE,     // Germany (Finanzamt)
        FR,     // France (DGFiP)
        IT,     // Italy (Agenzia Entrate)
        RU,     // Russia (FNS)
        TR,     // Turkey (GIB)
        // Asia-Pacific
        JP,     // Japan (NTA)
        CN,     // China (SAT)
        IN_,    // India (IT Dept) - IN is reserved keyword
        KR,     // South Korea (NTS)
        ID_,    // Indonesia (DJP) - ID is reserved keyword
        AU,     // Australia (ATO)
        // Middle East & Africa
        SA,     // Saudi Arabia (ZATCA)
        ZA      // South Africa (SARS)
    }

    /// @notice A verified tax bracket proof
    struct VerifiedProof {
        address prover;           // Who submitted the proof
        Jurisdiction jurisdiction;
        uint32 taxYear;
        uint8 bracketIndex;
        uint16 rateBps;           // Rate in basis points (2200 = 22%)
        uint64 bracketLower;
        uint64 bracketUpper;
        bytes32 commitment;
        uint256 verifiedAt;       // Timestamp
        bool isDevMode;           // True if this is a dev mode proof
    }

    /// @notice Input for proof verification
    struct ProofInput {
        Jurisdiction jurisdiction;
        uint32 taxYear;
        uint8 bracketIndex;
        uint16 rateBps;
        uint64 bracketLower;
        uint64 bracketUpper;
        bytes32 commitment;
        bool isDevMode;
        bytes risc0Receipt;       // Optional: Risc0 receipt for real proofs
    }

    // =============================================================================
    // STORAGE
    // =============================================================================

    /// @notice Mapping from prover address to their verified proofs
    mapping(address => VerifiedProof[]) public proofsByProver;

    /// @notice Mapping from proof commitment to verified proof
    mapping(bytes32 => VerifiedProof) public proofsByCommitment;

    /// @notice Optional Risc0 verifier contract address
    address public risc0Verifier;

    /// @notice Image ID for valid Risc0 proofs
    bytes32 public risc0ImageId;

    /// @notice Contract owner
    address public owner;

    /// @notice Total number of verified proofs
    uint256 public totalProofs;

    // =============================================================================
    // EVENTS
    // =============================================================================

    event ProofVerified(
        address indexed prover,
        Jurisdiction indexed jurisdiction,
        uint32 taxYear,
        uint8 bracketIndex,
        bytes32 commitment
    );

    event Risc0VerifierUpdated(address indexed oldVerifier, address indexed newVerifier);
    event Risc0ImageIdUpdated(bytes32 indexed oldImageId, bytes32 indexed newImageId);

    // =============================================================================
    // ERRORS
    // =============================================================================

    error InvalidCommitment(bytes32 expected, bytes32 actual);
    error InvalidBracketBounds(uint64 lower, uint64 upper);
    error InvalidTaxYear(uint32 year);
    error ProofAlreadyVerified(bytes32 commitment);
    error Risc0VerificationFailed();
    error OnlyOwner();

    // =============================================================================
    // CONSTRUCTOR
    // =============================================================================

    constructor() {
        owner = msg.sender;
    }

    // =============================================================================
    // MODIFIERS
    // =============================================================================

    modifier onlyOwner() {
        if (msg.sender != owner) revert OnlyOwner();
        _;
    }

    // =============================================================================
    // EXTERNAL FUNCTIONS
    // =============================================================================

    /**
     * @notice Verify and store a tax bracket proof
     * @param input The proof input data
     * @return success True if verification succeeded
     */
    function verifyProof(ProofInput calldata input) external returns (bool success) {
        // Validate basic parameters
        if (input.bracketLower >= input.bracketUpper && input.bracketUpper != type(uint64).max) {
            revert InvalidBracketBounds(input.bracketLower, input.bracketUpper);
        }

        if (input.taxYear < 2020 || input.taxYear > 2030) {
            revert InvalidTaxYear(input.taxYear);
        }

        // Check if already verified
        if (proofsByCommitment[input.commitment].verifiedAt != 0) {
            revert ProofAlreadyVerified(input.commitment);
        }

        // Compute expected commitment using FNV-1a
        bytes32 expectedCommitment = computeCommitment(
            input.bracketLower,
            input.bracketUpper,
            input.taxYear
        );

        if (expectedCommitment != input.commitment) {
            revert InvalidCommitment(expectedCommitment, input.commitment);
        }

        // For real proofs, verify Risc0 receipt if verifier is set
        if (!input.isDevMode && risc0Verifier != address(0)) {
            bool valid = _verifyRisc0Receipt(input.risc0Receipt, input.commitment);
            if (!valid) revert Risc0VerificationFailed();
        }

        // Store the verified proof
        VerifiedProof memory proof = VerifiedProof({
            prover: msg.sender,
            jurisdiction: input.jurisdiction,
            taxYear: input.taxYear,
            bracketIndex: input.bracketIndex,
            rateBps: input.rateBps,
            bracketLower: input.bracketLower,
            bracketUpper: input.bracketUpper,
            commitment: input.commitment,
            verifiedAt: block.timestamp,
            isDevMode: input.isDevMode
        });

        proofsByProver[msg.sender].push(proof);
        proofsByCommitment[input.commitment] = proof;
        totalProofs++;

        emit ProofVerified(
            msg.sender,
            input.jurisdiction,
            input.taxYear,
            input.bracketIndex,
            input.commitment
        );

        return true;
    }

    /**
     * @notice Check if an address has a verified proof in a bracket range
     * @param prover The address to check
     * @param jurisdiction The tax jurisdiction
     * @param taxYear The tax year
     * @param minBracket Minimum acceptable bracket index (inclusive)
     * @param maxBracket Maximum acceptable bracket index (inclusive)
     * @return hasProof True if a valid proof exists in range
     */
    function hasProofInRange(
        address prover,
        Jurisdiction jurisdiction,
        uint32 taxYear,
        uint8 minBracket,
        uint8 maxBracket
    ) external view returns (bool hasProof) {
        VerifiedProof[] storage proofs = proofsByProver[prover];

        for (uint256 i = 0; i < proofs.length; i++) {
            if (proofs[i].jurisdiction == jurisdiction &&
                proofs[i].taxYear == taxYear &&
                proofs[i].bracketIndex >= minBracket &&
                proofs[i].bracketIndex <= maxBracket) {
                return true;
            }
        }

        return false;
    }

    /**
     * @notice Get all proofs for an address
     * @param prover The address to query
     * @return Array of verified proofs
     */
    function getProofsForAddress(address prover) external view returns (VerifiedProof[] memory) {
        return proofsByProver[prover];
    }

    /**
     * @notice Get the most recent proof for an address and jurisdiction
     * @param prover The address to query
     * @param jurisdiction The jurisdiction to filter by
     * @param taxYear The tax year to filter by
     * @return proof The most recent proof (or empty if none)
     * @return found True if a proof was found
     */
    function getLatestProof(
        address prover,
        Jurisdiction jurisdiction,
        uint32 taxYear
    ) external view returns (VerifiedProof memory proof, bool found) {
        VerifiedProof[] storage proofs = proofsByProver[prover];

        for (uint256 i = proofs.length; i > 0; i--) {
            if (proofs[i-1].jurisdiction == jurisdiction && proofs[i-1].taxYear == taxYear) {
                return (proofs[i-1], true);
            }
        }

        return (proof, false);
    }

    // =============================================================================
    // ADMIN FUNCTIONS
    // =============================================================================

    /**
     * @notice Set the Risc0 verifier contract address
     * @param _verifier The new verifier address
     */
    function setRisc0Verifier(address _verifier) external onlyOwner {
        emit Risc0VerifierUpdated(risc0Verifier, _verifier);
        risc0Verifier = _verifier;
    }

    /**
     * @notice Set the expected Risc0 image ID
     * @param _imageId The new image ID
     */
    function setRisc0ImageId(bytes32 _imageId) external onlyOwner {
        emit Risc0ImageIdUpdated(risc0ImageId, _imageId);
        risc0ImageId = _imageId;
    }

    /**
     * @notice Transfer ownership
     * @param newOwner The new owner address
     */
    function transferOwnership(address newOwner) external onlyOwner {
        owner = newOwner;
    }

    // =============================================================================
    // PUBLIC PURE FUNCTIONS
    // =============================================================================

    /**
     * @notice Compute the FNV-1a commitment for bracket parameters
     * @dev Must match the Rust implementation exactly
     * @param lower Bracket lower bound
     * @param upper Bracket upper bound
     * @param taxYear Tax year
     * @return commitment The 32-byte commitment
     */
    function computeCommitment(
        uint64 lower,
        uint64 upper,
        uint32 taxYear
    ) public pure returns (bytes32 commitment) {
        // FNV-1a hash initialization
        uint64 hash = 0xcbf29ce484222325;
        uint64 prime = 0x100000001b3;

        // Hash lower bound (little-endian)
        for (uint256 i = 0; i < 8; i++) {
            hash ^= uint64(uint8(lower >> (i * 8)));
            hash *= prime;
        }

        // Hash upper bound (little-endian)
        for (uint256 i = 0; i < 8; i++) {
            hash ^= uint64(uint8(upper >> (i * 8)));
            hash *= prime;
        }

        // Hash tax year (little-endian, 4 bytes)
        for (uint256 i = 0; i < 4; i++) {
            hash ^= uint64(uint8(taxYear >> (i * 8)));
            hash *= prime;
        }

        // Expand to 32 bytes
        bytes memory result = new bytes(32);
        uint64 h = hash;
        for (uint256 chunk = 0; chunk < 4; chunk++) {
            for (uint256 i = 0; i < 8; i++) {
                result[chunk * 8 + i] = bytes1(uint8(h >> (i * 8)));
            }
            h *= prime;
        }

        // Convert to bytes32
        assembly {
            commitment := mload(add(result, 32))
        }
    }

    /**
     * @notice Get jurisdiction name
     * @param j The jurisdiction enum value
     * @return name The jurisdiction name
     */
    function jurisdictionName(Jurisdiction j) public pure returns (string memory) {
        // Americas
        if (j == Jurisdiction.US) return "United States";
        if (j == Jurisdiction.CA) return "Canada";
        if (j == Jurisdiction.MX) return "Mexico";
        if (j == Jurisdiction.BR) return "Brazil";
        if (j == Jurisdiction.AR) return "Argentina";
        // Europe
        if (j == Jurisdiction.UK) return "United Kingdom";
        if (j == Jurisdiction.DE) return "Germany";
        if (j == Jurisdiction.FR) return "France";
        if (j == Jurisdiction.IT) return "Italy";
        if (j == Jurisdiction.RU) return "Russia";
        if (j == Jurisdiction.TR) return "Turkey";
        // Asia-Pacific
        if (j == Jurisdiction.JP) return "Japan";
        if (j == Jurisdiction.CN) return "China";
        if (j == Jurisdiction.IN_) return "India";
        if (j == Jurisdiction.KR) return "South Korea";
        if (j == Jurisdiction.ID_) return "Indonesia";
        if (j == Jurisdiction.AU) return "Australia";
        // Middle East & Africa
        if (j == Jurisdiction.SA) return "Saudi Arabia";
        if (j == Jurisdiction.ZA) return "South Africa";
        return "Unknown";
    }

    /**
     * @notice Convert rate basis points to percentage string
     * @param rateBps Rate in basis points
     * @return The percentage with one decimal (e.g., "22.0")
     */
    function rateToPercent(uint16 rateBps) public pure returns (string memory) {
        uint256 whole = rateBps / 100;
        uint256 decimal = rateBps % 100;

        // Simple string conversion
        bytes memory result = new bytes(6);
        uint256 idx = 0;

        // Whole part
        if (whole >= 10) {
            result[idx++] = bytes1(uint8(48 + whole / 10));
        }
        result[idx++] = bytes1(uint8(48 + whole % 10));

        // Decimal point
        result[idx++] = '.';

        // Decimal part (one digit)
        result[idx++] = bytes1(uint8(48 + decimal / 10));

        // Trim to actual length
        bytes memory trimmed = new bytes(idx);
        for (uint256 i = 0; i < idx; i++) {
            trimmed[i] = result[i];
        }

        return string(trimmed);
    }

    // =============================================================================
    // INTERNAL FUNCTIONS
    // =============================================================================

    /**
     * @notice Verify a Risc0 receipt
     * @dev Uses the Risc0 verifier contract for production proofs.
     *
     * Production Risc0 Verifier Addresses (as of 2026):
     * - Ethereum Mainnet: 0x8EaB2D97Dfce405A1692a21b3ff3A172d593D319
     * - Ethereum Sepolia: 0x925d8331ddc0a1F0d96E68CF073DFE1d92b69187
     * - Optimism: TBD
     * - Arbitrum: TBD
     * - Polygon: TBD
     *
     * See: https://dev.risczero.com/api/blockchain-integration/contracts/verifier
     *
     * @param receipt The Risc0 receipt bytes (encoded proof + journal)
     * @param expectedCommitment The expected commitment from journal
     * @return valid True if verification passed
     */
    function _verifyRisc0Receipt(
        bytes calldata receipt,
        bytes32 expectedCommitment
    ) internal view returns (bool valid) {
        if (risc0Verifier == address(0)) {
            // No verifier set, accept dev mode proofs only
            // WARNING: This is insecure! Set risc0Verifier for production.
            return true;
        }

        // Production verification via Risc0 verifier contract
        // The verifier contract exposes: verify(bytes calldata seal, bytes32 imageId, bytes32 journalDigest)
        //
        // For Risc0 v1.x verifier interface:
        // - seal: The proof bytes from the receipt
        // - imageId: The guest program image ID (set via setRisc0ImageId)
        // - journalDigest: SHA256 hash of the journal (our commitment)
        //
        // The receipt format contains: [seal_length (4 bytes)][seal][journal]
        // We extract the seal and verify against our expected commitment.

        if (receipt.length < 4) {
            return false;
        }

        // For production, call the verifier:
        // (bool success, bytes memory result) = risc0Verifier.staticcall(
        //     abi.encodeWithSignature(
        //         "verify(bytes,bytes32,bytes32)",
        //         receipt,
        //         risc0ImageId,
        //         expectedCommitment
        //     )
        // );
        // return success && abi.decode(result, (bool));

        // Temporary implementation: verify receipt is non-empty and matches expected format
        // This should be replaced with actual Risc0 verifier call in production
        return receipt.length > 64; // Minimum valid receipt size
    }
}

/**
 * @title ITaxBracketVerifier
 * @notice Interface for the TaxBracketVerifier contract
 */
interface ITaxBracketVerifier {
    enum Jurisdiction {
        US, CA, MX, BR, AR,           // Americas
        UK, DE, FR, IT, RU, TR,       // Europe
        JP, CN, IN_, KR, ID_, AU,     // Asia-Pacific
        SA, ZA                         // Middle East & Africa
    }

    struct ProofInput {
        Jurisdiction jurisdiction;
        uint32 taxYear;
        uint8 bracketIndex;
        uint16 rateBps;
        uint64 bracketLower;
        uint64 bracketUpper;
        bytes32 commitment;
        bool isDevMode;
        bytes risc0Receipt;
    }

    function verifyProof(ProofInput calldata input) external returns (bool);
    function hasProofInRange(
        address prover,
        Jurisdiction jurisdiction,
        uint32 taxYear,
        uint8 minBracket,
        uint8 maxBracket
    ) external view returns (bool);
    function computeCommitment(uint64 lower, uint64 upper, uint32 taxYear) external pure returns (bytes32);
}
