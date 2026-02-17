// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

/**
 * @title GasOptimized
 * @notice Library of gas-optimized utility functions
 * @dev Collection of common operations optimized for minimal gas usage
 */
library GasOptimized {
    // ============================================================
    // Math Operations
    // ============================================================

    /// @notice Unchecked increment for loop counters
    function uncheckedInc(uint256 i) internal pure returns (uint256) {
        unchecked { return i + 1; }
    }

    /// @notice Unchecked decrement for loop counters
    function uncheckedDec(uint256 i) internal pure returns (uint256) {
        unchecked { return i - 1; }
    }

    /// @notice Safe percentage calculation (basis points)
    /// @param amount The base amount
    /// @param bps Basis points (10000 = 100%)
    function percentage(uint256 amount, uint256 bps) internal pure returns (uint256) {
        unchecked {
            return (amount * bps) / 10000;
        }
    }

    /// @notice Calculate share with rounding
    /// @param amount Total amount to split
    /// @param numerator Share numerator
    /// @param denominator Share denominator
    /// @param roundUp Whether to round up
    function calculateShare(
        uint256 amount,
        uint256 numerator,
        uint256 denominator,
        bool roundUp
    ) internal pure returns (uint256) {
        if (roundUp) {
            return (amount * numerator + denominator - 1) / denominator;
        }
        unchecked {
            return (amount * numerator) / denominator;
        }
    }

    /// @notice Integer square root (Babylonian method) - optimized
    function sqrt(uint256 x) internal pure returns (uint256 y) {
        if (x == 0) return 0;
        if (x <= 3) return 1;

        assembly {
            // Initial estimate
            y := x
            let z := add(div(x, 2), 1)

            // Iterate until convergence
            for {} lt(z, y) {} {
                y := z
                z := div(add(div(x, z), z), 2)
            }
        }
    }

    /// @notice Minimum of two values
    function min(uint256 a, uint256 b) internal pure returns (uint256) {
        return a < b ? a : b;
    }

    /// @notice Maximum of two values
    function max(uint256 a, uint256 b) internal pure returns (uint256) {
        return a > b ? a : b;
    }

    // ============================================================
    // Array Operations
    // ============================================================

    /// @notice Sum array elements
    function sum(uint256[] memory arr) internal pure returns (uint256 total) {
        uint256 len = arr.length;
        unchecked {
            for (uint256 i; i < len; ++i) {
                total += arr[i];
            }
        }
    }

    /// @notice Check if array contains value
    function contains(address[] memory arr, address value) internal pure returns (bool) {
        uint256 len = arr.length;
        unchecked {
            for (uint256 i; i < len; ++i) {
                if (arr[i] == value) return true;
            }
        }
        return false;
    }

    /// @notice Find index of value in array (returns type(uint256).max if not found)
    function indexOf(address[] memory arr, address value) internal pure returns (uint256) {
        uint256 len = arr.length;
        unchecked {
            for (uint256 i; i < len; ++i) {
                if (arr[i] == value) return i;
            }
        }
        return type(uint256).max;
    }

    // ============================================================
    // Bit Operations
    // ============================================================

    /// @notice Check if bit is set
    function getBit(uint256 bitmap, uint8 index) internal pure returns (bool) {
        return (bitmap >> index) & 1 == 1;
    }

    /// @notice Set bit
    function setBit(uint256 bitmap, uint8 index) internal pure returns (uint256) {
        return bitmap | (1 << index);
    }

    /// @notice Clear bit
    function clearBit(uint256 bitmap, uint8 index) internal pure returns (uint256) {
        return bitmap & ~(1 << index);
    }

    /// @notice Toggle bit
    function toggleBit(uint256 bitmap, uint8 index) internal pure returns (uint256) {
        return bitmap ^ (1 << index);
    }

    /// @notice Count set bits (population count)
    function popCount(uint256 x) internal pure returns (uint256 count) {
        unchecked {
            while (x != 0) {
                x &= x - 1;
                count++;
            }
        }
    }

    // ============================================================
    // Hashing
    // ============================================================

    /// @notice Efficient hash for two addresses
    function hash2Addresses(address a, address b) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(a, b));
    }

    /// @notice Efficient hash for address and uint
    function hashAddressUint(address a, uint256 n) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(a, n));
    }

    /// @notice Create a unique ID from multiple params
    function createId(
        address creator,
        uint256 nonce,
        uint256 timestamp
    ) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(creator, nonce, timestamp, block.chainid));
    }

    // ============================================================
    // Address Utilities
    // ============================================================

    /// @notice Check if address is contract
    function isContract(address account) internal view returns (bool) {
        uint256 size;
        assembly { size := extcodesize(account) }
        return size > 0;
    }

    /// @notice Safe call with gas limit
    function safeCall(
        address target,
        bytes memory data,
        uint256 gasLimit
    ) internal returns (bool success, bytes memory result) {
        (success, result) = target.call{gas: gasLimit}(data);
    }

    // ============================================================
    // Time Utilities
    // ============================================================

    /// @notice Check if timestamp is in the past
    function isPast(uint256 timestamp) internal view returns (bool) {
        return timestamp < block.timestamp;
    }

    /// @notice Check if timestamp is in the future
    function isFuture(uint256 timestamp) internal view returns (bool) {
        return timestamp > block.timestamp;
    }

    /// @notice Get current period (e.g., for daily/weekly bucketing)
    function getPeriod(uint256 periodLength) internal view returns (uint256) {
        unchecked {
            return block.timestamp / periodLength;
        }
    }

    // ============================================================
    // Storage Utilities
    // ============================================================

    /// @notice Pack two uint128 values into one uint256
    function pack128(uint128 a, uint128 b) internal pure returns (uint256) {
        return uint256(a) | (uint256(b) << 128);
    }

    /// @notice Unpack uint256 into two uint128 values
    function unpack128(uint256 packed) internal pure returns (uint128 a, uint128 b) {
        a = uint128(packed);
        b = uint128(packed >> 128);
    }

    /// @notice Pack four uint64 values into one uint256
    function pack64(uint64 a, uint64 b, uint64 c, uint64 d) internal pure returns (uint256) {
        return uint256(a) |
               (uint256(b) << 64) |
               (uint256(c) << 128) |
               (uint256(d) << 192);
    }

    /// @notice Unpack uint256 into four uint64 values
    function unpack64(uint256 packed) internal pure returns (uint64 a, uint64 b, uint64 c, uint64 d) {
        a = uint64(packed);
        b = uint64(packed >> 64);
        c = uint64(packed >> 128);
        d = uint64(packed >> 192);
    }
}

/**
 * @title Multicall
 * @notice Enables calling multiple methods in a single transaction
 */
abstract contract Multicall {
    /// @notice Execute multiple calls in a single transaction
    function multicall(bytes[] calldata data) external returns (bytes[] memory results) {
        results = new bytes[](data.length);

        unchecked {
            for (uint256 i; i < data.length; ++i) {
                (bool success, bytes memory result) = address(this).delegatecall(data[i]);

                if (!success) {
                    // Bubble up revert reason
                    if (result.length > 0) {
                        assembly {
                            revert(add(32, result), mload(result))
                        }
                    }
                    revert("Multicall: call failed");
                }

                results[i] = result;
            }
        }
    }
}

/**
 * @title SelfPermit
 * @notice Enables permit-style approvals for contracts
 */
abstract contract SelfPermit {
    /// @notice Permit spending of ERC20 tokens
    function selfPermit(
        address token,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external {
        IERC20Permit(token).permit(msg.sender, address(this), value, deadline, v, r, s);
    }

    /// @notice Permit spending with DAI-style permit
    function selfPermitAllowed(
        address token,
        uint256 nonce,
        uint256 expiry,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external {
        IERC20PermitAllowed(token).permit(msg.sender, address(this), nonce, expiry, true, v, r, s);
    }
}

interface IERC20Permit {
    function permit(
        address owner,
        address spender,
        uint256 value,
        uint256 deadline,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external;
}

interface IERC20PermitAllowed {
    function permit(
        address holder,
        address spender,
        uint256 nonce,
        uint256 expiry,
        bool allowed,
        uint8 v,
        bytes32 r,
        bytes32 s
    ) external;
}
