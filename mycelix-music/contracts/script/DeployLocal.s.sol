// SPDX-License-Identifier: MIT
pragma solidity ^0.8.23;

import "forge-std/Script.sol";
import "../src/EconomicStrategyRouter.sol";
import "../src/strategies/PayPerStreamStrategy.sol";
import "../src/strategies/GiftEconomyStrategy.sol";

/**
 * @title MockFLOWToken
 * @notice Mock FLOW token for local testing
 */
contract MockFLOWToken {
    string public name = "FLOW Token";
    string public symbol = "FLOW";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor() {
        // Mint 1 million FLOW to deployer
        totalSupply = 1_000_000 ether;
        balanceOf[msg.sender] = totalSupply;
        emit Transfer(address(0), msg.sender, totalSupply);
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        balanceOf[msg.sender] -= amount;
        balanceOf[to] += amount;
        emit Transfer(msg.sender, to, amount);
        return true;
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        allowance[from][msg.sender] -= amount;
        balanceOf[from] -= amount;
        balanceOf[to] += amount;
        emit Transfer(from, to, amount);
        return true;
    }

    function mint(address to, uint256 amount) external {
        totalSupply += amount;
        balanceOf[to] += amount;
        emit Transfer(address(0), to, amount);
    }
}

/**
 * @title MockCGCRegistry
 * @notice Mock CGC registry for local testing
 */
contract MockCGCRegistry {
    mapping(address => uint256) public cgcBalance;

    event CGCAwarded(address indexed recipient, uint256 amount, string reason);

    function awardCGC(address recipient, uint256 amount, string memory reason) external {
        cgcBalance[recipient] += amount;
        emit CGCAwarded(recipient, amount, reason);
    }

    function getBalance(address account) external view returns (uint256) {
        return cgcBalance[account];
    }
}

/**
 * @title DeployLocal
 * @notice Deployment script for local Anvil network
 */
contract DeployLocal is Script {
    function run() external {
        // Get deployer from environment
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);

        console.log("Deploying from:", deployer);
        console.log("Deployer balance:", deployer.balance);

        vm.startBroadcast(deployerPrivateKey);

        // 1. Deploy Mock FLOW Token
        console.log("\n=== Deploying Mock FLOW Token ===");
        MockFLOWToken flowToken = new MockFLOWToken();
        console.log("FLOW Token deployed at:", address(flowToken));
        console.log("Total supply:", flowToken.totalSupply() / 1 ether, "FLOW");

        // 2. Deploy Mock CGC Registry
        console.log("\n=== Deploying Mock CGC Registry ===");
        MockCGCRegistry cgcRegistry = new MockCGCRegistry();
        console.log("CGC Registry deployed at:", address(cgcRegistry));

        // 3. Deploy Economic Strategy Router
        console.log("\n=== Deploying Economic Strategy Router ===");
        address protocolTreasury = deployer; // Use deployer as treasury for testing
        EconomicStrategyRouter router = new EconomicStrategyRouter(
            address(flowToken),
            protocolTreasury
        );
        console.log("Router deployed at:", address(router));

        // 4. Deploy Pay Per Stream Strategy
        console.log("\n=== Deploying Pay Per Stream Strategy ===");
        PayPerStreamStrategy payPerStream = new PayPerStreamStrategy(
            address(flowToken),
            address(router)
        );
        console.log("PayPerStream Strategy deployed at:", address(payPerStream));

        // 5. Deploy Gift Economy Strategy
        console.log("\n=== Deploying Gift Economy Strategy ===");
        GiftEconomyStrategy giftEconomy = new GiftEconomyStrategy(
            address(flowToken),
            address(router),
            address(cgcRegistry)
        );
        console.log("GiftEconomy Strategy deployed at:", address(giftEconomy));

        // 6. Register strategies with router
        console.log("\n=== Registering Strategies ===");
        bytes32 payPerStreamId = keccak256("pay-per-stream-v1");
        bytes32 giftEconomyId = keccak256("gift-economy-v1");

        router.registerStrategy(payPerStreamId, address(payPerStream));
        console.log("Registered PayPerStream strategy with ID:", vm.toString(payPerStreamId));

        router.registerStrategy(giftEconomyId, address(giftEconomy));
        console.log("Registered GiftEconomy strategy with ID:", vm.toString(giftEconomyId));

        // 7. Distribute FLOW tokens to test accounts
        console.log("\n=== Distributing FLOW Tokens ===");

        // Anvil test accounts (first 10 accounts)
        address[] memory testAccounts = new address[](10);
        testAccounts[0] = 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266; // Account #0
        testAccounts[1] = 0x70997970C51812dc3A010C7d01b50e0d17dc79C8; // Account #1
        testAccounts[2] = 0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC; // Account #2
        testAccounts[3] = 0x90F79bf6EB2c4f870365E785982E1f101E93b906; // Account #3
        testAccounts[4] = 0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65; // Account #4
        testAccounts[5] = 0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc; // Account #5
        testAccounts[6] = 0x976EA74026E726554dB657fA54763abd0C3a0aa9; // Account #6
        testAccounts[7] = 0x14dC79964da2C08b23698B3D3cc7Ca32193d9955; // Account #7
        testAccounts[8] = 0x23618e81E3f5cdF7f54C3d65f7FBc0aBf5B21E8f; // Account #8
        testAccounts[9] = 0xa0Ee7A142d267C1f36714E4a8F75612F20a79720; // Account #9

        // Give each account 1000 FLOW
        for (uint256 i = 0; i < testAccounts.length; i++) {
            flowToken.transfer(testAccounts[i], 1000 ether);
            console.log("Sent 1000 FLOW to:", testAccounts[i]);
        }

        vm.stopBroadcast();

        // Print deployment summary
        console.log("\n=== DEPLOYMENT COMPLETE ===");
        console.log("\nAdd these to your .env file:");
        console.log("NEXT_PUBLIC_FLOW_TOKEN_ADDRESS=%s", address(flowToken));
        console.log("NEXT_PUBLIC_ROUTER_ADDRESS=%s", address(router));
        console.log("NEXT_PUBLIC_PAY_PER_STREAM_ADDRESS=%s", address(payPerStream));
        console.log("NEXT_PUBLIC_GIFT_ECONOMY_ADDRESS=%s", address(giftEconomy));
        console.log("CGC_REGISTRY_ADDRESS=%s", address(cgcRegistry));

        console.log("\nStrategy IDs:");
        console.log("PAY_PER_STREAM_ID=%s", vm.toString(payPerStreamId));
        console.log("GIFT_ECONOMY_ID=%s", vm.toString(giftEconomyId));

        console.log("\nTest Accounts (each has 1000 FLOW):");
        for (uint256 i = 0; i < testAccounts.length; i++) {
            console.log("Account #%s: %s", i, testAccounts[i]);
        }
    }
}
