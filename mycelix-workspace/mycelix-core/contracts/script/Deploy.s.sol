// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Script, console} from "forge-std/Script.sol";
import {ReputationAnchor} from "../src/ReputationAnchor.sol";
import {PaymentRouter} from "../src/PaymentRouter.sol";
import {MycelixRegistry} from "../src/MycelixRegistry.sol";

/**
 * @title DeployMycelix
 * @notice Deployment script for Mycelix smart contracts
 * @dev Run with:
 *   forge script script/Deploy.s.sol:DeployMycelix --rpc-url $RPC_URL --broadcast --verify
 *
 * Required environment variables:
 *   - PRIVATE_KEY: Deployer private key
 *   - OWNER_ADDRESS: Contract owner address
 *   - ANCHOR_ADDRESS: Initial anchor node address
 *   - FEE_RECIPIENT: Address to receive platform fees
 */
contract DeployMycelix is Script {
    function run() external {
        // Load configuration from environment
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address owner = vm.envAddress("OWNER_ADDRESS");
        address anchor = vm.envAddress("ANCHOR_ADDRESS");
        address payable feeRecipient = payable(vm.envAddress("FEE_RECIPIENT"));

        console.log("Deploying Mycelix contracts...");
        console.log("Owner:", owner);
        console.log("Anchor:", anchor);
        console.log("Fee Recipient:", feeRecipient);

        vm.startBroadcast(deployerPrivateKey);

        // Deploy ReputationAnchor
        ReputationAnchor reputationAnchor = new ReputationAnchor(owner, anchor);
        console.log("ReputationAnchor deployed at:", address(reputationAnchor));

        // Deploy PaymentRouter
        PaymentRouter paymentRouter = new PaymentRouter(owner, feeRecipient);
        console.log("PaymentRouter deployed at:", address(paymentRouter));

        // Deploy MycelixRegistry
        MycelixRegistry registry = new MycelixRegistry(owner, feeRecipient);
        console.log("MycelixRegistry deployed at:", address(registry));

        vm.stopBroadcast();

        // Log summary
        console.log("\n=== Deployment Summary ===");
        console.log("ReputationAnchor:", address(reputationAnchor));
        console.log("PaymentRouter:", address(paymentRouter));
        console.log("MycelixRegistry:", address(registry));
        console.log("========================\n");
    }
}

/**
 * @title DeployMumbai
 * @notice Deployment script specifically for Polygon Mumbai testnet
 */
contract DeployMumbai is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);

        console.log("Deploying to Polygon Mumbai...");
        console.log("Deployer:", deployer);

        vm.startBroadcast(deployerPrivateKey);

        // For testnet, deployer is owner, anchor, and fee recipient
        ReputationAnchor reputationAnchor = new ReputationAnchor(deployer, deployer);
        console.log("ReputationAnchor:", address(reputationAnchor));

        PaymentRouter paymentRouter = new PaymentRouter(deployer, payable(deployer));
        console.log("PaymentRouter:", address(paymentRouter));

        MycelixRegistry registry = new MycelixRegistry(deployer, payable(deployer));
        console.log("MycelixRegistry:", address(registry));

        vm.stopBroadcast();

        // Output JSON for SDK configuration
        string memory json = string(
            abi.encodePacked(
                '{\n',
                '  "network": "polygon-mumbai",\n',
                '  "chainId": 80001,\n',
                '  "contracts": {\n',
                '    "reputationAnchor": "', vm.toString(address(reputationAnchor)), '",\n',
                '    "paymentRouter": "', vm.toString(address(paymentRouter)), '",\n',
                '    "mycelixRegistry": "', vm.toString(address(registry)), '"\n',
                '  }\n',
                '}'
            )
        );
        console.log("\nSDK Configuration:");
        console.log(json);
    }
}

/**
 * @title DeployAmoy
 * @notice Deployment script for Polygon Amoy testnet (newer testnet)
 */
contract DeployAmoy is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);

        console.log("Deploying to Polygon Amoy...");
        console.log("Deployer:", deployer);

        vm.startBroadcast(deployerPrivateKey);

        ReputationAnchor reputationAnchor = new ReputationAnchor(deployer, deployer);
        console.log("ReputationAnchor:", address(reputationAnchor));

        PaymentRouter paymentRouter = new PaymentRouter(deployer, payable(deployer));
        console.log("PaymentRouter:", address(paymentRouter));

        MycelixRegistry registry = new MycelixRegistry(deployer, payable(deployer));
        console.log("MycelixRegistry:", address(registry));

        vm.stopBroadcast();

        string memory json = string(
            abi.encodePacked(
                '{\n',
                '  "network": "polygon-amoy",\n',
                '  "chainId": 80002,\n',
                '  "contracts": {\n',
                '    "reputationAnchor": "', vm.toString(address(reputationAnchor)), '",\n',
                '    "paymentRouter": "', vm.toString(address(paymentRouter)), '",\n',
                '    "mycelixRegistry": "', vm.toString(address(registry)), '"\n',
                '  }\n',
                '}'
            )
        );
        console.log("\nSDK Configuration:");
        console.log(json);
    }
}

/**
 * @title DeploySepolia
 * @notice Deployment script for Ethereum Sepolia testnet
 * @dev No mainnet ETH required - use https://sepolia-faucet.pk910.de/
 */
contract DeploySepolia is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);

        console.log("Deploying to Ethereum Sepolia...");
        console.log("Deployer:", deployer);

        vm.startBroadcast(deployerPrivateKey);

        ReputationAnchor reputationAnchor = new ReputationAnchor(deployer, deployer);
        console.log("ReputationAnchor:", address(reputationAnchor));

        PaymentRouter paymentRouter = new PaymentRouter(deployer, payable(deployer));
        console.log("PaymentRouter:", address(paymentRouter));

        MycelixRegistry registry = new MycelixRegistry(deployer, payable(deployer));
        console.log("MycelixRegistry:", address(registry));

        vm.stopBroadcast();

        string memory json = string(
            abi.encodePacked(
                '{\n',
                '  "network": "sepolia",\n',
                '  "chainId": 11155111,\n',
                '  "contracts": {\n',
                '    "reputationAnchor": "', vm.toString(address(reputationAnchor)), '",\n',
                '    "paymentRouter": "', vm.toString(address(paymentRouter)), '",\n',
                '    "mycelixRegistry": "', vm.toString(address(registry)), '"\n',
                '  }\n',
                '}'
            )
        );
        console.log("\nSDK Configuration:");
        console.log(json);
    }
}
