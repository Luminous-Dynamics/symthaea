// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test, console} from "forge-std/Test.sol";
import {MycelixRegistry} from "../src/MycelixRegistry.sol";

contract MycelixRegistryTest is Test {
    MycelixRegistry public registry;

    address public owner = address(1);
    address payable public feeRecipient = payable(address(2));
    address public user1 = address(3);
    address public user2 = address(4);
    address public user3 = address(5);

    bytes32 public constant GOVERNANCE_ROLE = keccak256("GOVERNANCE_ROLE");
    bytes32 public constant HAPP_REGISTRAR_ROLE = keccak256("HAPP_REGISTRAR_ROLE");

    function setUp() public {
        vm.prank(owner);
        registry = new MycelixRegistry(owner, feeRecipient);

        // Fund users
        vm.deal(user1, 10 ether);
        vm.deal(user2, 10 ether);
    }

    // ============ Constructor Tests ============

    function test_Constructor() public view {
        assertTrue(registry.hasRole(registry.DEFAULT_ADMIN_ROLE(), owner));
        assertTrue(registry.hasRole(GOVERNANCE_ROLE, owner));
        assertTrue(registry.hasRole(HAPP_REGISTRAR_ROLE, owner));
        assertEq(registry.feeRecipient(), feeRecipient);
        assertEq(registry.registrationFee(), 0);
    }

    function test_Constructor_RevertZeroOwner() public {
        vm.expectRevert(abi.encodeWithSignature("OwnableInvalidOwner(address)", address(0)));
        new MycelixRegistry(address(0), feeRecipient);
    }

    // ============ DID Registration Tests ============

    function test_RegisterDID() public {
        bytes32 did = keccak256("did:mycelix:test1");
        bytes32 metadataHash = keccak256("metadata");

        vm.prank(user1);
        registry.registerDID(did, metadataHash);

        (address resolvedOwner, address[] memory alternates, bool isActive) =
            registry.resolveAddress(did);

        assertEq(resolvedOwner, user1);
        assertEq(alternates.length, 0);
        assertTrue(isActive);
        assertEq(registry.totalDids(), 1);
    }

    function test_RegisterDID_WithFee() public {
        uint256 fee = 0.01 ether;
        vm.prank(owner);
        registry.setRegistrationFee(fee);

        bytes32 did = keccak256("did:mycelix:test1");

        vm.prank(user1);
        registry.registerDID{value: fee}(did, keccak256("metadata"));

        (address resolvedOwner, , ) = registry.resolveAddress(did);
        assertEq(resolvedOwner, user1);
    }

    function test_RegisterDID_RevertInsufficientFee() public {
        vm.prank(owner);
        registry.setRegistrationFee(0.01 ether);

        bytes32 did = keccak256("did:mycelix:test1");

        vm.prank(user1);
        vm.expectRevert(MycelixRegistry.InsufficientFee.selector);
        registry.registerDID{value: 0.005 ether}(did, keccak256("metadata"));
    }

    function test_RegisterDID_RevertAlreadyRegistered() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.prank(user1);
        registry.registerDID(did, keccak256("metadata"));

        vm.prank(user2);
        vm.expectRevert(MycelixRegistry.DIDAlreadyRegistered.selector);
        registry.registerDID(did, keccak256("metadata2"));
    }

    function test_RegisterDID_RevertAddressAlreadyLinked() public {
        bytes32 did1 = keccak256("did:mycelix:test1");
        bytes32 did2 = keccak256("did:mycelix:test2");

        vm.prank(user1);
        registry.registerDID(did1, keccak256("metadata"));

        vm.prank(user1);
        vm.expectRevert(MycelixRegistry.AddressAlreadyLinked.selector);
        registry.registerDID(did2, keccak256("metadata2"));
    }

    // ============ DID Update Tests ============

    function test_UpdateOwner() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.prank(user1);
        registry.registerDID(did, keccak256("metadata"));

        vm.prank(user1);
        registry.updateOwner(did, user2);

        (address resolvedOwner, , ) = registry.resolveAddress(did);
        assertEq(resolvedOwner, user2);

        // Reverse lookup updated
        assertEq(registry.getDIDForAddress(user1), bytes32(0));
        assertEq(registry.getDIDForAddress(user2), did);
    }

    function test_UpdateOwner_RevertNotOwner() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.prank(user1);
        registry.registerDID(did, keccak256("metadata"));

        vm.prank(user2);
        vm.expectRevert(MycelixRegistry.NotDIDOwner.selector);
        registry.updateOwner(did, user3);
    }

    function test_UpdateOwner_RevertSameAddress() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.prank(user1);
        registry.registerDID(did, keccak256("metadata"));

        vm.prank(user1);
        vm.expectRevert(MycelixRegistry.SameAddress.selector);
        registry.updateOwner(did, user1);
    }

    // ============ Alternate Address Tests ============

    function test_AddAlternate() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.prank(user1);
        registry.registerDID(did, keccak256("metadata"));

        vm.prank(user1);
        registry.addAlternate(did, user2);

        (, address[] memory alternates, ) = registry.resolveAddress(did);
        assertEq(alternates.length, 1);
        assertEq(alternates[0], user2);

        // Reverse lookup for alternate
        assertEq(registry.getDIDForAddress(user2), did);
    }

    function test_AddAlternate_Multiple() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.startPrank(user1);
        registry.registerDID(did, keccak256("metadata"));
        registry.addAlternate(did, user2);
        registry.addAlternate(did, user3);
        vm.stopPrank();

        (, address[] memory alternates, ) = registry.resolveAddress(did);
        assertEq(alternates.length, 2);
    }

    function test_AddAlternate_RevertTooMany() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.startPrank(user1);
        registry.registerDID(did, keccak256("metadata"));

        // Add max alternates
        for (uint256 i = 0; i < 5; i++) {
            registry.addAlternate(did, address(uint160(100 + i)));
        }

        // Try to add one more
        vm.expectRevert(MycelixRegistry.TooManyAlternates.selector);
        registry.addAlternate(did, address(200));
        vm.stopPrank();
    }

    function test_RemoveAlternate() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.startPrank(user1);
        registry.registerDID(did, keccak256("metadata"));
        registry.addAlternate(did, user2);
        registry.removeAlternate(did, user2);
        vm.stopPrank();

        (, address[] memory alternates, ) = registry.resolveAddress(did);
        assertEq(alternates.length, 0);

        // Reverse lookup cleared
        assertEq(registry.getDIDForAddress(user2), bytes32(0));
    }

    function test_RemoveAlternate_RevertNotFound() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.prank(user1);
        registry.registerDID(did, keccak256("metadata"));

        vm.prank(user1);
        vm.expectRevert(MycelixRegistry.AlternateNotFound.selector);
        registry.removeAlternate(did, user2);
    }

    // ============ Revocation Tests ============

    function test_RevokeDID() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.startPrank(user1);
        registry.registerDID(did, keccak256("metadata"));
        registry.addAlternate(did, user2);
        registry.revokeDID(did);
        vm.stopPrank();

        (, , bool isActive) = registry.resolveAddress(did);
        assertFalse(isActive);

        // Reverse lookups cleared
        assertEq(registry.getDIDForAddress(user1), bytes32(0));
        assertEq(registry.getDIDForAddress(user2), bytes32(0));
    }

    function test_RevokeDID_RevertUpdateAfterRevoke() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.startPrank(user1);
        registry.registerDID(did, keccak256("metadata"));
        registry.revokeDID(did);

        vm.expectRevert(MycelixRegistry.DIDIsRevoked.selector);
        registry.updateMetadata(did, keccak256("new metadata"));
        vm.stopPrank();
    }

    // ============ hApp Registry Tests ============

    function test_RegisterHApp() public {
        bytes32 happId = keccak256("happ:mycelix:test");
        address contractAddr = address(0x1234);

        vm.prank(owner);
        registry.registerHApp(happId, contractAddr, keccak256("metadata"));

        (address resolved, bool isActive) = registry.resolveHApp(happId);
        assertEq(resolved, contractAddr);
        assertTrue(isActive);
        assertEq(registry.totalHapps(), 1);
    }

    function test_RegisterHApp_RevertUnauthorized() public {
        bytes32 happId = keccak256("happ:mycelix:test");

        vm.prank(user1);
        vm.expectRevert();
        registry.registerHApp(happId, address(0x1234), keccak256("metadata"));
    }

    function test_UpdateHApp() public {
        bytes32 happId = keccak256("happ:mycelix:test");
        address oldContract = address(0x1234);
        address newContract = address(0x5678);

        vm.startPrank(owner);
        registry.registerHApp(happId, oldContract, keccak256("metadata"));
        registry.updateHApp(happId, newContract);
        vm.stopPrank();

        (address resolved, ) = registry.resolveHApp(happId);
        assertEq(resolved, newContract);
    }

    function test_DeactivateHApp() public {
        bytes32 happId = keccak256("happ:mycelix:test");

        vm.startPrank(owner);
        registry.registerHApp(happId, address(0x1234), keccak256("metadata"));
        registry.deactivateHApp(happId);
        vm.stopPrank();

        (, bool isActive) = registry.resolveHApp(happId);
        assertFalse(isActive);
    }

    // ============ Admin Tests ============

    function test_SetRegistrationFee() public {
        vm.prank(owner);
        registry.setRegistrationFee(0.01 ether);

        assertEq(registry.registrationFee(), 0.01 ether);
    }

    function test_SetFeeRecipient() public {
        address payable newRecipient = payable(address(0x9999));

        vm.prank(owner);
        registry.setFeeRecipient(newRecipient);

        assertEq(registry.feeRecipient(), newRecipient);
    }

    function test_WithdrawFees() public {
        // Set a fee and register a DID to generate fees
        vm.prank(owner);
        registry.setRegistrationFee(0.01 ether);

        vm.prank(user1);
        registry.registerDID{value: 0.01 ether}(keccak256("did"), keccak256("meta"));

        uint256 recipientBefore = feeRecipient.balance;

        vm.prank(owner);
        registry.withdrawFees();

        assertEq(feeRecipient.balance - recipientBefore, 0.01 ether);
    }

    function test_Pause() public {
        vm.prank(owner);
        registry.pause();

        assertTrue(registry.paused());

        vm.prank(user1);
        vm.expectRevert();
        registry.registerDID(keccak256("did"), keccak256("meta"));
    }

    // ============ View Tests ============

    function test_GetDIDRecord() public {
        bytes32 did = keccak256("did:mycelix:test1");
        bytes32 metadataHash = keccak256("metadata");

        vm.prank(user1);
        registry.registerDID(did, metadataHash);

        MycelixRegistry.DIDRecord memory record = registry.getDIDRecord(did);
        assertEq(record.owner, user1);
        assertEq(record.metadataHash, metadataHash);
        assertFalse(record.revoked);
        assertTrue(record.registeredAt > 0);
    }

    function test_GetHAppRecord() public {
        bytes32 happId = keccak256("happ:mycelix:test");
        address contractAddr = address(0x1234);
        bytes32 metadataHash = keccak256("metadata");

        vm.prank(owner);
        registry.registerHApp(happId, contractAddr, metadataHash);

        MycelixRegistry.HAppRecord memory record = registry.getHAppRecord(happId);
        assertEq(record.happId, happId);
        assertEq(record.contractAddress, contractAddr);
        assertEq(record.registrar, owner);
        assertEq(record.metadataHash, metadataHash);
        assertTrue(record.active);
    }

    function test_IsAddressLinked() public {
        bytes32 did = keccak256("did:mycelix:test1");

        assertFalse(registry.isAddressLinked(user1));

        vm.prank(user1);
        registry.registerDID(did, keccak256("metadata"));

        assertTrue(registry.isAddressLinked(user1));
    }

    function test_GetDIDForAddress() public {
        bytes32 did = keccak256("did:mycelix:test1");

        vm.prank(user1);
        registry.registerDID(did, keccak256("metadata"));

        assertEq(registry.getDIDForAddress(user1), did);
        assertEq(registry.getDIDForAddress(user2), bytes32(0));
    }
}
