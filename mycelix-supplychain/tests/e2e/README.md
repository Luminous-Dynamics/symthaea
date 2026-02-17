# End-to-End Tests

This directory will contain end-to-end tests for the supply chain provenance system.

## Running Tests

```bash
# Rust integration tests
cd rust/service
cargo test --test integration_tests

# TypeScript E2E tests
cd ts/sdk
npm test
```

## Test Scenarios

1. **Full Event Flow**: PRODUCED → TRANSFORMED → SHIPPED → RECEIVED
2. **Lineage Verification**: Validate parent-child relationships
3. **VC Signature Verification**: Ensure cryptographic integrity
4. **API Error Handling**: Test validation and error responses
5. **Concurrent Operations**: Test race conditions and consistency
