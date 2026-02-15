## [Unreleased]

### Features

* **fhe:** add Fully Homomorphic Encryption module
  - BFV, CKKS, and TFHE scheme support
  - FHEClient for encrypted computation
  - SecureAggregator for threshold decryption with Shamir Secret Sharing
  - Privacy-preserving voting and analytics
  - Presets for development, voting, analytics, and production

* **fl-hub:** add Federated Learning Hub for production deployments
  - FLHubCoordinator for session management
  - ModelRegistry for model versioning and checkpoints
  - PrivacyManager for differential privacy budget tracking
  - PrivateAggregator with calibrated noise injection
  - Support for Rényi Differential Privacy accounting

* **mobile:** add Mobile Wallet SDK
  - WalletManager with PIN and biometric protection
  - BiometricManager with cross-platform abstraction (React Native, Expo)
  - QR code encoding/decoding for wallet connect, credentials, payments
  - CredentialRequestBuilder for selective disclosure
  - Contact exchange and deep link support

* **ai:** add AI Integration module
  - Model inference with multiple provider support
  - Embedding generation and similarity search
  - RAG (Retrieval-Augmented Generation) pipelines
  - Structured output with type-safe responses

* **rtc:** add Real-Time Communication module
  - WebRTC signaling and peer connection management
  - Media stream handling (audio, video, screen share)
  - Data channel support for low-latency messaging
  - TURN/STUN server configuration

## 0.6.0 (2026-01-09)

### Features

* add consciousness field services ([0f6e10d](https://github.com/Luminous-Dynamics/mycelix/commit/0f6e10de24fc9124cb851646466a60dc3a18a92d))
* add sacred port registry system ([a7e4a02](https://github.com/Luminous-Dynamics/mycelix/commit/a7e4a02620d348febe83c93253e9f8e238d96dd7))
* add Sacred Shell to v0.1.0 release ([34e37a1](https://github.com/Luminous-Dynamics/mycelix/commit/34e37a118050b50f4d8497465c84593c21aef50b))
* Align with Sophia-Noesis vision and fix security vulnerabilities ([902ea89](https://github.com/Luminous-Dynamics/mycelix/commit/902ea890e7756042b3728e2c36a16fed38fb28b4))
* Complete K-Codex migration in kosmic-lab ([3ef7d62](https://github.com/Luminous-Dynamics/mycelix/commit/3ef7d62b40e465b0bde85b31a5638a6beacf8a96))
* Complete LuminousOS installer suite ([f106ccc](https://github.com/Luminous-Dynamics/mycelix/commit/f106ccc0620967d693aa9c67dfcfb8dada30ec4c))
* Create mycelial-consciousness integration ([fb12773](https://github.com/Luminous-Dynamics/mycelix/commit/fb12773f209b92fe7a6b4af30b780edfb52eeb7d))
* Enhance LuminousOS with wellness monitoring and unified shell ([2f3c16f](https://github.com/Luminous-Dynamics/mycelix/commit/2f3c16f6c55093e67343503145284628d3745138))
* Implement real consciousness integration bridge ([ee041f1](https://github.com/Luminous-Dynamics/mycelix/commit/ee041f19c701c4841e5a6bd236c5cdc41b516089))
* **kernel:** Add consciousness-aware Stillpoint Kernel extensions ([d365518](https://github.com/Luminous-Dynamics/mycelix/commit/d365518d9604d901bec9ed607aecbfb59de7a463))
* Living deployment with consciousness tools ([9a5a02a](https://github.com/Luminous-Dynamics/mycelix/commit/9a5a02a11942de4676f081b1f797fa8209374d5b))
* **mycelix:** Phase 2 SDK-hApp Integration Complete ([29afb09](https://github.com/Luminous-Dynamics/mycelix/commit/29afb09fb1186058a898070bc49319b78cf8883d))
* **mycelix:** Phase 3 Network Integration Complete ([3814531](https://github.com/Luminous-Dynamics/mycelix/commit/3814531b195e4a65b0ca012c743cdc0cfc359480))
* **mycelix:** Phase 4 Production Hardening Complete ([6068ff9](https://github.com/Luminous-Dynamics/mycelix/commit/6068ff95ced823baf90397a834ca71c0434b59ae))
* **mycelix:** Phase 5 - SDK Ecosystem & Developer Tools ([665e89e](https://github.com/Luminous-Dynamics/mycelix/commit/665e89e5be9eb9a3bde807009a82ed4439d59c66))
* prepare v0.1.0 release - First Light ([371bdc5](https://github.com/Luminous-Dynamics/mycelix/commit/371bdc5062062096b4e3271791b65ef2092023c2))
* Sacred Commerce - consciousness-based payment system ([7bc0d2c](https://github.com/Luminous-Dynamics/mycelix/commit/7bc0d2c570f5b7646aee813a3592f8e1eea5a7c7))
* **sdk-ts:** add security hardening - PQC signatures and capability access control ([4dd826a](https://github.com/Luminous-Dynamics/mycelix/commit/4dd826ab864beebaeef91c048478ef69e2595e0d))

### Bug Fixes

* **sdk-ts:** resolve epistemic circular import and add codegen CLI ([4de699a](https://github.com/Luminous-Dynamics/mycelix/commit/4de699a829c105b8c135cd4fc8915b9cfc8a469b))
* **sdk-ts:** resolve test failures in propagation and discovery modules ([c75f68f](https://github.com/Luminous-Dynamics/mycelix/commit/c75f68f4d9ed6fa399ea4922a0ec222dca917a17))
* **sdk-ts:** security hardening and validation fixes ([5e2b96b](https://github.com/Luminous-Dynamics/mycelix/commit/5e2b96b0d6571d9339db5d7bf48cb861cf7e5d48))
