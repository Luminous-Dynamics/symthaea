# symthaea-gpu-fabric

Backend-neutral semantic GPU execution contracts for Symthaea.

This crate intentionally starts with a deterministic CPU reference executor. It defines the semantic operation plan, resource limits, backend identity, kernel identity, and execution receipt that native Vulkan and browser WebGPU implementations will later share.

## First operation

The first contract is binary HDC binding:

`HdcBindXor` = bitwise XOR over canonical bit-packed hypervectors.

The CPU reference executor is the qualification oracle. A future accelerated backend must produce the same semantic result for the same plan and inputs, while proving that it actually executed on the claimed backend.

## Evidence boundary

Receipts bind the operation, plan digest, kernel identity/digest, input digest, output digest, backend, determinism mode, and acceleration claim.

The CPU reference backend is never reported as accelerated. Receipt verification rejects an inconsistent acceleration claim.

Resource validation occurs before execution and bounds input bytes, output bytes, dispatch invocations, vector dimensions, and canonical tail bits.

## Intended backend layering

```
semantic operation plan
        |
        +--> CPU reference
        +--> WebGPU / wgpu
        +--> native Vulkan
```

The semantic contract does not depend on any graphics API and does not introduce a parallel provenance ontology.
