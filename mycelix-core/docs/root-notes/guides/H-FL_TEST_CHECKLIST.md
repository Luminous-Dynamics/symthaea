# H-FL Testing Checklist

## Phase 1: Local Python Testing ⏳
- [ ] Install PyTorch via Nix (in progress...)
- [ ] Run `test_hfl_local.py` to validate federated averaging
- [ ] Verify Byzantine fault tolerance works
- [ ] Confirm 3 agents can train with different data distributions

## Phase 2: Holochain Integration Testing
- [ ] Build Holochain DNA with `BUILD_H-FL_MVP.sh`
- [ ] Start Holochain conductor
- [ ] Run single FL client
- [ ] Test gradient submission to DHT
- [ ] Test gradient retrieval from DHT

## Phase 3: Multi-Agent Testing
- [ ] Launch 3 FL agents simultaneously
- [ ] Verify agents can see each other's gradients
- [ ] Confirm federated averaging across agents
- [ ] Test model convergence over 5 rounds

## Phase 4: Resilience Testing
- [ ] Test with one malicious agent
- [ ] Test network partition recovery
- [ ] Test agent dropout/reconnection
- [ ] Verify DHT consistency

## Expected Results

### Local Test (test_hfl_local.py)
- Global accuracy should improve each round
- Should reach ~80%+ accuracy after 5 rounds
- Byzantine attack should be successfully filtered

### Full H-FL Demo
- All agents should converge to similar accuracy
- DHT should store gradients immutably
- No central server should be required
- System should be resilient to failures

## Commands Reference

```bash
# Check PyTorch installation
nix develop --command python3 -c "import torch; print(torch.__version__)"

# Run local test
nix develop --command python3 test_hfl_local.py

# Build Holochain DNA
./BUILD_H-FL_MVP.sh

# Run full demo
./run_hfl_demo.sh

# Manual agent launch
AGENT_ID=1 nix develop --command python3 fl_client.py
```

## Troubleshooting

### PyTorch not found
- Make sure you're in `nix develop` environment
- Check that flake.nix includes torch in python packages

### Holochain connection fails
- Check conductor is running on port 8888
- Verify WebSocket connection works
- Check firewall settings

### Gradients not aggregating
- Verify all agents are in same round
- Check gradient dimensions match
- Ensure DHT is syncing properly