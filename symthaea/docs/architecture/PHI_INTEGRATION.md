# Φ (Phi) Integration in Symthaea

## What Phi Actually Does

Phi (Φ) is computed as a unified consciousness metric combining temporal coherence,
voice feedback quality, and flow state. However, its actual behavioral influence
is **intentionally minimal** - it serves primarily as an observability metric.

### Decision Points Where Phi Affects Behavior

| Location | Effect | Threshold |
|----------|--------|-----------|
| `select_strategy()` | High Φ → Exploratory mode, Low Φ → Supportive mode | ≥0.6 / <0.3 |
| `is_conscious()` | Binary consciousness check | >0.3 |
| `episodic_memory.encode()` | Phi passed as context when storing memories | N/A |

### What Phi Does NOT Affect

Despite some documentation suggesting otherwise, Phi does **not** currently:
- Modulate learning rate (that's handled by semantic memory lr_factor, flow state, curiosity)
- Drive adaptive HDC dimensionality (that's based on prediction error directly)
- Affect attention priority in the cognitive loop (attention is bid-based)

### Why This Is Intentional

The current minimal Phi integration is a deliberate design choice:

1. **Avoid consciousness theater** - Adding Phi to every decision would be complexity
   without clear benefit. The Q-learning + semantic memory approach is more directly useful.

2. **Observability value** - Phi provides a single number to monitor system health,
   which is valuable even without driving every decision.

3. **Strategy gating is sufficient** - The high-Φ → exploratory, low-Φ → supportive
   mapping captures the intuition that confident systems explore while uncertain
   systems play it safe.

### Phi Computation

Phi is computed in `CognitiveLoopService::step()` as:

```rust
let unified_phi = (coherence_phi + voice_phi + flow_phi).clamp(0.0, 1.0);
```

Where:
- `coherence_phi`: CfC temporal coherence from the coherence bridge
- `voice_phi`: Voice synthesis quality feedback
- `flow_phi`: Flow state intensity

### Future Considerations

If deeper Phi integration is desired, candidate areas include:
- Attention bid weighting (PhiAwareScoring exists but isn't wired to main loop)
- Memory consolidation priority
- Exploration/exploitation balance in primitive discovery

However, each addition should demonstrate measurable improvement over simpler
alternatives before being adopted.

## Summary

Phi is primarily an **observability metric** with light behavioral influence
through strategy selection. This is honest and intentional - more integration
would add complexity without proven benefit.
