# RCAF Event-Triggered Spatial Transient Nuisance v1

**Status:** executable synthetic reference family; not an empirical motion model.  
**Scope:** broad and localized transient observation artifacts, with explicit event times and graph-distance fields.

## 1. Why a separate family is needed

The common-mode challenge isolates spatially uniform observation contamination. S1/S2 adds smooth low-rank and localized static spatial fields. Neither family represents a burst that begins at a known time and decays over subsequent samples.

The next challenge models a generic transient:

[
h_j(t)=a_jexp(-(t-t_j)/\tau_j)\mathbf{1}(t\ge t_j).
]

This is useful for testing how event timing relative to segmentation, filtering and censoring changes a connectivity estimate. The executable model is deliberately stylized: it is not rigid-body motion, an MRI acquisition simulator, or a validated model of motion-induced BOLD.

## 2. Spatial-temporal construction

The current Rust module combines two components:

**Broad transient**
- a spatially constant loading across all ROIs;
- a sum of explicit broad event trains with individual onset, signed amplitude and exponential decay.

**Localized transient**
- one event train per declared seed ROI;
- graph-distance loading from each seed;
- edge traversal cost defined as inverse adjacency weight;
- disconnected ROIs receive zero localized loading.

For ROI r and seed s:

[
q_r(s)=\exp(-d_G(s,r)/\ell),
qquad
A_{t,r}=\lambda\{b h^{broad}_t+
\ell_{load}\sum_j q_r(s_j)h^{local}_j(t)\}.
]

Here \(\ell\) is the declared graph-distance decay length, \(b\) the broad loading and \(\ell_{load}\) the localized loading. The graph geometry and inverse-weight edge convention are part of the synthetic model contract; ROI array order is not geometry.

Event lists are explicit and deterministic. No RNG is used in this v1 generator. It returns the same field for the same graph and event/configuration inputs. One generated field can therefore be reused unchanged in nuisance-only and signal-plus-nuisance counterfactuals.

## 3. Required adversarial grid

The next simulator should cross:

- event onset relative to run/segment boundaries and censor windows;
- event amplitude, sign, duration and decay time;
- broad versus localized loading;
- seed location, graph-distance scale, graph connectivity and ROI coverage;
- event density, overlap and repeated bursts;
- signal effect size and temporal coherence with the transient;
- preprocessing order (regression/filter/censoring) and retained temporal degrees of freedom.

At minimum include:
- isolated broad events;
- isolated localized events;
- overlapping broad and localized events;
- positive and negative signed transient controls;
- no-event exact null;
- event just before, inside and just after the retained segment;
- disconnected-node controls and ROI permutation equivariance.

The four counterfactual cells remain mandatory: clean null, signal only, nuisance only, and signal plus nuisance. Nuisance-only events preserve the declared latent target. The combined cell must reuse the exact event list and generated field from the nuisance-only cell.

## 4. Outputs and qualification

The first smoke report records graph labels/edges, edge-length convention, event times/amplitudes/decay constants, seed ROIs, spatial loading fields, and realized field RMS/max magnitude.

The full inference report should additionally quantify:
- artifact-only false alarms;
- signal-plus-transient directional recovery and wrong-direction rejection;
- effect attenuation/inflation and interval coverage;
- retained DoF and participant/data attrition after event censoring;
- event timing sensitivity and estimator-by-decay response surfaces;
- correct abstention on observationally equivalent target/artifact cases.

No nuisance dose should be compared across graph geometries using only a raw additive coefficient. Reuse the variance-fraction calibration layer and report clean–nuisance covariance separately.

## 5. Implementation boundaries and next steps

The present module is a deterministic graph-distance exponential field generator. The inverse-adjacency edge-length convention is synthetic and must not be mistaken for physical distance. Future work should use explicit atlas coordinates or validated surface/volume geodesics, then drive event timing and amplitudes from measured motion traces and acquisition metadata. It should model the temporal point-spread and slice/volume acquisition geometry, and test real denoising pipelines under frozen splits.

A genuine motion qualification must compare against independent motion/physiology measurements and not infer that a generated transient is physically faithful merely because its graph and time series are reproducible.

## Claim limits

This is a synthetic nuisance family and smoke fixture. It does not reproduce scanner mechanics, head pose, slice timing, spin-history effects, physiological BOLD contamination, or motion correction. It does not qualify fMRIPrep, XCP-D, CONN or an empirical biological endpoint. No SOTA or clinical claim is made.
