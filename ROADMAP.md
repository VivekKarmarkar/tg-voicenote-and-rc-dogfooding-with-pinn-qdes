# Project Roadmap: PINN-QDE (Physics-Informed Neural Networks for Quaternionic Differential Equations)

> **Created:** 2026-05-03
> **Status:** Draft
> **Owner:** VivekKarmarkar

## Vision
Publish a paper (journal + arXiv) demonstrating that PINNs can solve quaternionic kinematic ODEs for rotational motion estimation using only angular velocity data and boundary quaternions — no interior supervision required. The project doubles as a dogfooding experiment: all research is directed via Telegram voice notes.

## Current State
- Segment 23 solved: PINN achieves 0.0062 rad/s max error (pass threshold: ±0.05)
- Ablation study complete: max-error penalty term reduces error by 35% and stabilizes training
- Architecture validated: [129, 512, 512, 256, 4] MLP with linear Fourier features (64 freqs, scale 4.0)
- 3-phase training strategy: BC-anchored → physics-focused → max-error refinement
- Theoretical insight identified: quaternionic loss geometry vs real-valued latent space mismatch
- Only 1 segment tested so far — robustness unknown
- Paper writing infrastructure exists (verbatim/claude extraction pipeline, section markdown files)

---

## Phase 1: Robustness Validation — Multi-Segment Testing
**Goal:** Prove the method generalizes across diverse motion profiles with minimal hyperparameter tweaking
**Status:** Not Started

| Priority | Task | Description | Dependencies |
|----------|------|-------------|-------------|
| P0 | Prepare multi-segment data | Vivek provides 10+ segments with gyro + BC quaternions only | Vivek strips reference quaternions |
| P0 | Run baseline (no tweaking) | Apply current architecture to all new segments, report pass/fail | Segment data available |
| P1 | Build Fourier feature diagnostic | Power spectrum analysis of gyro signal → principled NUM_FF/FF_SCALE selection | — |
| P1 | Track tweaking log | For each segment that fails, document what changes were needed | Baseline results |
| P2 | Automated batch runner | Script to run all segments sequentially and produce summary table | Segment data available |

**Exit criteria:**
- [ ] ≥10 segments tested
- [ ] Pass rate documented (with and without tweaking)
- [ ] Fourier diagnostic tested on all segments
- [ ] Tweaking log complete — categorized by failure mode

---

## Phase 2: Analysis & Novel Contributions — Paper-Ready Results
**Goal:** Produce publication-quality results, figures, and theoretical framing
**Status:** Not Started

| Priority | Task | Description | Dependencies |
|----------|------|-------------|-------------|
| P0 | Max-error penalty analysis | Formal description + ablation across all segments, not just segment 23 | Phase 1 complete |
| P0 | Fourier feature justification | Spectral analysis showing why linear spacing works for gyro signals | Diagnostic from Phase 1 |
| P1 | Quaternion supervision comparison | Run interior-supervision variant on same segments, show it's harder | Phase 1 segments |
| P1 | Training dynamics visualization | Loss curves, MaxErr trajectories, phase transition plots for all segments | Phase 1 results |
| P2 | Sensitivity analysis | Vary network width, depth, learning rates — characterize robustness envelope | Phase 1 complete |

**Exit criteria:**
- [ ] All figures publication-quality (vector/high-DPI)
- [ ] Ablation results span multiple segments
- [ ] Fourier feature choice is mechanistically justified
- [ ] Quaternion supervision comparison documented

---

## Phase 3: Paper Writing — Draft to Submission
**Goal:** Complete paper draft suitable for journal submission and arXiv upload
**Status:** Not Started

| Priority | Task | Description | Dependencies |
|----------|------|-------------|-------------|
| P0 | Methods section | PINN architecture, loss function, training strategy, Fourier features | Phase 2 analysis |
| P0 | Results section | Multi-segment results, ablation, robustness findings | Phase 2 figures |
| P0 | Introduction | Position within PINN literature, motivation from Karniadakis/Raissi | Existing intro drafts + Phase 2 |
| P1 | Discussion | Quaternionic loss geometry insight, implications for QDE solvers | Phase 2 comparison |
| P1 | Abstract + conclusion | Written last, summarizing findings | All other sections |
| P2 | LaTeX conversion | Convert markdown sections to submission-ready LaTeX | All sections drafted |

**Exit criteria:**
- [ ] All sections drafted in markdown
- [ ] Internal review pass (Vivek)
- [ ] LaTeX compiled and formatted for target journal
- [ ] arXiv-ready PDF generated

---

## Phase 4: Dogfooding Narrative — Process Documentation
**Goal:** Document the Telegram voice-driven research workflow as a secondary contribution
**Status:** In Progress (extraction pipeline exists)

| Priority | Task | Description | Dependencies |
|----------|------|-------------|-------------|
| P0 | Complete voice note extractions | All research-relevant exchanges extracted via slice-extract pipeline | Ongoing |
| P1 | Process analysis | Document how voice-driven direction affected research trajectory | Paper draft |
| P2 | Project webpage | Cleaned-up public-facing page showing results and process | Paper submission |

**Exit criteria:**
- [ ] Voice note transcripts organized by research phase
- [ ] Process narrative integrated into paper or supplementary material
- [ ] Project webpage live

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|-----------|
| Method fails on diverse segments | High | Medium | Track failure modes, adapt architecture per-segment, report honestly |
| Fourier features need per-segment tuning | Medium | Medium | Build automated spectral diagnostic to make it principled |
| Quaternion supervision comparison is inconclusive | Low | Medium | Frame as open question rather than definitive claim |
| Single-GPU training bottleneck for many segments | Low | Low | Parallelize across segments, each is ~3 min on GPU |
| Paper framing too niche for broad audience | Medium | Medium | Emphasize general PINN-for-ODEs contribution, rotation is illustrative |

## Open Questions
- How many segments constitute "enough" for a robustness claim? (10? 20? All non-stationary?)
- Target journal: applied math, computational physics, or ML venue?
- Should the dogfooding narrative be part of the main paper or a separate blog post?
- Does the quaternionic loss geometry insight warrant its own experiment section or just a discussion paragraph?

## Out of Scope
- Real-time PINN inference (this is an offline estimation problem)
- Quaternion-valued neural networks (noted as future work, not implemented here)
- Full IMU fusion (accelerometer + magnetometer) — gyro-only is the scope
- ChatGPT/browser Claude integrations (infrastructure, not research)
- Phone call layer integration (deferred)
