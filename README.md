# Extending PINNs to Quaternionic Differential Equations

Research project extending physics-informed neural networks (PINNs) to quaternionic differential equations, with rigid body rotational motion as the proof of concept — written entirely via Telegram voice notes and remote control slash commands.

## Overview

This project has two goals:

1. **Finish and publish** independent research on extending PINNs (in the Karniadakis sense) to quaternionic differential equations (QDEs). Quaternions are the natural language for rotational dynamics, just as complex numbers are fundamental to electromagnetism and quantum mechanics. We customize JAX to output unit quaternions from real-valued neural networks, build quaternionic operations compatible with JAX's autodiff engine, and validate on industry-grade IMU sensor data from Movella.

2. **Dogfood a voice-first workflow.** All work is driven from a phone — Telegram voice notes for context and direction, remote control slash commands for execution. No sitting at a desk. Claude is the sole writer and builder; Vivek directs verbally.

## Key Results

### Segment 23 — Gyro-Only Boundary Value Problem (PASS)

The PINN solves the quaternionic kinematic ODE **ω = 2q\*⊗q̇** using only:
- Measured angular velocity (gyroscope data, 720 samples at 200 Hz)
- Two boundary quaternions (q_i, q_f) — no interior quaternion supervision

**Max error: 0.0062 rad/s** (pass threshold: ±0.05 rad/s)

### Architecture
- MLP: [129, 512, 512, 256, 4] with tanh activations
- Fourier features: 64 linearly spaced frequencies (scale 4.0)
- Output: normalized unit quaternions
- Framework: JAX with GPU (CUDA)

### Training Strategy
- **Phase 1** (15K epochs): BC-anchored (w_bc=500, w_omega=1, w_max=0)
- **Phase 2** (30K epochs): Physics-focused (w_bc=100, w_omega=2, w_max=10)
- **Phase 3** (20K epochs): Max-error refinement (w_bc=100, w_omega=2, w_max=50)

### Novel Contribution: Soft L-infinity Penalty
A max-error penalty term `mean(max(|ω_err| - threshold, 0)²)` stabilizes training and reduces max error by 35% (0.0062 vs 0.0096 without). Ablation study confirms the term prevents the optimizer from creating large local errors while minimizing global MSE.

### Key Findings
- Gyro-only BVP (no interior quaternion supervision) achieves 5x better accuracy than the supervised version
- Hypothesis: quaternionic loss geometry conflicts with real-valued latent space and optimizer — the angular velocity formulation avoids this by comparing in R³
- Linearly spaced Fourier features (not 2^L NeRF-style) resolve spectral bias for gyro signals

## Project Structure

| File | Purpose |
|------|---------|
| `project_high_level_idea.md` | Three-point project summary |
| `rules_of_the_game.md` | 8 rules governing the voice-first workflow |
| `ROADMAP.md` | Multi-phase project roadmap |
| `section_writing_strategy.md` | 5-phase pipeline: conversation → extraction → context → polish → final |
| `introduction_*.md` | Introduction drafts, verbatim transcripts, and Claude responses |
| `inspiration_vibe_physics.md` | Notes on Anthropic's "Vibe Physics" article by Matthew Schwartz |
| `observations.md` | Workflow observations and discoveries |
| `skills_created_here.md` | Index of skills created during this project |
| `Telegram_calls.md` | Audit log of all Telegram exchanges |
| `literature_review/` | Background literature search on quaternionic NNs and PINNs |

### Claude Sandbox (`claude_sandbox/`)

Isolated workspace for PINN solving — Claude's independent attempt at the problem.

| File | Purpose |
|------|---------|
| `01_preprocess.py` | Extract non-stationary segments from raw IMU data |
| `02_pinn.py` | Initial PINN implementation (with reference quaternions) |
| `03_run_all.py` | Batch runner for all segments |
| `04_solve_segment23.py` | Segment 23 solver with interior quaternion supervision |
| `05_solve_segment23_pinn.py` | **True PINN**: gyro-only BVP solver (the working solution) |
| `06_ablation_max_term.py` | Ablation study: with vs without max-error penalty |
| `07_ablation_with_curves.py` | Ablation with training dynamics curves |
| `problem-statement/` | Problem definition, BCs, and additional context |
| `segments/` | Extracted segment data (gyro + time only) |
| `results/` | Output plots (angular velocity overlays, ablation comparisons) |

## The Writing Pipeline

Each paper section follows a 5-phase process:

1. **Conversation** — Natural back-and-forth via Telegram voice. Raw, unstructured, multi-session.
2. **Extraction** — Slash commands pull verbatim transcriptions and Claude's responses into dedicated markdown files.
3. **Additional Context** — External references, existing files, and browser research enrich the material.
4. **Conversational Polish** — Discussion and debate to refine understanding.
5. **Final Polish** — Polished draft written into `section_polished.md`.

## Skills Created

Seven globally-available [Claude Code](https://claude.ai/code) skills were built during this project:

| Skill | Purpose |
|-------|---------|
| `new-md` | Create empty markdown files |
| `extract-tg-exchange` | Find and extract a Telegram exchange from the audit log |
| `verbatim-input-transcribed-from-tg-exchange` | Extract verbatim transcribed voice input |
| `claude-tool-output-from-tg-exchange` | Extract Claude's reply text |
| `tg-exchange-loop-extract` | Batch extract over a message range (composes the above) |
| `tg-exchange-loop-slice-extract` | Incremental extraction, picking up from where the last extraction ended |
| `new-md-tg-exchange` | Create the verbatim + claude file pair for a section |

## The 8 Rules

1. Laptop terminal is last resort
2. Text prompts come from phone via remote control with slash commands
3. Telegram voice is the default channel
4. Repeated tasks become skills (Unix philosophy)
5. Vivek writes nothing — all writing is delegated
6. No new schematics — only existing ones provided
7. One markdown file per paper section
8. Delegate to subagents aggressively

## Inspiration

This project draws from [Vibe Physics: The AI Grad Student](https://www.anthropic.com/research/vibe-physics) by Matthew Schwartz (Harvard), who completed a theoretical physics paper using Claude — but goes further by making voice notes and mobile slash commands the *only* interface.

## Tech Stack

- **Claude Code** — AI research collaborator and sole writer
- **JAX** — Quaternionic operations with autodiff, Fourier features, GPU-accelerated training
- **Optax** — Optimizer (Adam with cosine decay schedules)
- **Telegram** — Voice note channel with transcription pipeline
- **Whisper** — Voice transcription
- **NVIDIA GPU** — CUDA-accelerated PINN training (~3 min per segment)

## License

No license specified yet.

---

🤖 Generated with [Claude Code](https://claude.ai/code)
