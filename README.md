# Extending PINNs to Quaternionic Differential Equations

Research project extending physics-informed neural networks (PINNs) to quaternionic differential equations, with rigid body rotational motion as the proof of concept — written entirely via Telegram voice notes and remote control slash commands.

## Overview

This project has two goals:

1. **Finish and publish** independent research on extending PINNs (in the Karniadakis sense) to quaternionic differential equations (QDEs). Quaternions are the natural language for rotational dynamics, just as complex numbers are fundamental to electromagnetism and quantum mechanics. We customize JAX to output unit quaternions from real-valued neural networks, build quaternionic operations compatible with JAX's autodiff engine, and validate on industry-grade IMU sensor data from Movella.

2. **Dogfood a voice-first workflow.** All work is driven from a phone — Telegram voice notes for context and direction, remote control slash commands for execution. No sitting at a desk. Claude is the sole writer and builder; Vivek directs verbally.

## Key Results

### Generalized Solver — 6 Datasets Tested

A single algorithm (`05_solve_segment_general_pinn.py`) solves the quaternionic kinematic ODE **ω = 2q\*⊗q̇** across all test datasets using only:
- Measured angular velocity (gyroscope data)
- Two boundary quaternions (q_i, q_f) — no interior quaternion supervision

| Dataset | Max Error (rad/s) | Pass (±0.1) |
|---------|-------------------|-------------|
| seg23 | 0.0269 | ✓ |
| trial_000 | 0.0203 | ✓ |
| trial_006 | 0.0854 | ✓ |
| trial_016 | 0.0201 | ✓ |
| trial_011 | 0.1714 | ✗ |
| trial_009 | 0.4328 | ✗ |

**4/6 pass** with a single set of hyperparameters. No dataset-specific tuning.

### Architecture: Learnable Fourier Series + FF-PINN Correction

The architecture decomposes quaternion prediction into two parts:

1. **Linear component** — Learnable Fourier series (trainable coefficients on fixed frequency bases)
2. **Nonlinear correction** — A feed-forward PINN (MLP) that takes the Fourier features as input, not raw time

This makes the correction an FF-PINN perturbation to a spectral method, not a generic neural network approximation.

- Fourier features: linearly spaced frequencies (scale 4.0), count adapted to signal duration
- MLP: [Fourier features → 128 → 128 → 4] with tanh activations
- Output: normalized unit quaternions
- Framework: JAX with `vmap(jacfwd)` for automatic differentiation

### Novel Contribution: Soft L-infinity Penalty

A max-error penalty term `mean(max(|ω_err| - threshold, 0)²)` prevents the optimizer from creating large local errors while minimizing global MSE. Ablation study confirms 35% max-error reduction on segment 23.

### Key Findings

- Gyro-only BVP (no interior quaternion supervision) dramatically outperforms the supervised version
- Hypothesis: quaternionic loss geometry conflicts with real-valued latent space and optimizer — the angular velocity formulation compares in R³ instead
- Linearly spaced Fourier features (not 2^L NeRF-style) resolve spectral bias for gyro signals

## Project Structure

| Path | Purpose |
|------|---------|
| `project_high_level_idea.md` | Three-point project summary |
| `rules_of_the_game.md` | 8 rules governing the voice-first workflow |
| `ROADMAP.md` | Multi-phase project roadmap |
| `section_writing_strategy.md` | 5-phase pipeline: conversation → extraction → context → polish → final |
| `introduction_*.md` | Introduction drafts, verbatim transcripts, and Claude responses |
| `inspiration_vibe_physics.md` | Notes on Anthropic's "Vibe Physics" article by Matthew Schwartz |
| `affection.md` | Vivek's affection journal — moments where Claude was endearing |
| `ai_agentic_discovery/` | Papers on AI-driven scientific discovery experiments |
| `ai_creativity_papers/` | Papers on AI and creativity (incl. Terence Tao) |
| `literature_review/` | Background literature search on quaternionic NNs and PINNs |

### Claude Sandbox (`claude_sandbox/`)

Isolated workspace for PINN solving — Claude's independent attempt at the problem.

| Path | Purpose |
|------|---------|
| `05_solve_segment_general_pinn.py` | **The working solver** — single algorithm for all datasets |
| `05_solve_segment23_pinn.py` | Original single-segment solver (segment 23 only) |
| `06_ablation_max_term.py` | Ablation study: with vs without max-error penalty |
| `07_ablation_with_curves.py` | Ablation with training dynamics curves |
| `test_files/` | CSV datasets: gyro data + metadata for all 6 test cases |
| `results/` | Per-dataset angular velocity overlay plots |
| `example_plots/` | Reference plots from successful runs |
| `problem-statement/` | Versioned problem statements (1, 2, 3) — the "game rules" |
| `attempt_history.md` | Game log tracking attempt times against 40-minute budget |

### Earlier Sandbox Files

| Path | Purpose |
|------|---------|
| `01_preprocess.py` | Extract non-stationary segments from raw IMU data |
| `02_pinn.py` | Initial PINN implementation (with reference quaternions) |
| `03_run_all.py` | Batch runner for all segments |
| `04_solve_segment23.py` | Segment 23 solver with interior quaternion supervision |

## The Writing Pipeline

Each paper section follows a 5-phase process:

1. **Conversation** — Natural back-and-forth via Telegram voice. Raw, unstructured, multi-session.
2. **Extraction** — Slash commands pull verbatim transcriptions and Claude's responses into dedicated markdown files.
3. **Additional Context** — External references, existing files, and browser research enrich the material.
4. **Conversational Polish** — Discussion and debate to refine understanding.
5. **Final Polish** — Polished draft written into `section_polished.md`.

## Skills Created

Eight globally-available [Claude Code](https://claude.ai/code) skills were built during this project:

| Skill | Purpose |
|-------|---------|
| `new-md` | Create empty markdown files |
| `extract-tg-exchange` | Find and extract a Telegram exchange from the audit log |
| `verbatim-input-transcribed-from-tg-exchange` | Extract verbatim transcribed voice input |
| `claude-tool-output-from-tg-exchange` | Extract Claude's reply text |
| `tg-exchange-loop-extract` | Batch extract over a message range (composes the above) |
| `tg-exchange-loop-slice-extract` | Incremental extraction, picking up from where the last extraction ended |
| `new-md-tg-exchange` | Create the verbatim + claude file pair for a section |
| `obsidian-code-viz` | Visualize a codebase as an Obsidian vault graph with wikilinked notes |

## The 8 Rules

1. Telegram voice is the default channel
2. Text prompts come from phone via remote control with slash commands
3. Laptop terminal is last resort
4. Repeated tasks become skills (Unix philosophy)
5. Vivek writes nothing — all writing is delegated
6. No new schematics — only existing ones provided
7. One markdown file per paper section
8. Delegate to subagents aggressively

## Inspiration

This project draws from:

- [Vibe Physics: The AI Grad Student](https://www.anthropic.com/research/vibe-physics) by Matthew Schwartz (Harvard), who completed a theoretical physics paper using Claude — but goes further by making voice notes and mobile slash commands the *only* interface.
- [Tips for Long-Running Claude Code Sessions](https://www.anthropic.com/engineering/claude-code-best-practices) — Anthropic's engineering insights on maintaining productive multi-hour sessions.
- Papers on AI-driven scientific discovery (`ai_agentic_discovery/`)
- Papers on AI and creativity (`ai_creativity_papers/`)

## Tech Stack

- **JAX** — Quaternionic operations with autodiff, Fourier features, GPU-accelerated training
- **Optax** — Optimizer (Adam with cosine decay schedules)
- **Telegram** — Voice note channel with transcription pipeline
- **Whisper** — Voice transcription
- **NVIDIA GPU / Google Colab TPU** — Hardware-accelerated PINN training
- **Obsidian** — Code architecture visualization via graph view

## License

No license specified yet.

---

🤖 Generated with [Claude Code](https://claude.ai/code)
