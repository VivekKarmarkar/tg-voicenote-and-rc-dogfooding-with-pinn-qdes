# Extending PINNs to Quaternionic Differential Equations

Research project extending physics-informed neural networks (PINNs) to quaternionic differential equations, with rigid body rotational motion as the proof of concept — written entirely via Telegram voice notes and remote control slash commands.

## Overview

This project has two goals:

1. **Finish and publish** independent research on extending PINNs (in the Karniadakis sense) to quaternionic differential equations (QDEs). Quaternions are the natural language for rotational dynamics, just as complex numbers are fundamental to electromagnetism and quantum mechanics. We customize JAX to output unit quaternions from real-valued neural networks, build quaternionic operations compatible with JAX's autodiff engine, and validate on industry-grade IMU sensor data from Movella.

2. **Dogfood a voice-first workflow.** All work is driven from a phone — Telegram voice notes for context and direction, remote control slash commands for execution. No sitting at a desk. Claude is the sole writer and builder; Vivek directs verbally.

## Key Findings

- Vanilla PINNs fail out of the box on quaternionic DEs due to **spectral bias**
- **Wavelet scalograms** serve as a practical diagnostic for spectral bias
- **Fourier features** + quaternionic neural network representation + custom quaternion ops in JAX + initial condition loss accounting for quaternions = working PINN extension to QDEs

## Project Structure

| File | Purpose |
|------|---------|
| `project_high_level_idea.md` | Three-point project summary |
| `rules_of_the_game.md` | 8 rules governing the voice-first workflow |
| `section_writing_strategy.md` | 5-phase pipeline: conversation → extraction → context → polish → final |
| `introduction_verbatim.md` | Verbatim transcribed voice inputs for the introduction |
| `introduction_claude.md` | Claude's structured responses to each introduction ramble |
| `introduction_polished.md` | Final polished introduction (in progress) |
| `inspiration_vibe_physics.md` | Notes on Anthropic's "Vibe Physics" article by Matthew Schwartz |
| `observations.md` | Workflow observations and discoveries |
| `skills_created_here.md` | Index of skills created during this project |
| `Telegram_calls.md` | Audit log of all Telegram exchanges |
| `CLAUDE.md` | Project guidance for Claude Code sessions |

Audio versions (`.mp3`) of key documents are generated via `/audify-and-share` and shared on Google Drive.

## The Writing Pipeline

Each paper section follows a 5-phase process:

1. **Conversation** — Natural back-and-forth via Telegram voice. Raw, unstructured, multi-session.
2. **Extraction** — Slash commands pull verbatim transcriptions and Claude's responses into dedicated markdown files.
3. **Additional Context** — External references, existing files, and browser research enrich the material.
4. **Conversational Polish** — Discussion and debate to refine understanding.
5. **Final Polish** — Polished draft written into `section_polished.md`.

## Skills Created

Five globally-available [Claude Code](https://claude.ai/code) skills were built during this project:

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
- **JAX** — Framework for custom quaternionic operations with autodiff compatibility
- **Telegram** — Voice note channel with transcription pipeline
- **Whisper** — Voice transcription
- **PyTorch/JAX** — PINN implementation

## License

No license specified yet.

---

🤖 Generated with [Claude Code](https://claude.ai/code)
