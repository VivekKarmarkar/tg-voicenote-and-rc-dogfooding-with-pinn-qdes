# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This is a research project to finish and publish independent work on extending PINNs (physics-informed neural networks) to quaternionic differential equations (QDEs), with rotational motion as the illustrative example. The deliverables are a polished paper (journal submission + arXiv), and a cleaned-up project webpage.

The project doubles as a dogfooding experiment: all work is driven through Telegram voice notes and remote control slash commands from Vivek's phone. Claude is the sole writer and builder; Vivek directs verbally.

## Cardinal Rule: NEVER touch anything that works

Never modify, refactor, or "improve" existing working code, hooks, skills, or configurations. Under no circumstances. Always build new, clean, lean, modular things — one thing at a time, doing one thing well, strictly adhering to the Unix philosophy. Do not create monolithic anything. Do not ask whether to modify working systems. The answer is always no.

## Interaction Rules

These rules govern how this project operates. Follow them strictly.

1. **Telegram voice is the default channel.** Vivek's messages arrive as voice notes. Text prompts and laptop use are exceptions, not the norm.
2. **Text prompts come from phone via remote control and must involve a slash command**, unless one of the laptop-only conditions below applies.
3. **Laptop terminal is last resort** — only for: (a) creating skills, (b) recovering from internal bash errors, (c) masking sleep before going mobile, (d) transferring files from university Microsoft account.
4. **Repeated tasks become skills.** If a task recurs (evidenced by prompts/transcripts), convert it into a skill. Complex tasks get broken into composable skills following the Unix philosophy (one skill, one job).
5. **Vivek writes nothing.** All code, LaTeX, English text, equations, web pages, and schematics are written by Claude. Vivek directs; Claude executes.
6. **Vivek provides existing schematics only.** He will not create new ones — Claude edits, enhances, or rebuilds from existing references or verbal descriptions.
7. **One markdown file per paper section.** No monolithic files. Each section gets a dedicated .md file. Sections with multiple subsections get a .md per subsection as well. All drafting happens in markdown before conversion to .tex.
8. **Delegate to subagents aggressively.** Whenever a task can be handled by agents or agentic teams, delegate it to keep the main conversation context lean.

## Repository Structure

- `project_high_level_idea.md` — project motivation and goals
- `rules_of_the_game.md` — the 8 rules governing the workflow
- `Telegram_calls.md` — audit log of Telegram reply tool invocations
- Paper sections will be individual markdown files (per Rule 7) as they are created
