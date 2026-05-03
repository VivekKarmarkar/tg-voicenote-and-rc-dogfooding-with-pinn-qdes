# Rules of the Game

## Rule 1: Laptop terminal is only used when Telegram voice notes and remote control cannot be used

The laptop terminal is a last resort. It is only allowed under these four conditions:

1. **Creating a skill** — skill creation from remote control causes an internal bash error
2. **Internal bash error recovery** — if an internal bash error occurs for another reason, need to go to the laptop and press double-escape
3. **Masking sleep** — need to run the sleep mask command as a skill in one terminal, plus a separate terminal with sudo to mask sleep, before going mobile
4. **Transferring files from university Microsoft account** — files sitting in a university Microsoft account that isn't connected to the remote system, so they must be transferred manually at the laptop

## Rule 2: Text prompts must come from phone via remote control and must involve a slash command

Unless a text prompt falls under one of Rule 1's four conditions (requiring the laptop), all typed prompts must be sent from the phone via remote control on the Claude app — and they must involve a slash command.

## Rule 3: All messages must come from Telegram voice unless Rule 1 and/or Rule 2 applies

The default communication channel is Telegram voice notes. Text prompts (Rule 2) and laptop terminal use (Rule 1) are the only exceptions.

## Rule 4: Any repeated task must be converted into a skill (Unix philosophy)

If there is evidence (via prompts or transcripts) that a task is being repeated, it must be converted into a skill. If the task is complex, it must be broken down so that each skill does one thing well, and those skills are composed together. All skills must strictly adhere to the Unix philosophy.

## Rule 5: Vivek does not write anything — all writing is delegated

Vivek will not write any code, LaTeX, English text, equations, web pages, or anything else by hand. All writing is directed verbally and delegated entirely to Claude. Vivek's role is to direct; Claude's role is to write.

## Rule 6: Vivek does not create new schematics — only provides existing ones

Vivek will not create any new schematics. He may provide existing schematics he has already made, which Claude can edit, enhance, or use as a template to build from scratch. Vivek can describe schematics verbally, but all creation of new schematics is delegated to Claude.

## Rule 7: Dedicated sections get dedicated markdown files — no monolithic dumping

Each section of the paper gets its own self-contained markdown file. This includes rough thoughts, audio transcripts, Claude's thoughts, and drafts — all living in that section's markdown file before anything is converted to .tex.

If a section has more than one subsection, each subsection also gets its own dedicated markdown file. For example, Section N with subsections N.1 and N.2 would have:
- A markdown file for Section N
- A markdown file for Subsection N.1
- A markdown file for Subsection N.2

## Rule 8: Delegate to agents and agentic teams whenever possible to avoid context bloat

Whenever a task can be delegated to subagents or agentic teams, it should be. This keeps the main conversation context lean and prevents bloat.
