# Validity of the single processor approach to achieving large scale computing capabilities

**Author:** Gene M. Amdahl (IBM, United States)
**Year:** 1967
**Publication date:** 1967-01-01
**Venue:** AFIPS Spring Joint Computer Conference 1967, Atlantic City NJ, April 18–20, pp. 483–485 *(OpenAlex `primary_location.source` is null for this defunct proceedings series — venue filled from the paper itself)*
**Type:** Conference article
**Language:** English
**Open access:** No (status: closed)

## Identifiers

- **DOI:** [10.1145/1465482.1465560](https://doi.org/10.1145/1465482.1465560)
- **OpenAlex ID:** [W2150871235](https://openalex.org/W2150871235)
- **Citations (OpenAlex):** 4,190
- **References:** 0 (OpenAlex has not parsed reference list)

## Abstract

For over a decade prophets have voiced the contention that the organization of a single computer has reached its limits and that truly significant advances can be made only by interconnection of a multiplicity of computers in such a manner as to permit cooperative solution. Variously the proper direction has been pointed out as general purpose computers with a generalized interconnection of memories, or as specialized computers with geometrically related memory interconnections and controlled by one or more instruction streams.

## Topics

- Parallel Computing and Optimization Techniques (score 0.89)
- Interconnection Networks and Systems (score 0.87)

## Concepts

| Concept | Level | Score |
|---|---|---|
| Interconnection | 2 | 0.86 |
| Computer science | 0 | 0.78 |
| Parallel computing | 1 | 0.46 |
| Scale (ratio) | 2 | 0.43 |
| Computer architecture | 1 | 0.43 |
| Distributed computing | 1 | 0.36 |
| Computer network | 1 | 0.18 |

## Why this paper matters

This is the seminal statement of what became known as **Amdahl's Law** — the theoretical ceiling on speedup achievable by parallelizing a computation when a fraction of the work remains inherently sequential. Although the algebraic form `S = 1 / ((1 − p) + p/N)` is not literally written in this paper, Amdahl argued the ceiling verbally and with example workloads, showing that even a small sequential housekeeping fraction caps total speedup regardless of processor count. The argument continues to bound the limits of parallel and distributed systems six decades later, and frames every contemporary discussion of bottlenecks in computer architecture, GPU scaling, and now LLM training pipelines.

## Source PDF

- Local file: `amdahls_law_1967.pdf` (288,357 bytes, PDF v1.5)
- Retrieved via `/paper-download-hack` → Cat 3 (Sci-Hub.ru fallback after ACM 403 and Unpaywall no-OA result)

## Metadata provenance

- Source: OpenAlex API (`https://api.openalex.org/works/doi:10.1145/1465482.1465560`)
- Retrieved: 2026-05-14
- Note: Helper script `~/.claude/skills/paper-metadata/helpers/build_metadata_pdf.py` crashed on `primary_location.source = null` (OpenAlex coverage gap for AFIPS proceedings). Per project cardinal rule, helper was not modified; metadata was fetched directly via the OpenAlex API and formatted as markdown.
