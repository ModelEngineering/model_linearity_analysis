# Course Outliner Agent — Design Spec

**Date:** 2026-06-05

## Overview

A Claude Code subagent that generates and reviews course outlines for technical/programming courses focused on AI-assisted software development (specifications, implementation, code reviews, features, tests).

## Agent Identity

- **File:** `~/.claude/agents/course-outliner-agent.md` (global)
- **Model:** sonnet
- **Tools:** Read, Write, Edit

## Invocation

The parent agent (or user) invokes the subagent by passing two parameters in natural language:

- **mode:** `generate` or `review`
- **file path(s):** objectives file (generate) or outline file + objectives file (review)

## Generate Mode

**Input:** path to a course objectives markdown file containing draft learning objectives for the full course.

**Output:** writes `course-outline.md` to the same directory as the objectives file.

### Episode structure (one section per episode):

| Field | Description |
|---|---|
| Title | Short descriptive name |
| Duration | Estimated time (e.g., "45 min") |
| Topic | One-sentence summary of what the episode covers |
| Learning Objectives | Bullet list — what the learner will be able to do after this episode |
| Concepts & Skills | Bullet list of what is taught; notes which prior episodes each concept builds on |
| Exercises | 1–3 exercises with brief descriptions |

**Sequencing rule:** episodes are ordered so that every concept or skill used in an episode has been introduced in an earlier episode. The agent derives the number and order of episodes from the objectives — no fixed count.

## Review Mode

**Input:** path to the user-edited outline file + path to the original objectives file.

**Output:** a structured markdown review returned as response text in the Claude sidebar panel (not written to a file).

### Review covers four areas:

1. **Pedagogical sequencing** — are concepts introduced before they are used? Are there dependency gaps or ordering problems?
2. **Objectives coverage** — does the set of episodes fully address all course learning objectives? Any gaps or redundancies?
3. **Exercise quality** — are exercises concrete, appropriately scoped, and aligned with the episode's learning objectives?
4. **Duration estimates** — are episode lengths plausible given the scope of content and exercises?

Each finding is an actionable suggestion with a specific recommendation, not a pass/fail grade.

## Out of Scope

- Writing individual episode files (a separate concern)
- Generating course content or lecture notes
- Any format other than markdown
