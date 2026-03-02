# TaskSpec: 000 - <short-title>

Status: <draft | ready | in-progress | implemented>

Note: Update Status and check off completed acceptance criteria as part
of the feature branch —
never as a separate commit to `main`. The TaskSpec update must be
included in the PR
that delivers the work.

Copy this file for new tasks and fill in the blanks. Keep it short and
testable.

## Objective

What outcome are we trying to achieve (1-2 sentences)?

## Context / References

- Relevant docs, code paths, design notes (links/paths).

## Scope

**In scope:**
- …

**Out of scope:**
- …

## Guardrails / Constraints

- All work runs inside Docker containers — no host-native Python.
- No new Python dependencies without justification.
- Any additional constraints specific to this task.

## Acceptance Criteria (must be testable)

- [ ] …
- [ ] …

## Verification

```bash
docker compose run --rm app pytest tests/unit/test_<module>.py -v
docker compose run --rm app pytest tests/integration/test_<module>.py -v
docker compose run --rm app ruff check tsd/
docker compose run --rm app mypy tsd/
```

Add any seed data, config, or setup steps needed before verification.

## Deliverables

- Code changes (paths).
- Tests (paths).
- Docs to update (paths).

## Risks / Open Questions

- …

## Learnings

Populated during and after implementation. Remove this placeholder text
when adding entries.

- Reusable patterns discovered, surprising findings, or architectural
  insights worth carrying forward.
- What worked well, what was harder than expected, and what the next
  TaskSpec should know.

## Follow-ups / Backlog (if not done here)

- …
