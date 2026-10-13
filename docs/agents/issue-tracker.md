# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `chillyweather/russian-tts`. Use the `gh` CLI with `--repo chillyweather/russian-tts` where supported.

## Issue operations

- Publish one GitHub issue per approved ticket. Supply multiline bodies through a file using `--body-file`.
- Read the full issue body, labels, and comments with `gh issue view <number> --comments`; use JSON output when structured data is needed.
- List issues with `gh issue list`, filtering by state and labels.
- Change labels with `gh issue edit`; use the role mapping in `triage-labels.md`.
- Post comments with `gh issue comment --body-file`. Close work only when its acceptance criteria are satisfied.
- Preserve existing parent issues. If the source is a GitHub issue, link tickets as native sub-issues using the supported CLI or REST operation; if unavailable, reference the parent in the child body.

## Dependencies and frontier

Publish blockers before dependents. Use native GitHub blocking relationships:
`gh api --method POST repos/chillyweather/russian-tts/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-database-id>`.

Get the numeric database ID from `gh api repos/chillyweather/russian-tts/issues/<number> --jq .id`; it differs from the issue number and node ID. If native dependencies are unavailable, add a body section listing blocking issue references.

Work only on open, unclaimed tickets whose blockers are complete. The `ready-for-agent` label describes implementation readiness; it does not override open blockers.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## Wayfinding operations

Use a map issue labeled `wayfinder:map`, with native sub-issues labeled by type: `wayfinder:research`, `wayfinder:prototype`, `wayfinder:grilling`, or `wayfinder:task`. Maintain Notes, Decisions-so-far, and Fog in the map body. If sub-issues are unavailable, maintain a task list in the map and parent references in children.

Claim an unblocked child by assigning it to the driving developer. On resolution, record its answer, close the child, and add a concise result and link to the map's Decisions-so-far.
