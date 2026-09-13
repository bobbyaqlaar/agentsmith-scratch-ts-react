# Scratch tenant — managed by automation

This repository is a **test fixture for AgentSmith**, not a product. A small
app is onboarded by AgentSmith's provisioning hook so that tenant CI runs for
real on GitHub.

AgentSmith's **Scratch tenants** workflow re-provisions this repo from the
current framework (weekly, and whenever provisioning changes), pushes the
result, and fails unless this repo's CI goes green.

- **Provisioned (overwritten every run — don't edit here):** `.github/`,
  `runtime/`, `fixtures/`, vendored `scripts/`, `.agents/`, `.cursorrules`,
  `CLAUDE.md`, `AGENTS.md`, `GEMINI.md`, `.agent-rfc/` except `security/`.
- **Owned by this repo (edit freely):** the app, `.agent-rfc/security/`, and
  anything else not listed above.

Full documentation, setup and triage:
https://github.com/bobbyaqlaar/AgentSmith/blob/main/docs/scratch-tenants.md
