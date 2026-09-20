# Security model

The harness is designed for local learning and controlled agent experiments. It reduces
risk; it is not a hardened multi-tenant isolation boundary. Run untrusted workloads inside a
dedicated container or VM in addition to these controls.

## Assets and trust boundaries

Protected assets include host files outside the configured workspace, secrets in the process
environment, network services, approval authority, persistent memory belonging to another
namespace, and the integrity of run history. Model output, retrieved documents, user-provided
task text, MCP servers, and workspace repositories are untrusted inputs.

## Default policy

- The offline mock provider is selected by default.
- Network tools and MCP servers are disabled until explicitly configured.
- Paths are resolved against a per-run workspace; absolute paths, traversal, symlinks,
  junctions, and hard-linked files are rejected.
- Workspace failures are typed: policy denials raise `SandboxViolation` (a
  `PermissionError`) with a stable machine-readable `code`, missing files raise
  `FileNotFoundError`, and exclusive creates that lose to an existing target raise
  `FileExistsError`. Tool adapters map these to the `error_code` field instead of a
  generic failure, so callers can distinguish denial, absence, and conflict.
- Parent directories are created component-by-component and re-verified after every
  concurrent-create race; a parent swapped for a link or a regular file fails closed.
  Opened files are re-identified through their descriptor (device/inode, link count,
  regular-file check) before any byte is read or written, and overwrite truncation
  happens only after that check, so a concurrently replaced path cannot redirect an
  operation outside the workspace or leave partial content behind.
- Read, write, process, and network effects are declared in tool metadata. Undeclared tools
  are denied.
- Side-effecting executors are wrapped with `SecuredToolExecutor`. Writes require approval
  by default; process, network, and critical-risk capabilities are disabled until explicitly
  enabled. An application may intentionally grant a narrower pre-approved capability, but
  that exception must be expressed in tool metadata rather than hidden in a prompt.
- Approvals are tied to exact arguments, the run ID, tool name, and current checkpoint.
- Approvals expire and are consumed once. A stale or replayed decision fails closed.
- If a process exits after recording an approval decision but before resuming the run, the
  resolved-but-unconsumed request stays visible and the same decision can resume it without
  creating a second decision event. The opposite decision is rejected.
- If an approval was consumed but no tool result was durably recorded, the harness does not
  automatically replay the side effect. An operator must reconcile the external state, or the
  tool must provide durable idempotency beyond the process-local demonstration receipt cache.
- Command execution uses a small allowlist and argument validation. There is no arbitrary
  shell tool in v1.
- Tool output is bounded before it can enter model context or the event log.
- Child agents receive only explicitly delegated capabilities and cannot reserve more than
  the parent's remaining budget.
- API errors and traces never include provider credentials. `.env` is ignored by Git.

## Threats and controls

| Threat | Control | Residual risk |
| --- | --- | --- |
| `../` or absolute-path escape | canonical root-relative resolver | OS/filesystem defects |
| symlink/junction escape | reject every linked path component; re-verify parents after create races | time-of-check/time-of-use on hostile hosts |
| hard-link escape | reject regular files whose link count is not one, checked again on the open descriptor | link creation race on hostile hosts |
| path replaced mid-operation | descriptor identity (device/inode) re-check before I/O; deferred truncation | residual kernel-level race on hostile hosts |
| concurrent parent creation | atomic mkdir with post-race directory re-validation | none known locally |
| prompt-injected tool request | deterministic policy outside the model | an approved action may still be harmful |
| approval replay | arguments digest, revision, expiry, one-time consume | stolen live operator session |
| crash before approval consumption | retry the persisted decision; checkpoint CAS and one-shot consume | expired approval still fails closed |
| crash after approval consumption | no automatic replay; operator reconciliation or durable tool idempotency | result may be unknown until reconciled |
| repeated write after crash | stable idempotency key and tool-owned durable receipt | demo registry receipts are process-local |
| secret leakage in logs | structured safe fields and provider error normalization | secrets deliberately put in user content |
| cross-run memory leak | required namespace in every read/write and composite identity | application choosing a shared namespace |
| corrupt vector index | finite-value checks and one embedding dimension per namespace | poor retrieval from a weak embedder |
| malicious MCP server | opt-in adapter, allowlisted discovery, and documented policy-wrapping boundary | the reference workflows do not auto-wire MCP; its process needs OS isolation |
| denial of service | budgets, timeouts, bounded queues/output | local resource exhaustion remains possible |

## Reporting a vulnerability

Do not open a public issue for an exploitable vulnerability. Use GitHub's private security
advisory workflow for this repository and include a minimal reproducer, impact, and affected
version.
