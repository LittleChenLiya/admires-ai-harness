# Trace export

The run inspector can export a run's persisted events as a single bounded JSON
document. The export is a pure read over the event store: it never invokes a
model and never mutates run state.

## API

`GET /api/runs/{run_id}/export?limit={limit}`

- `limit` (optional, default 1000, maximum 5000) caps the number of events in
  the document. When a run has more events than the limit, the first `limit`
  events (in sequence order) are returned and `truncated` is `true`.
- Success returns `200` with `Content-Disposition: attachment` so browsers
  download the document directly. The console's run inspector exposes this as
  an **Export** button.

## Document shape

The document is versioned with `schema_version: "1"` and
`kind: "admires-harness/trace-export"`:

```json
{
  "schema_version": "1",
  "kind": "admires-harness/trace-export",
  "generated_at": "2026-09-20T00:00:00+00:00",
  "run": {
    "id": "...",
    "demo": "research",
    "status": "succeeded",
    "created_at": "...",
    "updated_at": "...",
    "error": null
  },
  "redaction": {
    "placeholder": "[redacted]",
    "strategy": "exact sensitive key match plus secret pattern scan, applied recursively"
  },
  "event_count": 12,
  "total_events": 12,
  "truncated": false,
  "events": ["..."]
}
```

`events` preserves the persisted sequence order and uses the same snake_case
envelope as `GET /api/runs/{run_id}/events` (the versioned `RunEvent`
contract), so an export round-trips with the live event feed.

## Redaction policy

Event payloads may contain tool arguments or headers with credentials. Every
event payload is redacted recursively (including nested objects and lists)
before serialization:

- Dict keys whose normalized name (lowercased, `-`/space mapped to `_`)
  exactly matches a sensitive name — `api_key`, `token`, `secret`,
  `password`, `authorization`, `cookie`, `private_key`, and similar —
  have their entire value replaced with `[redacted]`. Exact matching keeps
  operational fields such as `input_tokens` intact.
- String values are scanned for common secret shapes (`Bearer ...` headers,
  `sk-...` API keys, PEM private-key blocks); matches are replaced with
  `[redacted]`.

Only payloads are redacted; envelope fields (`event_id`, `trace_id`,
timestamps, sequences) are preserved so exports remain joinable with the
event log.

## Behavior for edge cases

- **Missing run / empty run**: a run with no persisted events is
  indistinguishable from a run that never existed; both return `404`.
- **Failed run**: exports normally. `run.status` is `"failed"`,
  `run.error` carries the failure message, and the `run.failed` event is
  present in `events`.
- **Invalid `limit`**: values outside `1..5000` return `422`.
