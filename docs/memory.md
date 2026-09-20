# Memory and offline retrieval

The memory stack (`src/admires_harness/memory/`) works fully offline: no model download,
no API key. `HashingEmbedder` provides deterministic feature-hashing vectors,
`TextChunker` splits documents, `SQLiteMemoryStore` persists namespaced records, and
`RAGRetriever` ties ingestion, retrieval, and prompt rendering together behind the
`MemoryStore` protocol so production adapters can replace any piece.

## Chunking invariants

`TextChunker.split` guarantees, for any Unicode input (CJK, emoji, mixed scripts):

- Whitespace is normalized first (`" ".join(text.split())`); empty input yields no chunks.
- Every non-separator character of the normalized text appears in at least one chunk,
  in order; only single separator spaces at chunk seams may be dropped.
- No chunk exceeds `chunk_size` characters. Words are kept whole when a space boundary
  exists inside the window; spaceless (overlong) tokens are hard-cut at `chunk_size`.
- Overlap never runs away: when no usable word boundary exists, the next chunk starts
  at least `chunk_size - overlap` characters forward, so duplication stays bounded by
  the configured `chunk_size / (chunk_size - overlap)` ratio even for overlap ratios
  close to 1.

## Store consistency guarantees

`SQLiteMemoryStore` keeps behavior identical across concurrent writers and restarts:

- Namespaces are mandatory and isolated; the primary key is `(namespace, id)`.
- The first record in a namespace fixes its embedding dimension. Mismatched `put` calls
  fail closed with `ValueError`, including under concurrent writes (serialized by a
  write lock plus `BEGIN IMMEDIATE`) and after reopening the database.
- Candidate selection is deterministic: `ORDER BY updated_at DESC, id ASC` with a
  configurable `max_candidates` cap, so repeated searches and reopened stores return
  the same hits in the same order.
- Hits are sorted by `(-score, id)`, so tied scores (including zero vectors, which
  score exactly 0.0) have a stable, reproducible order.
- Corrupt or anomalous rows (broken JSON, invalid ids, ragged embeddings) are skipped
  during search instead of failing the whole query; scores are clamped to `[0, 1]` so
  floating-point noise cannot reject a perfect match.
- Query embeddings must be non-empty and finite; `limit` and `min_score` are validated
  before any I/O.

## Prompt rendering

`RAGRetriever.render_context` wraps excerpts in a `trust="untrusted"` envelope and
HTML-escapes sources and content, so markup and prompt-injection text in retrieved
records stays inert data. `max_characters` truncates whole excerpts, never mid-tag.

See `tests/unit/test_memory.py` for the deterministic evidence behind each guarantee.

