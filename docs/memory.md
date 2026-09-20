# Memory and offline retrieval

The memory stack (`src/admires_harness/memory/`) works entirely offline: the default
embedder is a deterministic feature-hashing baseline with no model download and no API
key, and the default store is a local SQLite file. Production embedding providers and
vector stores can replace either half through the `Embedder` and `MemoryStore`
protocols without touching call sites.

## Chunking guarantees (`TextChunker`)

- Whitespace is normalized (`" ".join(text.split())`) before chunking, so runs of
  spaces, tabs, newlines, and ideographic spaces collapse to single separators.
- Every non-whitespace character of the normalized text appears in at least one
  chunk; chunking never drops content.
- Each chunk is at most `chunk_size` characters. Word boundaries are preferred, but
  tokens longer than `chunk_size` (for example long spaceless strings) are hard-cut.
- Consecutive chunks overlap by at most `overlap` characters and the start position
  strictly advances, so duplication stays bounded and splitting always terminates.

## Embedding guarantees (`HashingEmbedder`)

- Tokenization is Unicode-aware: CJK ideographs become per-character tokens, and
  emoji or other symbols become single tokens instead of vanishing, so Chinese and
  emoji-only text produce meaningful, non-degenerate vectors.
- Vectors are L2-normalized; empty or tokenless text yields the zero vector, which
  scores 0.0 against everything.
- The same text always hashes to the same vector across processes and restarts.

## Store consistency (`SQLiteMemoryStore`)

- Namespaces are mandatory and isolate reads, writes, and deletes.
- The first record in a namespace fixes its embedding dimension; conflicting writes
  fail with `ValueError`, including under concurrent writers (serialized with
  `BEGIN IMMEDIATE`). Other namespaces may use different dimensions.
- `max_candidates` is persisted in the `memory_meta` table on first initialization
  and adopted when the database is reopened, so the candidate cap is consistent
  across restarts regardless of constructor arguments.
- Candidate selection orders by `updated_at DESC, id ASC`, and final ranking orders
  by `(-score, id)`, so tied scores and identical timestamps return in a stable,
  deterministic order.
- Corrupt rows (invalid JSON, mismatched stored dimensions) are skipped during
  search instead of failing the whole query; zero vectors score 0.0.
- Metadata filters require exact equality on every supplied key; records missing a
  key never match.

## Prompt safety (`RAGRetriever.render_context`)

Retrieved excerpts are rendered inside a `trust="untrusted"` envelope with
HTML-escaped content and sources, so embedded markup or prompt-injection text is
passed to the model as inert data. `max_characters` truncates whole excerpts and
never splits the envelope.

See `tests/unit/test_memory.py` for the deterministic witnesses of these guarantees.

