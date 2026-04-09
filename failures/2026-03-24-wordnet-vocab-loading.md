# Failure: WordNet Vocab Loading — Memory Spike

**Date:** 2026-03-24
**Status:** resolved

---

`EmbeddingPerturbationFamily` tried to build synonym directions from WordNet for the entire tokenizer vocabulary (50k+ tokens). This caused a massive memory spike at module import time, and silently fell back to random orthogonal directions for almost all tokens without telling anyone.

So the "synonym directions" were mostly random. Not ideal.

Fixed with a `vocab_size` cap (default 10k), explicit logging when falling back, and a `max_synonyms_per_token` parameter. Should have thought about this earlier — 50k WordNet lookups at import is obviously going to be a problem.
