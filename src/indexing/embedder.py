"""
Embedder: turns chunk text into vectors for semantic search.

Model is configurable (REPO_SENTINEL_EMBEDDING_MODEL, see config.py) rather
than hardcoded, because the right choice differs by environment:

  - Local dev (default): jinaai/jina-embeddings-v2-base-code -- a
    code-specific embedding model, not a general sentence-similarity one.
    Trained on (code, natural-language-description) pairs -- the exact
    cross-modal alignment this project needs. Confirmed empirically in
    this project's own eval harness: a real function was consistently
    unretrievable under a general sentence model (all-MiniLM-L6-v2) until
    switching to this one. Also has an 8192-token context window vs. 256
    for MiniLM, which matters -- a single undocumented 146-line function's
    parameter list alone was found to exceed a 256-token budget, silently
    truncating the model's input before it reached the function's actual
    logic.
  - Deployed (Render free tier, see render.yaml): a smaller model such as
    sentence-transformers/all-MiniLM-L6-v2. Found via real deployment
    testing: Render's free web service has a hard 512MB RAM limit, and
    jina-embeddings-v2-base-code's weights + PyTorch runtime overhead
    exceed that on their own -- the process was OOM-killed (exit 137)
    before the app ever finished starting up. This is a genuine, stated
    tradeoff: the deployed demo has weaker embedding relevance than local
    dev, in exchange for fitting in free-tier memory at all.

trust_remote_code=True is passed unconditionally -- required for
jina-embeddings-v2 (custom ALiBi-based BERT modeling code) and harmless
for standard models like MiniLM that don't have any custom code to trust.

Output dimension depends on the model (768 for jina-code, 384 for
MiniLM) -- if you change REPO_SENTINEL_EMBEDDING_MODEL on an existing
index, you MUST rebuild it from scratch (delete data/index and let
full_index() run again). Mixing embedding spaces of different dimensions
in one ChromaDB collection produces silently meaningless nearest-neighbor
results, not an error.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# huggingface_hub's model download has NO timeout by default -- a stalled
# connection (firewall silently dropping packets, a flaky network) hangs
# the whole process indefinitely with no way to recover short of killing
# it, which is exactly what happened during real testing of this project.
# These must be set before sentence_transformers/huggingface_hub is
# imported, since they're read at import time.
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "15")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import load_settings

logger = logging.getLogger(__name__)

_settings, _, _ = load_settings()

MODEL_NAME = _settings.embedding_model
# Upper bound on sequence length regardless of what the model natively
# supports -- prevents a specific real bug: this project's own chunker can
# produce pathologically long inputs (method_group chunks, which
# concatenate several full methods' source into one string and can run to
# thousands of lines). When a batch mixes one such giant sequence with
# shorter ones, attention memory scales quadratically with the LONGEST
# sequence in the batch, not the average -- this caused an attempted
# 103GB allocation and crashed indexing entirely under jina-code's native
# 8192-token window. Applied as min(model's own native max, this cap) in
# _get_model() below -- NEVER raised above a smaller model's native
# max_seq_length, since forcing a longer sequence than a model's
# positional embeddings support would crash it outright (e.g.
# all-MiniLM-L6-v2's absolute position embeddings top out well below
# this value).
MAX_SEQ_LENGTH_CAP = 2048
ENCODE_BATCH_SIZE = 8  # smaller batches bound worst-case memory when a batch mixes very different sequence lengths

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model %s (first run downloads it)...", MODEL_NAME)
        try:
            _model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        except Exception as e:
            logger.error(
                "Failed to load embedding model %s. This is usually a network "
                "issue reaching huggingface.co (firewall, VPN, or a stalled "
                "connection) rather than a problem with this code. Try: "
                "(1) curl -I https://huggingface.co to check connectivity, "
                "(2) checking firewall/antivirus/VPN settings, "
                "(3) retrying -- partial downloads resume automatically. "
                "Original error: %s", MODEL_NAME, e,
            )
            raise

        native_max = _model.max_seq_length
        effective_max = min(native_max, MAX_SEQ_LENGTH_CAP)
        _model.max_seq_length = effective_max
        logger.info(
            "Embedding model max_seq_length: native=%d, capped=%d "
            "(cap prevents pathologically long chunks from exploding "
            "attention memory; never raises a model above its own native max).",
            native_max, effective_max,
        )
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Returns shape (len(texts), model_dim) --
    dimension depends on which model is configured, see module docstring."""
    model = _get_model()
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True, batch_size=ENCODE_BATCH_SIZE)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([query])[0]


if __name__ == "__main__":
    import time

    texts = [
        "def solve_dependencies(request, dependant): resolve dependency tree",
        "def get_current_user(token): decode jwt and fetch user from db",
        "class Depends: marker used to declare a FastAPI dependency",
    ]
    t0 = time.time()
    vecs = embed_texts(texts)
    print(f"Embedded {len(texts)} chunks in {time.time()-t0:.2f}s, shape={vecs.shape}")

    query_vec = embed_query("how does dependency resolution work")
    sims = vecs @ query_vec / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(query_vec))
    for text, sim in sorted(zip(texts, sims), key=lambda x: -x[1]):
        print(f"  {sim:.3f}  {text[:60]}")
