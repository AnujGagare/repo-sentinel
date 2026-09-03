"""
Embedder: turns chunk text into vectors for semantic search.

Model: jinaai/jina-embeddings-v2-base-code -- a code-specific embedding
model, not a general sentence-similarity model. This matters and isn't a
cosmetic choice:

  - It's trained on (code, natural-language-description) pairs -- the
    exact cross-modal alignment this project needs ("how does X resolve
    dependencies" <-> the Python code that does it). A general sentence
    model like all-MiniLM-L6-v2 was never trained on that alignment task
    and is measurably weaker at it -- confirmed empirically in this
    project's own eval harness (see README "Embedding model" section):
    a real function was consistently unretrievable for a plainly-worded
    question until we either added a natural-language summary OR (this
    change) switched to a model actually trained to understand code.
  - It supports an 8192-token context window vs. 256 for MiniLM. Verified
    in this project that a single undocumented 146-line function's
    parameter list alone can exceed a 256-token budget, silently
    truncating the model's input before it ever reaches the function's
    actual logic. At 8192 tokens this is a non-issue for any function
    this project is likely to encounter.

Requires trust_remote_code=True -- jina-embeddings-v2 uses a custom
ALiBi-based BERT variant with its own modeling code, not a stock
transformers architecture. This is expected and safe for this specific,
well-known model; sentence-transformers will otherwise raise an explicit
error asking for it rather than silently misbehaving.

Output dimension is 768 (not 384, as with MiniLM). If you're upgrading an
existing index built with a prior embedder version, the vector store MUST
be rebuilt from scratch (delete data/index and let full_index() run
again) -- mixing embedding spaces in one ChromaDB collection silently
produces meaningless nearest-neighbor results, not an error.
"""

from __future__ import annotations

import logging
import os

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

logger = logging.getLogger(__name__)

MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 2048
ENCODE_BATCH_SIZE = 8  # smaller batches bound worst-case memory when a batch mixes very different sequence lengths

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model %s (first run downloads ~640MB)...", MODEL_NAME)
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

        # Cap effective sequence length well below the model's rated 8192
        # max. Real bug found in production testing of this project: this
        # project's own chunker can produce pathologically long inputs --
        # specifically the method_group chunks (see code_chunker.py), which
        # concatenate several full methods' source into one string and can
        # run to thousands of lines. When a batch mixes one such giant
        # sequence with shorter ones, attention memory scales quadratically
        # with the LONGEST sequence in the batch, not the average -- this
        # caused an attempted 103GB allocation and crashed indexing
        # entirely. 2048 tokens comfortably covers any realistic single
        # function while making pathological group-chunk outliers get
        # truncated (silently losing the tail of the least-realistic
        # inputs) instead of blowing up memory for the whole batch.
        _model.max_seq_length = MAX_SEQ_LENGTH
        logger.info("Capped embedding model max_seq_length to %d (model's rated max is 8192) "
                    "to prevent pathologically long chunks from exploding attention memory.",
                    MAX_SEQ_LENGTH)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Returns shape (len(texts), 768)."""
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
    assert vecs.shape[1] == EMBEDDING_DIM, f"expected {EMBEDDING_DIM}-dim vectors, got {vecs.shape[1]}"

    query_vec = embed_query("how does dependency resolution work")
    sims = vecs @ query_vec / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(query_vec))
    for text, sim in sorted(zip(texts, sims), key=lambda x: -x[1]):
        print(f"  {sim:.3f}  {text[:60]}")
