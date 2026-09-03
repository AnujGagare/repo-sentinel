from unittest.mock import MagicMock, patch

from indexing import embedder


def test_large_native_max_seq_length_gets_capped_down():
    """Regression test for a real bug found in production testing: a batch
    containing one of this project's own pathologically long method_group
    chunks (see code_chunker.py -- concatenates several full methods'
    source into one string) caused an attempted 103GB memory allocation
    during attention computation, because a model with a large native
    context (e.g. jina-code's 8192 tokens) was being used uncapped."""
    embedder._model = None

    fake_model = MagicMock()
    fake_model.max_seq_length = 8192  # e.g. jina-embeddings-v2-base-code's native max
    with patch("indexing.embedder.SentenceTransformer", return_value=fake_model) as mock_cls:
        result = embedder._get_model()

    mock_cls.assert_called_once_with(embedder.MODEL_NAME, trust_remote_code=True)
    assert result.max_seq_length == embedder.MAX_SEQ_LENGTH_CAP
    assert embedder.MAX_SEQ_LENGTH_CAP < 8192, "cap must be meaningfully below a large model's rated max to have any effect"

    embedder._model = None


def test_small_native_max_seq_length_is_never_raised():
    """A model whose native context is SMALLER than our cap (e.g.
    all-MiniLM-L6-v2's 256 tokens, used for memory-constrained deployments
    -- see render.yaml) must never be raised above its own native max.
    Forcing a longer sequence than a model's positional embeddings support
    would crash it outright, not just waste memory."""
    embedder._model = None

    fake_model = MagicMock()
    fake_model.max_seq_length = 256  # e.g. all-MiniLM-L6-v2's native max
    with patch("indexing.embedder.SentenceTransformer", return_value=fake_model):
        result = embedder._get_model()

    assert result.max_seq_length == 256, "must stay at the model's own native max, never be raised"

    embedder._model = None


def test_embed_texts_uses_bounded_batch_size():
    """Smaller batch sizes bound worst-case memory when a batch mixes very
    different sequence lengths -- verifies encode() is actually called
    with the configured batch size, not left at sentence-transformers'
    default of 32."""
    embedder._model = None
    fake_model = MagicMock()
    fake_model.max_seq_length = 512
    fake_model.encode.return_value = "fake_embeddings"

    with patch("indexing.embedder.SentenceTransformer", return_value=fake_model):
        embedder.embed_texts(["some text"])

    _, kwargs = fake_model.encode.call_args
    assert kwargs["batch_size"] == embedder.ENCODE_BATCH_SIZE

    embedder._model = None
