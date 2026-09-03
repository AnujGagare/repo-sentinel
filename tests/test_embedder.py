from unittest.mock import MagicMock, patch

from indexing import embedder


def test_max_seq_length_is_capped_below_model_rated_maximum():
    """Regression test for a real bug found in production testing: a batch
    containing one of this project's own pathologically long method_group
    chunks (see code_chunker.py -- concatenates several full methods'
    source into one string) caused an attempted 103GB memory allocation
    during attention computation, because the embedding model's rated max
    context (8192 tokens) was being used uncapped. Verifies the fix
    (explicitly capping max_seq_length after model load) is actually
    applied, without needing to download the real ~640MB model."""
    embedder._model = None  # reset any cached instance from other tests

    fake_model = MagicMock()
    with patch("indexing.embedder.SentenceTransformer", return_value=fake_model) as mock_cls:
        result = embedder._get_model()

    mock_cls.assert_called_once_with(embedder.MODEL_NAME, trust_remote_code=True)
    assert result.max_seq_length == embedder.MAX_SEQ_LENGTH
    assert embedder.MAX_SEQ_LENGTH < 8192, "cap must be meaningfully below the model's rated max to have any effect"

    embedder._model = None  # don't leak the mock into other tests


def test_embed_texts_uses_bounded_batch_size():
    """Smaller batch sizes bound worst-case memory when a batch mixes very
    different sequence lengths -- verifies encode() is actually called
    with the configured batch size, not left at sentence-transformers'
    default of 32."""
    embedder._model = None
    fake_model = MagicMock()
    fake_model.encode.return_value = "fake_embeddings"

    with patch("indexing.embedder.SentenceTransformer", return_value=fake_model):
        embedder.embed_texts(["some text"])

    _, kwargs = fake_model.encode.call_args
    assert kwargs["batch_size"] == embedder.ENCODE_BATCH_SIZE

    embedder._model = None
