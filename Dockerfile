# Repo Sentinel API container.
#
# Note on embedding model: jina-embeddings-v2-base-code (~640MB) downloads
# on first startup and is cached in the image's filesystem layer if you
# bake indexing into the build, or on first request otherwise. For a
# faster cold start in production, consider adding a build step that
# pre-warms the model cache -- left as a documented tradeoff rather than
# implemented here, since it meaningfully increases image size and this
# project's actual deployment target (Render free tier) doesn't persist
# a volume between deploys anyway (see render.yaml's comment on this).

FROM python:3.12-slim AS base

# git is required at runtime (GitPython shells out to the real git binary,
# not a pure-Python reimplementation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runs as a non-root user -- a real (if easy to skip) production hardening
# step: if the app is ever compromised via a dependency vulnerability, it
# shouldn't have root inside the container.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
