FROM ghcr.io/astral-sh/uv:0.13.10-python3.12-trixie-slim AS base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

# The image the eval service runs: no dev dependencies. The configuration is
# piped in on stdin at run time, so editing one does not rebuild the image.
# It also runs scripts/run-eval.py's non-interactive mode and scripts/vm-entry.py,
# which run each run with the Docker CLI and compose plugin through the Docker
# host's socket.
FROM base AS eval
COPY --from=docker:29.10.0-cli /usr/local/bin/docker /usr/local/bin/docker
COPY --from=docker:29.10.0-cli /usr/local/libexec/docker/cli-plugins/docker-compose \
    /usr/local/libexec/docker/cli-plugins/docker-compose
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev
# run-eval.py's dependencies, from its inline script metadata, in their own
# environment, apart from the harness's locked one.
COPY scripts ./scripts
RUN uv venv /opt/run-eval && uv pip install --python /opt/run-eval -r scripts/run-eval.py
COPY compose.yaml ./
# The harness commit recorded in each experiment's metadata.
ARG HARNESS_COMMIT=""
ENV EVAL_HARNESS_COMMIT=$HARNESS_COMMIT

# The image the test service runs: dev dependencies, tests and .env.example.
FROM base AS test
RUN uv sync --locked --no-install-project
COPY src ./src
RUN uv sync --locked
COPY tests ./tests
COPY .env.example ./
