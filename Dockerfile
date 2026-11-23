FROM ghcr.io/astral-sh/uv:0.13.1-python3.12-trixie-slim AS base

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./

# The image the eval service and the batch run: no dev dependencies. The
# configurations are copied last, for the batch to read; a run by hand still
# takes its configuration on stdin.
FROM base AS eval
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev
COPY configs ./configs
# The harness commit recorded in each experiment's metadata.
ARG HARNESS_COMMIT=""
ENV EVAL_HARNESS_COMMIT=$HARNESS_COMMIT

# The image the test service runs: dev dependencies, tests, the configurations
# and the variables .env.example lists for them.
FROM base AS test
RUN uv sync --locked --no-install-project
COPY src ./src
RUN uv sync --locked
COPY tests ./tests
COPY configs ./configs
COPY .env.example ./
