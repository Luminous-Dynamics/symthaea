# Build from the repository root:
# docker compose -f deploy/aml/compose.yaml build
FROM rust:1.96.0-bookworm AS builder

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev libclang-dev cmake \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN cargo build --locked --release \
    --no-default-features \
    -p symthaea-broca \
    --features aml-http-server \
    --bin broca-aml-server

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl libgcc-s1 libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 broca \
    && useradd --uid 10001 --gid 10001 \
        --home-dir /var/lib/symthaea-aml \
        --no-create-home --shell /usr/sbin/nologin broca \
    && install -d -o broca -g broca /var/lib/symthaea-aml

COPY --from=builder /workspace/target/release/broca-aml-server /usr/local/bin/broca-aml-server

ENV AML_BIND=0.0.0.0:8080 \
    AML_DB_PATH=/var/lib/symthaea-aml/aml.sqlite \
    AML_PAYLOAD_ROOT=/var/lib/symthaea-aml/payloads

WORKDIR /var/lib/symthaea-aml
USER 10001:10001
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/broca-aml-server"]
