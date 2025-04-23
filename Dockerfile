FROM ghcr.io/pyo3/maturin:v1.8.3 AS builder

RUN --mount=type=bind,source=.,target=/opt/endpoints/build \
    cd /opt/endpoints/build/hfendpoints && \
    CARGO_TARGET_DIR=/usr/local/hfendpoints/dist maturin build -f --release --out /usr/local/hfendpoints/dist --features python && \