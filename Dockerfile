FROM ghcr.io/pyo3/maturin:v1.8.3 AS builder

RUN --mount=type=bind,source=.,target=/opt/endpoints/build \
    cd /opt/endpoints/build/hfendpoints && \
    CARGO_TARGET_DIR=/opt/endpoints/dist maturin build -f --release --out /opt/endpoints/dist --features python