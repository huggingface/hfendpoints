FROM ghcr.io/pyo3/maturin:v1.8.3 AS builder

RUN --mount=type=bind,source=.,target=/opt/hfendpoints/build \
    cd /opt/hfendpoints/build/hfendpoints && \
    CARGO_TARGET_DIR=/opt/hfendpoints/dist maturin build -f --release --out /opt/hfendpoints/dist --features python