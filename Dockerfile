FROM ghcr.io/pyo3/maturin:v1.8.3 AS builder

COPY . /opt/endpoints/build
RUN cd /opt/endpoints/build/hfendpoints && \
    CARGO_TARGET_DIR=/usr/local/hfendpoints/dist maturin build -f --release --out /usr/local/hfendpoints/dist --features python