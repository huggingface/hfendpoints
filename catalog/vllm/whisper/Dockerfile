FROM vllm/vllm-openai:v0.8.3

RUN --mount=type=bind,from=huggingface/endpoints-sdk:v1.0.0-beta-py312-manylinux,source=/opt/endpoints/dist,target=/opt/endpoints/dist \
    --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
    python3 -m pip install -r /tmp/requirements.txt && \
    python3 -m pip install /opt/endpoints/dist/*.whl

COPY endpoint.py /opt/endpoints/

EXPOSE 8000

ENTRYPOINT ["python3"]
CMD ["/opt/endpoints/endpoint.py"]
