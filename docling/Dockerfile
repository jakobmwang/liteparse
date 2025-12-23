FROM quay.io/docling-project/docling-serve-cu126:latest

USER 0
RUN pip install --no-cache-dir docling-surya \
 && chown -R 1001:0 /opt/app-root/src/.cache

ENV HF_HOME=/root/.cache/huggingface

USER 1001