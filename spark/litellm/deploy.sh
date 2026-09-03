#!/bin/bash

helm upgrade --install litellm \
  oci://ghcr.io/berriai/litellm/chart/litellm \
  --version 1.93.0 \
  -n llm-system \
  --create-namespace \
  --values values.yaml
