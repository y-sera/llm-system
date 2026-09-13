#!/bin/bash

kubectl create -n llm-system secret generic hf-token-secret --from-literal=token="$(cat $HOME/.cache/huggingface/token)"
