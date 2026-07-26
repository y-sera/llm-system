#!/bin/bash

kubectl create -n llm-system secret generic hf-token-secret --from-file=token=$HOME/hf_token_$(hostname)
