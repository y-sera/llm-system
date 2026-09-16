#!/bin/bash

NETWORK_INTERFACE="${NETWORK_INTERFACE:-enP8p1s0}"
IP_ADDRESS="$(ip -4 -j addr show "${NETWORK_INTERFACE}" | jq '.[].addr_info.[].local' -r)"
curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644 --node-ip "${IP_ADDRESS}" --node-external-ip "${IP_ADDRESS}"
