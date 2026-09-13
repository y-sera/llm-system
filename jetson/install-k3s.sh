#!/bin/bash

curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644 --node-ip 192.168.1.96 --node-external-ip 192.168.1.96
