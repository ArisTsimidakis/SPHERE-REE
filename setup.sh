#!/bin/bash

# Download and install Helm
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod +x get_helm.sh
./get_helm.sh

# Download Checkov
pip3 install checkov

# Download kubeconform binary
curl -L https://github.com/yannh/kubeconform/releases/latest/download/kubeconform-linux-amd64.tar.gz | tar xvzf -

# Set up Python virtual environment and install dependencies
make venv
