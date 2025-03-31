#!/bin/bash

# Download Helm
echo "Downloading and installing Helm..."
curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3
chmod +x get_helm.sh
./get_helm.sh
rm get_helm.sh

# Download kubeconform binary
echo "Downloading kubeconform..."
curl -L https://github.com/yannh/kubeconform/releases/latest/download/kubeconform-linux-amd64.tar.gz | tar xvzf -

# Download Datree
echo "Downloading Datree..."
./shell-scripts/install_datree.sh

# Move binaries to bin directory
echo "Moving binaries to bin directory..."
mkdir -p bin
mv kubeconform bin/
mv helm bin/

# Set up Python virtual environment and install dependencies
echo "Setting up Python virtual environment..."
make venv

# Ensure pipeline script is executable
chmod +x pipeline.sh

echo "Setup complete! You can now run the pipeline."
