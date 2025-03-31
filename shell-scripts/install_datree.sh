#!/bin/bash

OWNER="datreeio"
REPO="datree"

API_URL="https://api.github.com/repos/$OWNER/$REPO/releases/latest"
RELEASE_JSON=$(curl -s "$API_URL")

# Detect OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

# Map OS and ARCH to GitHub naming conventions
case "$OS" in
    Linux) OS_NAME="Linux" ;;
    Darwin) OS_NAME="Darwin" ;;
    CYGWIN*|MINGW*|MSYS*) OS_NAME="windows" ;;
    *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64) ARCH_NAME="x86_64" ;;
    arm64|aarch64) ARCH_NAME="arm64" ;;
    i386|i686) ARCH_NAME="386" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

# Find correct download URL
DOWNLOAD_URL=$(echo "$RELEASE_JSON" | jq -r ".assets[] | select(.name | test(\"${OS_NAME}_${ARCH_NAME}.zip\")) | .browser_download_url")

if [[ -z "$DOWNLOAD_URL" ]]; then
    echo "No matching binary found for $OS_NAME $ARCH_NAME"
    exit 1
fi

echo "Downloading: $DOWNLOAD_URL"
curl -LO "$DOWNLOAD_URL"

# Extract filename from URL
FILENAME=$(basename "$DOWNLOAD_URL")
echo "Downloaded: $FILENAME"

# Extract the downloaded file
unzip "$FILENAME" "datree"
echo "Installation complete."
echo "Extracted: $FILENAME"

# Clean up
rm "$FILENAME"
echo "Removed downloaded file: $FILENAME"

# Move the datree binary to the bin directory]
mkdir -p bin
mv datree bin/
echo "Moved datree binary to bin directory."
echo "Installation of Datree complete."