#!/bin/bash
set -e

# Change directory to the location of this script
cd "$(dirname "$0")"

# Get the first 7 characters of the current git hash
GIT_HASH=$(git rev-parse --short=7 HEAD || echo "unknown")
# Get the current build timestamp
BUILD_TIME=$(date +"%Y-%m-%d %H:%M:%S")

# Write to version.txt (gitignored)
echo "hash=$GIT_HASH" > version.txt
echo "timestamp=$BUILD_TIME" >> version.txt

echo "Generated version.txt:"
cat version.txt
echo ""

LOCAL_IMAGE="esphome-doc-mcp"
REGISTRY_IMAGE="registry.shifamily.com/homestack/esphome-doc-mcp"

echo "========================================="
echo " BUILDING DOCKER IMAGE                   "
echo "========================================="
docker build -t ${LOCAL_IMAGE}:latest -t ${LOCAL_IMAGE}:${GIT_HASH} .

echo "========================================="
echo " TAGGING FOR REGISTRY                     "
echo "========================================="
docker tag ${LOCAL_IMAGE}:latest ${REGISTRY_IMAGE}:latest
docker tag ${LOCAL_IMAGE}:${GIT_HASH} ${REGISTRY_IMAGE}:${GIT_HASH}

echo "========================================="
echo " PUSHING TO REGISTRY                     "
echo "========================================="
docker push ${REGISTRY_IMAGE}:latest
docker push ${REGISTRY_IMAGE}:${GIT_HASH}

echo "========================================="
echo " BUILD AND PUSH SUCCESSFUL!              "
echo "========================================="
