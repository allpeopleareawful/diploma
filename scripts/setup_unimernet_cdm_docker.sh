#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-external/UniMERNet}"
IMAGE_NAME="${2:-unimernet-cdm:latest}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required" >&2
  exit 1
fi
docker info >/dev/null

if [[ ! -d "${TARGET_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${TARGET_DIR}")"
  git clone \
    --depth 1 \
    --filter=blob:none \
    --sparse \
    https://github.com/opendatalab/UniMERNet.git \
    "${TARGET_DIR}"
  git -C "${TARGET_DIR}" sparse-checkout set cdm
fi

test -f "${TARGET_DIR}/cdm/evaluation.py"
docker build \
  -f "${TARGET_DIR}/cdm/DockerFile" \
  -t "${IMAGE_NAME}" \
  "${TARGET_DIR}/cdm"

echo "CDM ready"
echo "evaluator: ${TARGET_DIR}/cdm/evaluation.py"
echo "docker image: ${IMAGE_NAME}"
