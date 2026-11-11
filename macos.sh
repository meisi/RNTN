#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
case "${1:-help}" in
  build)
    docker build --platform linux/amd64 -t hupkes-local .
    ;;
  smoke)
    docker run --rm --platform linux/amd64 -v "$PWD:/work" -w /work hupkes-local bash smoke.sh
    ;;
  train)
    docker run --rm -it --platform linux/amd64 -v "$PWD:/work" -w /work hupkes-local python train.py
    ;;
  resume)
    if [ "$#" -ne 2 ]; then
      echo 'Usage: bash macos.sh resume training/hupkes-transformer-seed1/model_step_NUMBER.pt'
      exit 1
    fi
    docker run --rm -it --platform linux/amd64 -v "$PWD:/work" -w /work hupkes-local python train.py --resume "$2"
    ;;
  *)
    echo 'Usage: bash macos.sh build|smoke|train|resume'
    ;;
esac
