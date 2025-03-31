#!/usr/bin/env bash
set -euo pipefail

OUTDIR="."

DIFFUSION_MODELS_ZIP="https://www.dropbox.com/scl/fo/daah2lixb3digjcp5i7ti/AIJ3po1nKmUSPKTBJvJn5qs?rlkey=wsq4b705kx8qi0sxq5j9ipzmz&st=pay2cq62&dl=1"
DIFFUSION_QL_MODELS_ZIP="https://www.dropbox.com/scl/fo/tmrnoz8sticj660oj5v2i/APZdOkl6FX-gXjLSeSxVG9s?rlkey=488l482qmr5kq4y5776j1k2qz&st=xkvfv1nq&dl=1"

DATASET_DUAL_ZIP="https://www.dropbox.com/scl/fi/w9o3c05ndyeavbiu3r1vu/dual_agent.zip?rlkey=u0bqnedvzlvwta8xfpzih382k&st=lzg49q8u&dl=1"
DATASET_SINGLE_ZIP="https://www.dropbox.com/scl/fi/e2mnzqsrh9wf96bhbthb7/single_agent.zip?rlkey=a8uf0gukb04te46zu4164o196&st=hpa9rwh0&dl=1"
DATASET_QL_DUAL_ZIP="https://www.dropbox.com/scl/fi/l3601jzsouu1pl3c6lv5w/ql_dual_agent.zip?rlkey=tgra17py4fh2qthj5m7tpcedh&st=qqq1jwex&dl=1"
DATASET_QL_SINGLE_ZIP="https://www.dropbox.com/scl/fi/i71hnenw21oxc9uqpbcy7/ql_single_agent.zip?rlkey=uxnewd77fl9tsfpj1o9sbh1ed&st=wonrl3nj&dl=1"

BENCHMARK_TASKS_TAR="https://multiarm.cs.columbia.edu/downloads/data/benchmark.tar.xz"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Error: '$1' not found. Please install it."; exit 127; }
}

usage() {
  cat <<'USAGE'
Fetch DG-MAP assets (models + datasets).

Usage:
  scripts/fetch_assets.sh all|models|datasets|benchmarks [--outdir PATH] [--list]

Options:
  --outdir PATH   Destination root (default: external)
  --list          Print what would be fetched and exit
  -h, --help      Show this help

USAGE
  exit 2
}

parse_args() {
  [[ $# -lt 1 ]] && usage
  MODE="$1"; shift
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --outdir) OUTDIR="$2"; shift 2;;
      --list) LIST_ONLY=1; shift;;
      -h|--help) usage;;
      *) echo "Unknown arg: $1"; usage;;
    esac
  done
}

download() {
  local url="$1"
  local out="$2"
  echo "→ Downloading: $url"
  # Prefer curl; fallback to wget
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 --retry-delay 2 -f -o "$out" "$url"
  else
    wget -O "$out" --tries=5 --waitretry=2 "$url"
  fi
}

unzip_into() {
  local zipfile="$1"
  local dest="$2"
  mkdir -p "$dest"
  echo "→ Extracting: $zipfile → $dest"
  unzip -o -q "$zipfile" -x / -d "$dest"
}

fetch_models() {
  local root="${OUTDIR}/application/runs"
  local d1="${root}/plain_diffusion"
  local d2="${root}/diffusion_ql"
  mkdir -p "$d1" "$d2"

  local z1="${root}/diffusion_models.zip"
  local z2="${root}/diffusion_ql_models.zip"

  download "$DIFFUSION_MODELS_ZIP" "$z1"
  download "$DIFFUSION_QL_MODELS_ZIP" "$z2"

  unzip_into "$z1" "$d1"
  unzip_into "$z2" "$d2"

  rm -f "$z1" "$z2"
  echo "Models fetched to: $root"
}

fetch_datasets() {
  local root="${OUTDIR}/datasets"
  mkdir -p "$root"

  local z_s="${root}/single_agent.zip"
  local z_d="${root}/dual_agent.zip"
  local z_qs="${root}/ql_single_agent.zip"
  local z_qd="${root}/ql_dual_agent.zip"

  download "$DATASET_SINGLE_ZIP" "$z_s"
  download "$DATASET_DUAL_ZIP" "$z_d"
  download "$DATASET_QL_SINGLE_ZIP" "$z_qs"
  download "$DATASET_QL_DUAL_ZIP" "$z_qd"

  unzip_into "$z_s" "$root"
  unzip_into "$z_d" "$root"
  unzip_into "$z_qs" "$root"
  unzip_into "$z_qd" "$root"

  rm -f "$z_s" "$z_d" "$z_qs" "$z_qd"
  echo "Datasets fetched to: $root"
}

fetch_benchmarks() {
  local root="${OUTDIR}/application/tasks"
  local archive="${root}/benchmark.tar.xz"
  mkdir -p "$root"

  if [[ -d "${root}/benchmark" ]]; then
    echo "Removing existing benchmark directory at ${root}/benchmark"
    rm -rf "${root}/benchmark"
  fi

  download "$BENCHMARK_TASKS_TAR" "$archive"
  echo "→ Extracting: $archive → $root"
  tar -xJf "$archive" -C "$root"

  rm -f "$archive"
  echo "Benchmark tasks fetched to: ${root}/benchmark"
}

main() {
  if command -v curl >/dev/null 2>&1; then :; elif command -v wget >/dev/null 2>&1; then :; else
    echo "Error: need either 'curl' or 'wget' to download." ; exit 127
  fi

  LIST_ONLY=0
  parse_args "$@"

  echo "Output directory: ${OUTDIR}"
  case "${MODE}" in
    all)
      need_cmd unzip
      need_cmd tar
      [[ "$LIST_ONLY" -eq 1 ]] && { echo "Would fetch: models + datasets + benchmarks"; exit 0; }
      fetch_models
      fetch_datasets
      fetch_benchmarks
      ;;
    models)
      need_cmd unzip
      [[ "$LIST_ONLY" -eq 1 ]] && { echo "Would fetch: models"; exit 0; }
      fetch_models
      ;;
    datasets)
      need_cmd unzip
      [[ "$LIST_ONLY" -eq 1 ]] && { echo "Would fetch: datasets"; exit 0; }
      fetch_datasets
      ;;
    benchmarks)
      need_cmd tar
      [[ "$LIST_ONLY" -eq 1 ]] && { echo "Would fetch: benchmarks"; exit 0; }
      fetch_benchmarks
      ;;
    *)
      echo "Unknown mode: ${MODE} (expected: all|models|datasets|benchmarks)"; exit 2;;
  esac

  echo "All done."
}

main "$@"
