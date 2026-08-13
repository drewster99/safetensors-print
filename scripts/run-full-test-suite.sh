#!/usr/bin/env bash
#
# Everything, in one command: the unit suite, then every command line combination
# against every file in the corpus.
#
#   ./scripts/run-full-test-suite.sh              # uses whatever corpus is present
#   ./scripts/run-full-test-suite.sh --fetch      # downloads the third-party files first
#   ./scripts/run-full-test-suite.sh --fetch --skip-large
#
# The corpus lives in tests/corpus, untracked, in three parts:
#
#   synthetic/    rebuilt each run; the edge cases no real file reaches
#   third-party/  models from other people's tools, fetched on demand
#   local/        anything you drop in yourself, symlinks included
#
# Nothing here depends on files that are only on one machine: a missing corpus part is
# reported and skipped rather than failing the run.

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_DIRECTORY="${REPOSITORY_ROOT}/tests/corpus"
PYTHON="${PYTHON:-${REPOSITORY_ROOT}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

FETCH=0
FETCH_ARGUMENTS=()
for argument in "$@"; do
  case "${argument}" in
    --fetch) FETCH=1 ;;
    --skip-large) FETCH_ARGUMENTS+=("--skip-large") ;;
    *) printf 'usage: %s [--fetch] [--skip-large]\n' "$(basename "$0")" >&2; exit 2 ;;
  esac
done

heading() {
  printf '\n\033[1m%s\033[0m\n' "$1"
}

heading "1. unit suite"
"${PYTHON}" -m pytest "${REPOSITORY_ROOT}/tests" -q

heading "2. synthetic corpus"
"${PYTHON}" "${REPOSITORY_ROOT}/scripts/make-synthetic-corpus.py" | tail -2

if ((FETCH)); then
  heading "3. third-party corpus"
  "${REPOSITORY_ROOT}/scripts/fetch-test-corpus.sh" "${FETCH_ARGUMENTS[@]+"${FETCH_ARGUMENTS[@]}"}" || true
fi

corpus_parts=()
for part in synthetic third-party local; do
  directory="${CORPUS_DIRECTORY}/${part}"
  if [[ -d "${directory}" ]] && compgen -G "${directory}/*.safetensors" >/dev/null; then
    corpus_parts+=("${directory}")
  else
    printf 'corpus part missing, skipping: tests/corpus/%s\n' "${part}"
  fi
done

if ((${#corpus_parts[@]} == 0)); then
  printf 'no corpus to run against. Try --fetch.\n' >&2
  exit 1
fi

heading "4. option matrix"
"${PYTHON}" "${REPOSITORY_ROOT}/scripts/run-option-matrix.py" \
  --command "${PYTHON} -m safetensors_print" \
  "${corpus_parts[@]}"

heading "all green"
