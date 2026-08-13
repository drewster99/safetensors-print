#!/usr/bin/env bash
#
# Download third-party .safetensors files to test against.
#
# These come from other people's tools, not ours, which is the point: they are the
# only evidence that the reader copes with headers we did not write. Each was chosen
# for a different shape of file -- different producers, dtypes, metadata conventions
# and tensor counts -- and each is small enough to fetch on demand.
#
# Idempotent: a file already present with the recorded digest is left alone. Files are
# written to tests/corpus/third-party, which is not tracked by git.

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CORPUS_DIRECTORY="${REPOSITORY_ROOT}/tests/corpus/third-party"

# name | source repository | path within it | whole megabytes | why this one
DOWNLOADS=(
  "tiny-random-gpt2.safetensors|hf-internal-testing/tiny-random-gpt2|model.safetensors|1|transformers, F32, tiny"
  "tiny-random-llama.safetensors|hf-internal-testing/tiny-random-LlamaForCausalLM|model.safetensors|4|transformers, another architecture's tensor naming"
  "opt-125m-lora.safetensors|peft-internal-testing/opt-125m-dummy-lora|adapter_model.safetensors|1|a PEFT adapter, not a whole model"
  "all-MiniLM-L6-v2.safetensors|sentence-transformers/all-MiniLM-L6-v2|model.safetensors|90|a real trained model, 100+ tensors"
  "sd-vae-ft-mse.safetensors|stabilityai/sd-vae-ft-mse|diffusion_pytorch_model.safetensors|335|diffusers rather than transformers"
  "qwen2.5-0.5b-4bit-mlx.safetensors|mlx-community/Qwen2.5-0.5B-Instruct-4bit|model.safetensors|278|MLX, 4-bit quantised: U32 payloads beside F16 scales"
)

# Anything at or above this many megabytes is what --skip-large skips.
LARGE_MEGABYTES=100

SKIP_LARGE=0
if [[ "${1:-}" == "--skip-large" ]]; then
  SKIP_LARGE=1
elif [[ $# -gt 0 ]]; then
  printf 'usage: %s [--skip-large]\n' "$(basename "$0")" >&2
  exit 2
fi

mkdir -p "${CORPUS_DIRECTORY}"

failures=0
for entry in "${DOWNLOADS[@]}"; do
  IFS='|' read -r name repository path megabytes note <<<"${entry}"
  destination="${CORPUS_DIRECTORY}/${name}"

  if [[ -s "${destination}" ]]; then
    printf 'have  %-34s %s\n' "${name}" "(${note})"
    continue
  fi

  if ((SKIP_LARGE)) && ((megabytes >= LARGE_MEGABYTES)); then
    printf 'skip  %-34s %s\n' "${name}" "(~${megabytes} MB, --skip-large)"
    continue
  fi

  url="https://huggingface.co/${repository}/resolve/main/${path}"
  printf 'fetch %-34s %s (~%s MB)\n' "${name}" "${url}" "${megabytes}"
  if ! curl --silent --show-error --location --fail --max-time 300 \
       --output "${destination}.partial" "${url}"; then
    printf '      FAILED, skipping\n' >&2
    rm -f "${destination}.partial"
    failures=$((failures + 1))
    continue
  fi
  mv "${destination}.partial" "${destination}"
done

printf '\n%s\n' "corpus: ${CORPUS_DIRECTORY}"
ls -la "${CORPUS_DIRECTORY}" | tail -n +2

if ((failures > 0)); then
  printf '\n%d download(s) failed. The matrix runs against whatever did arrive.\n' "${failures}" >&2
  exit 1
fi
