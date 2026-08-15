#!/usr/bin/env bash
set -Eeuo pipefail

# Release builder for safetensors-print.
#
# Default behavior:
#   - bumps the patch version in src/safetensors_print/__init__.py
#   - runs the unit suite and the full option matrix against the synthetic corpus
#   - builds the sdist, the wheel, and a standalone single-file zipapp
#   - installs the wheel into a throwaway venv and checks both artifacts actually run,
#     exit codes included, since a package that imports but misbehaves is worse than one
#     that fails to build
#   - writes a Homebrew formula pinned to the sdist's digest
#   - commits the version bump, tags it, pushes, creates and verifies the GitHub release
#
# Publishing to PyPI is not done here: .github/workflows/publish.yml runs on the
# published release, via trusted publishing, so no token lives on this machine.
#
# Examples:
#   ./release.sh                     # 0.1.0 -> 0.1.1, publish
#   ./release.sh --version 0.1.0     # hold the version (first release)
#   ./release.sh --dry-run           # build and verify locally; no commit/tag/push/release
#   ./release.sh --skip-github       # same, but keeps the version bump
#   ./release.sh --yes               # skip the confirmation prompt

PROJECT_NAME="safetensors-print"
DISTRIBUTION_NAME="safetensors_print"
REPO_SLUG="drewster99/safetensors-print"
VERSION_FILE="src/safetensors_print/__init__.py"
TAP_REPOSITORY="drewster99/homebrew-tap"

ROOT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIRECTORY"

BUILD_ROOT="${ROOT_DIRECTORY}/build/release"
PYTHON="${PYTHON:-${ROOT_DIRECTORY}/.venv/bin/python}"

OVERRIDE_VERSION=""
DRY_RUN=0
SKIP_GITHUB=0
YES=0

usage() {
  sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

log() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
success() { printf '\033[32m%s\033[0m\n' "$1"; }
fail() { printf '\033[31mERROR: %s\033[0m\n' "$1" >&2; exit 1; }

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required but not on PATH"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) OVERRIDE_VERSION="${2:-}"; [[ -n "$OVERRIDE_VERSION" ]] || fail "--version needs a value"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-github) SKIP_GITHUB=1; shift ;;
    --yes|-y) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "unknown argument: $1 (try --help)" ;;
  esac
done

log "Checking prerequisites"
require_command git
require_command gh
[[ -x "$PYTHON" ]] || fail "no interpreter at ${PYTHON}; set PYTHON=/path/to/python"
"$PYTHON" -c "import build, twine" 2>/dev/null ||
  fail "build and twine are needed: ${PYTHON} -m pip install -e '.[dev]'"
success "git, gh, and $("$PYTHON" --version)"

current_version() {
  "$PYTHON" - "$VERSION_FILE" <<'PYTHON'
import re, sys
text = open(sys.argv[1]).read()
match = re.search(r'^__version__ = "([^"]+)"$', text, re.MULTILINE)
if not match:
    raise SystemExit("could not find __version__ in " + sys.argv[1])
print(match.group(1))
PYTHON
}

next_patch_version() {
  "$PYTHON" - "$1" <<'PYTHON'
import sys
major, minor, patch = (int(part) for part in sys.argv[1].split("."))
print("{}.{}.{}".format(major, minor, patch + 1))
PYTHON
}

write_version() {
  "$PYTHON" - "$VERSION_FILE" "$1" <<'PYTHON'
import re, sys
path, version = sys.argv[1], sys.argv[2]
text = open(path).read()
updated = re.sub(r'^__version__ = "[^"]+"$', '__version__ = "{}"'.format(version), text, count=1, flags=re.MULTILINE)
if updated == text and '"{}"'.format(version) not in text:
    raise SystemExit("failed to rewrite __version__")
open(path, "w").write(updated)
PYTHON
}

CURRENT_VERSION="$(current_version)"
if [[ -n "$OVERRIDE_VERSION" ]]; then
  NEW_VERSION="$OVERRIDE_VERSION"
else
  NEW_VERSION="$(next_patch_version "$CURRENT_VERSION")"
fi
TAG="v${NEW_VERSION}"
log "Releasing ${PROJECT_NAME} ${NEW_VERSION} (currently ${CURRENT_VERSION})"

if [[ "$DRY_RUN" -eq 0 && "$SKIP_GITHUB" -eq 0 ]]; then
  log "Checking the working tree and the remote"
  branch="$(git rev-parse --abbrev-ref HEAD)"
  [[ "$branch" == "main" ]] || fail "on branch ${branch}; releases are cut from main"
  [[ -z "$(git status --porcelain)" ]] || fail "the working tree has uncommitted changes"
  git fetch --quiet origin
  [[ -z "$(git log --oneline HEAD..origin/main)" ]] || fail "main is behind origin; pull first"
  git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null &&
    fail "local tag ${TAG} already exists"
  git ls-remote --exit-code --tags origin "refs/tags/${TAG}" >/dev/null 2>&1 &&
    fail "remote tag ${TAG} already exists"
  gh release view "$TAG" --repo "$REPO_SLUG" >/dev/null 2>&1 &&
    fail "GitHub release ${TAG} already exists"
  success "clean, current, and ${TAG} is free"
fi

log "Running the unit suite"
"$PYTHON" -m pytest tests -q

log "Running every option combination against the synthetic corpus"
"$PYTHON" scripts/make-synthetic-corpus.py >/dev/null
"$PYTHON" scripts/run-option-matrix.py tests/corpus/synthetic \
  --command "${PYTHON} -m safetensors_print" | tail -1

log "Setting the version to ${NEW_VERSION}"
write_version "$NEW_VERSION"
[[ "$(current_version)" == "$NEW_VERSION" ]] || fail "version rewrite did not take"

log "Building the sdist and the wheel"
rm -rf "$BUILD_ROOT" dist
mkdir -p "$BUILD_ROOT"
"$PYTHON" -m build --outdir "$BUILD_ROOT" >/dev/null
"$PYTHON" -m twine check "$BUILD_ROOT"/*

SDIST="${BUILD_ROOT}/${DISTRIBUTION_NAME}-${NEW_VERSION}.tar.gz"
WHEEL="$(ls "${BUILD_ROOT}/${DISTRIBUTION_NAME}-${NEW_VERSION}"-*.whl)"
[[ -f "$SDIST" ]] || fail "expected an sdist at ${SDIST}"
[[ -f "$WHEEL" ]] || fail "expected a wheel for ${NEW_VERSION}"
success "$(basename "$SDIST") and $(basename "$WHEEL")"

log "Building the standalone zipapp"
# One file that runs on any Python 3.9+, for people who want neither a package manager
# nor a virtualenv. Its own __main__ passes the exit code through, which the module
# zipapp would generate does not.
ZIPAPP_STAGE="${BUILD_ROOT}/zipapp"
ZIPAPP="${BUILD_ROOT}/${PROJECT_NAME}-${NEW_VERSION}.pyz"
rm -rf "$ZIPAPP_STAGE"
mkdir -p "$ZIPAPP_STAGE"
cp -R "src/safetensors_print" "$ZIPAPP_STAGE/"
find "$ZIPAPP_STAGE" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cat > "${ZIPAPP_STAGE}/__main__.py" <<'PYTHON'
import sys

from safetensors_print.cli import main

sys.exit(main())
PYTHON
"$PYTHON" -m zipapp "$ZIPAPP_STAGE" --python "/usr/bin/env python3" --output "$ZIPAPP"
chmod +x "$ZIPAPP"
success "$(basename "$ZIPAPP")"

log "Checking the built artifacts actually run"
VERIFY_VENV="${BUILD_ROOT}/verify-venv"
rm -rf "$VERIFY_VENV"
"$PYTHON" -m venv "$VERIFY_VENV"
"${VERIFY_VENV}/bin/pip" install --quiet "$WHEEL"

SAMPLE="${BUILD_ROOT}/sample.safetensors"
"$PYTHON" - "$SAMPLE" <<'PYTHON'
import json, struct, sys
header = {"__metadata__": {"format": "pt"}, "w": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}
encoded = json.dumps(header).encode()
open(sys.argv[1], "wb").write(struct.pack("<Q", len(encoded)) + encoded + b"\x00")
PYTHON

check_runs() {
  local what="$1"; shift
  local reported
  reported="$("$@" --version)" || fail "${what}: --version failed"
  [[ "$reported" == "${PROJECT_NAME} ${NEW_VERSION}" ]] ||
    fail "${what}: reported '${reported}', expected '${PROJECT_NAME} ${NEW_VERSION}'"
  "$@" "$SAMPLE" >/dev/null || fail "${what}: could not describe a valid file"
  "$@" "$SAMPLE" --metadata | grep -q '"format": "pt"' || fail "${what}: metadata output is wrong"
  # A missing file must still exit 3 rather than crashing: the exit codes are the
  # contract, and an entry point that drops them looks fine until a script depends on it.
  set +e
  "$@" "${BUILD_ROOT}/definitely-absent.safetensors" >/dev/null 2>&1
  local code=$?
  set -e
  [[ "$code" -eq 3 ]] || fail "${what}: a missing file exited ${code}, expected 3"
  success "${what} runs and keeps its exit codes"
}

check_runs "installed wheel" "${VERIFY_VENV}/bin/${PROJECT_NAME}"
check_runs "zipapp" "$ZIPAPP"

log "Writing the Homebrew formula"
SDIST_SHA256="$("$PYTHON" -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$SDIST")"
FORMULA="${BUILD_ROOT}/${PROJECT_NAME}.rb"
cat > "$FORMULA" <<FORMULA_TEXT
class SafetensorsPrint < Formula
  include Language::Python::Virtualenv

  desc "Print the header, metadata and data-segment layout of a .safetensors file"
  homepage "https://github.com/${REPO_SLUG}"
  url "https://github.com/${REPO_SLUG}/releases/download/${TAG}/${DISTRIBUTION_NAME}-${NEW_VERSION}.tar.gz"
  sha256 "${SDIST_SHA256}"
  license "MIT"

  depends_on "python@3.13"

  # No resource blocks: the package has no runtime dependencies.
  def install
    virtualenv_install_with_resources
  end

  test do
    header = '{"w":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    # .b throughout: the packed length is a binary string, and concatenating it with a
    # UTF-8 one raises as soon as the length needs a byte above 127.
    (testpath/"m.safetensors").binwrite([header.bytesize].pack("Q<") + header.b + "\\0".b)

    output = shell_output("#{bin}/${PROJECT_NAME} #{testpath}/m.safetensors")
    assert_match "INTEGRITY", output
    assert_match "conforms to the safetensors specification", output

    # A file that cannot be read must exit 3, not merely print something.
    shell_output("#{bin}/${PROJECT_NAME} #{testpath}/absent.safetensors 2>&1", 3)
  end
end
FORMULA_TEXT
success "$(basename "$FORMULA") (sdist sha256 ${SDIST_SHA256:0:16}...)"

release_notes() {
  local previous_tag notes
  previous_tag="$(git tag --list 'v*' --sort=-v:refname | grep -v "^${TAG}\$" | head -n 1 || true)"
  notes="${BUILD_ROOT}/RELEASE_NOTES_${NEW_VERSION}.md"
  {
    echo "## Install"
    echo
    echo '```sh'
    echo "pip install ${PROJECT_NAME}          # or: uv tool install ${PROJECT_NAME}"
    echo "brew install ${TAP_REPOSITORY%/*}/tap/${PROJECT_NAME}"
    echo '```'
    echo
    echo "Or download \`${PROJECT_NAME}-${NEW_VERSION}.pyz\` below and run it directly:"
    echo "one file, no install, any Python 3.9 or later."
    echo
    echo '```sh'
    echo "chmod +x ${PROJECT_NAME}-${NEW_VERSION}.pyz"
    echo "./${PROJECT_NAME}-${NEW_VERSION}.pyz model.safetensors"
    echo '```'
    echo
    echo "## Changes"
    echo
    if [[ -n "$previous_tag" ]]; then
      git log --pretty='- %s (%h)' "${previous_tag}..HEAD"
    else
      git log --pretty='- %s (%h)' --max-count=20
    fi
  } > "$notes"
  printf '%s' "$notes"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "Dry run: restoring the version and stopping"
  write_version "$CURRENT_VERSION"
  success "Built and verified ${NEW_VERSION} without publishing."
  echo "Artifacts: ${BUILD_ROOT}"
  exit 0
fi

if [[ "$SKIP_GITHUB" -eq 1 ]]; then
  success "Built and verified ${NEW_VERSION}; publishing skipped."
  echo "Artifacts: ${BUILD_ROOT}"
  echo "The version bump in ${VERSION_FILE} is uncommitted."
  exit 0
fi

if [[ "$YES" -eq 0 ]]; then
  printf '\nPublish %s %s to github.com/%s? [y/N] ' "$PROJECT_NAME" "$NEW_VERSION" "$REPO_SLUG"
  read -r answer
  [[ "$answer" =~ ^[Yy]$ ]] || fail "cancelled; artifacts remain in ${BUILD_ROOT}"
fi

log "Committing, tagging and pushing"
git add "$VERSION_FILE"
git commit -m "Release ${NEW_VERSION}"
git tag -a "$TAG" -m "Release ${NEW_VERSION}"
git push origin HEAD:main
git push origin "$TAG"

log "Creating the GitHub release"
NOTES_FILE="$(release_notes)"
gh release create "$TAG" "$SDIST" "$WHEEL" "$ZIPAPP" \
  --repo "$REPO_SLUG" \
  --title "${PROJECT_NAME} ${NEW_VERSION}" \
  --notes-file "$NOTES_FILE"

log "Verifying the release and its assets"
RELEASE_JSON="$(gh release view "$TAG" --repo "$REPO_SLUG" --json tagName,url,assets)"
printf '%s' "$RELEASE_JSON" > "${BUILD_ROOT}/release-${NEW_VERSION}.json"
printf '%s' "$RELEASE_JSON" | grep -q "\"tagName\":\"${TAG}\"" ||
  fail "release verification failed: ${TAG} not in the release JSON"
for asset in "$SDIST" "$WHEEL" "$ZIPAPP"; do
  printf '%s' "$RELEASE_JSON" | grep -q "$(basename "$asset")" ||
    fail "release verification failed: $(basename "$asset") was not attached"
done

success "Published ${PROJECT_NAME} ${NEW_VERSION}"
echo "Release: https://github.com/${REPO_SLUG}/releases/tag/${TAG}"
echo
echo "PyPI:    publish.yml is running now; check with"
echo "           gh run list --repo ${REPO_SLUG} --workflow publish.yml"
echo
echo "Homebrew: the formula is at ${FORMULA}."
echo "          Publish it to the tap with:"
echo "            gh repo create ${TAP_REPOSITORY} --public --clone -d 'Homebrew formulae for Nuclear Cyborg tools'"
echo "            mkdir -p ${TAP_REPOSITORY#*/}/Formula"
echo "            cp ${FORMULA} ${TAP_REPOSITORY#*/}/Formula/"
echo "            cd ${TAP_REPOSITORY#*/} && git add . && git commit -m 'Add ${PROJECT_NAME} ${NEW_VERSION}' && git push"
echo "          Users then run:"
echo "            brew install ${TAP_REPOSITORY%/*}/tap/${PROJECT_NAME}"
