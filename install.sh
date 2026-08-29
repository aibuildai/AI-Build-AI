#!/bin/sh
# AIBuildAI installer.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/aibuildai/AI-Build-AI/main/install.sh | AIBUILDAI_LINE=science sh
#
# Set AIBUILDAI_LINE to the product line you have: science, v2.5, v2.0 or v1.0.
# The script always installs that line's current release. It stays correct when
# the line moves to a new release, so the command never changes.
#
# The script downloads the release into a private temporary directory, checks
# the download against the published sha256, and installs from there. It writes
# nothing into the directory you run it from, and it reads nothing from it.

set -eu

REPO_URL="https://github.com/aibuildai/AI-Build-AI"
LINE="${AIBUILDAI_LINE:-}"
LINES="science, v2.5, v2.0, v1.0"

fail() {
    echo "aibuildai install failed: $1" >&2
    exit 1
}

case "${LINE}" in
    science | v2.5 | v2.0 | v1.0) ;;
    "") fail "set AIBUILDAI_LINE to one of: ${LINES}." ;;
    *) fail "unknown AIBUILDAI_LINE '${LINE}'; use one of: ${LINES}." ;;
esac

OS="$(uname -s)"
MACHINE="$(uname -m)"
if [ "${OS}" != Linux ] || [ "${MACHINE}" != x86_64 ]; then
    fail "this release supports Linux x86_64 only; detected ${OS} ${MACHINE}."
fi

for TOOL in curl tar sha256sum; do
    command -v "${TOOL}" >/dev/null 2>&1 || fail "${TOOL} is needed but was not found."
done

TAG="${LINE}-latest"
ASSET="aibuildai-linux-x86_64-${TAG}.tar.gz"
DOWNLOAD_URL="${REPO_URL}/releases/download/${TAG}"

# Everything the download touches stays in this directory, and the trap removes
# it on success, on failure and on an interrupt.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/aibuildai-install.XXXXXXXX")"
trap 'rm -rf "${WORK}"' EXIT
trap 'rm -rf "${WORK}"; exit 130' INT
trap 'rm -rf "${WORK}"; exit 143' TERM

echo "Downloading ${ASSET}"
curl -fSL --progress-bar -o "${WORK}/${ASSET}" "${DOWNLOAD_URL}/${ASSET}" ||
    fail "could not download ${DOWNLOAD_URL}/${ASSET}"
curl -fsSL -o "${WORK}/SHA256SUMS" "${DOWNLOAD_URL}/SHA256SUMS" ||
    fail "could not download ${DOWNLOAD_URL}/SHA256SUMS"

echo "Checking the download against the published sha256"
# The release is not signed, so this check is the only guard against a damaged
# or replaced download. Install nothing when it does not match.
(cd "${WORK}" && grep " ${ASSET}\$" SHA256SUMS | sha256sum -c -) >/dev/null 2>&1 ||
    fail "the sha256 of ${ASSET} does not match the published SHA256SUMS."

echo "Unpacking"
mkdir "${WORK}/package"
# Each release unpacks a top-level directory whose name carries its own version,
# so strip that level and give the package one known location.
tar xzf "${WORK}/${ASSET}" -C "${WORK}/package" --strip-components=1

[ -x "${WORK}/package/install.sh" ] ||
    fail "the downloaded package has no runnable install.sh."

echo "Installing"
"${WORK}/package/install.sh"

BIN="${HOME}/.local/bin/aibuildai"
if [ -L "${BIN}" ] || [ -f "${BIN}" ]; then
    TARGET="$(readlink -f "${BIN}" 2>/dev/null || echo "${BIN}")"
    echo "Installed: ${BIN} -> ${TARGET}"
fi
