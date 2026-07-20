#!/usr/bin/env bash
#
# publish-github.sh — mirror the current working tree to the public
# GitHub repo as a single squashed "Initial public release" commit, then
# force-push to refs/heads/main.
#
# Invoked exclusively from the publish:github CI job on the main branch.
# DO NOT run this locally without understanding that it force-pushes
# github.com/Podwarden/vllm-warden:main.
#
# Secrets handling:
#   - GITHUB_PUBLISH_SSH_KEY is a protected GitLab CI *File* variable holding
#     a deploy-key private key; in CI the env var is the path to that file.
#     The CI job mounts it into the publish container and passes its path via
#     GITHUB_PUBLISH_SSH_KEY_FILE. We never echo the key and never `set -x`.

set -euo pipefail

# ---- defense in depth ------------------------------------------------------
# The YAML rules already gate this to main + the key being defined, but
# bail loudly here too in case someone wires the script up elsewhere.

SSH_KEY_FILE="${GITHUB_PUBLISH_SSH_KEY_FILE:-${GITHUB_PUBLISH_SSH_KEY:-}}"
if [[ -z "$SSH_KEY_FILE" || ! -s "$SSH_KEY_FILE" ]]; then
  echo "ERROR: SSH deploy key not available." >&2
  echo "Expected GITHUB_PUBLISH_SSH_KEY_FILE (or the GITHUB_PUBLISH_SSH_KEY" >&2
  echo "protected File CI variable) to point at a non-empty private key." >&2
  exit 1
fi

if [[ "${CI_COMMIT_REF_NAME:-}" != "main" ]]; then
  echo "ERROR: refusing to publish from ref '${CI_COMMIT_REF_NAME:-<unset>}'." >&2
  echo "This script must only run from the main branch." >&2
  exit 1
fi

if [[ -z "${CI_PROJECT_DIR:-}" ]]; then
  echo "ERROR: CI_PROJECT_DIR is unset; refusing to guess workspace." >&2
  exit 1
fi

# ---- stage a clean tree ----------------------------------------------------

# Stage outside the checkout so we never write into the (possibly
# read-only / root-owned) mounted workspace. Overridable for local runs.
STAGE_DIR="${PUBLISH_STAGE_DIR:-/tmp/vllm-warden-publish-stage}"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# Copy the full working tree, minus .git and the stage dir itself.
rsync -a \
  --exclude='.git' \
  --exclude='.publish-stage' \
  "${CI_PROJECT_DIR}/" "${STAGE_DIR}/"

cd "$STAGE_DIR"

# rsync -a preserves the source files' ownership (the runner uid), but git
# runs here as root inside the publish container, so the staged tree looks
# "dubiously owned" to git 2.35+. Whitelist it explicitly.
git config --global --add safe.directory "$STAGE_DIR"

git init -q -b main

export GIT_AUTHOR_NAME="vllm-warden CI"
export GIT_AUTHOR_EMAIL="ci@podwarden.com"
export GIT_COMMITTER_NAME="vllm-warden CI"
export GIT_COMMITTER_EMAIL="ci@podwarden.com"

git add -A
git commit -q -m "Initial public release"
SHA="$(git rev-parse HEAD)"

# ---- ssh auth --------------------------------------------------------------
# Copy the deploy key to a private location with 0600 perms (ssh refuses
# group/world-readable keys; the mounted CI file may be too permissive).

SSH_DIR="$(mktemp -d)"
chmod 700 "$SSH_DIR"
# GitLab strips the trailing newline from File-variable values (they are stored
# via command substitution, which drops it). OpenSSH then rejects the key with
# "error in libcrypto". Normalize to exactly one trailing newline regardless of
# how the variable was stored: `cat` drops all trailing newlines, `printf` adds
# exactly one back. Create via install(1) first so the file is 0600 before any
# key bytes land in it.
install -m 600 /dev/null "$SSH_DIR/id_ed25519"
printf '%s\n' "$(cat "$SSH_KEY_FILE")" > "$SSH_DIR/id_ed25519"

# Pin GitHub's host keys instead of blindly trusting on first use.
ssh-keyscan -t rsa,ecdsa,ed25519 github.com > "$SSH_DIR/known_hosts" 2>/dev/null

export GIT_SSH_COMMAND="ssh -i '$SSH_DIR/id_ed25519' -o IdentitiesOnly=yes -o UserKnownHostsFile='$SSH_DIR/known_hosts' -o StrictHostKeyChecking=yes"

# ---- push ------------------------------------------------------------------

git remote add github "git@github.com:Podwarden/vllm-warden.git"

# Force-push because the GitHub repo is dedicated to this publish flow and
# we replace history each release.
git push --force github HEAD:main

FILE_COUNT="$(git ls-files | wc -l | tr -d ' ')"
echo "Pushed ${FILE_COUNT} files to github.com/Podwarden/vllm-warden:main as commit ${SHA}"
