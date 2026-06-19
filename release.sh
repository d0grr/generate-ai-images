#!/usr/bin/env bash
# Build both platform release zips and publish a GitHub release.
#
# Usage:  ./release.sh [--draft] [--prerelease]
#   --draft        create the release as a draft (not published)
#   --prerelease   mark the release as a pre-release
#
# Version comes from chrome/package.json → tag vX.Y.Z, assets:
#   release/generate-ai-images-chrome_<version>.zip
#   release/generate-ai-images-firefox_<version>.zip
# Release notes are auto-generated from commits since the previous tag.
#
# Requires: node, gh (authenticated: `gh auth login`).
set -euo pipefail
cd "$(dirname "$0")"

REPO="d0grr/generate-ai-images"

DRAFT=""
PRERELEASE=""
for arg in "$@"; do
  case "$arg" in
    --draft)       DRAFT="--draft" ;;
    --prerelease)  PRERELEASE="--prerelease" ;;
    *) echo "✗ unknown option: $arg (use --draft / --prerelease)"; exit 1 ;;
  esac
done

VERSION="$(node -p "require('./chrome/package.json').version")"
TAG="v${VERSION}"
CHROME_ZIP="release/generate-ai-images-chrome_${VERSION}.zip"
FIREFOX_ZIP="release/generate-ai-images-firefox_${VERSION}.zip"

command -v gh >/dev/null || { echo "gh not found — install it and run 'gh auth login'"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh not authenticated — run 'gh auth login'"; exit 1; }

# Notes: header + commit log since the previous tag (before we create this one).
PREV_TAG="$(git tag --sort=-creatordate | head -1 || true)"
RANGE="HEAD"; [ -n "$PREV_TAG" ] && RANGE="${PREV_TAG}..HEAD"
CHANGELOG="$(git log "$RANGE" --no-merges --pretty='- %s' 2>/dev/null || true)"
NOTES="Local on-device WebGPU image generation. Chrome + Firefox (MV3) builds."
[ -n "$CHANGELOG" ] && NOTES="${NOTES}"$'\n\n'"## Changes"$'\n'"${CHANGELOG}"

echo "▸ Building Chrome + Firefox release (v${VERSION})…"
( cd chrome && node build.js && node build.js --firefox )
[ -f "$CHROME_ZIP" ] && [ -f "$FIREFOX_ZIP" ] || { echo "✗ build artifacts missing"; exit 1; }

# Tag the current commit (push if it doesn't exist yet) as the release anchor.
if ! git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "▸ Tagging ${TAG}…"
  git tag -a "$TAG" -m "Generate AI Images ${TAG}"
  git push origin "$TAG"
fi

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "▸ Release ${TAG} exists — refreshing notes + assets…"
  gh release edit "$TAG" --repo "$REPO" --notes "$NOTES"
  gh release upload "$TAG" --repo "$REPO" --clobber "$CHROME_ZIP" "$FIREFOX_ZIP"
else
  echo "▸ Creating release ${TAG}${DRAFT:+ (draft)}${PRERELEASE:+ (prerelease)}…"
  gh release create "$TAG" --repo "$REPO" --title "$TAG" --notes "$NOTES" \
    $DRAFT $PRERELEASE "$CHROME_ZIP" "$FIREFOX_ZIP"
fi

echo "✓ Done — https://github.com/${REPO}/releases/tag/${TAG}"
