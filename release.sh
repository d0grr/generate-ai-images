#!/usr/bin/env bash
# Build both platform release zips and publish a GitHub release.
#
# Usage:  ./release.sh
# Version comes from chrome/package.json → tag vX.Y.Z, assets:
#   release/generate-ai-images-chrome_<version>.zip
#   release/generate-ai-images-firefox_<version>.zip
#
# Requires: node, gh (authenticated: `gh auth login`).
set -euo pipefail
cd "$(dirname "$0")"

REPO="d0grr/generate-ai-images"
VERSION="$(node -p "require('./chrome/package.json').version")"
TAG="v${VERSION}"
CHROME_ZIP="release/generate-ai-images-chrome_${VERSION}.zip"
FIREFOX_ZIP="release/generate-ai-images-firefox_${VERSION}.zip"

command -v gh >/dev/null || { echo "gh not found — install it and run 'gh auth login'"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh not authenticated — run 'gh auth login'"; exit 1; }

echo "▸ Building Chrome + Firefox release (v${VERSION})…"
( cd chrome && node build.js && node build.js --firefox )
[ -f "$CHROME_ZIP" ] && [ -f "$FIREFOX_ZIP" ] || { echo "✗ build artifacts missing"; exit 1; }

# Tag the current commit (push if it doesn't exist yet) as the release anchor.
if ! git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "▸ Tagging ${TAG}…"
  git tag -a "$TAG" -m "Generate AI Images ${TAG}"
  git push origin "$TAG"
fi

NOTES="Local on-device WebGPU image generation. Chrome + Firefox (MV3) builds."
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "▸ Release ${TAG} exists — uploading assets (clobber)…"
  gh release upload "$TAG" --repo "$REPO" --clobber "$CHROME_ZIP" "$FIREFOX_ZIP"
else
  echo "▸ Creating release ${TAG}…"
  gh release create "$TAG" --repo "$REPO" --title "$TAG" --notes "$NOTES" \
    "$CHROME_ZIP" "$FIREFOX_ZIP"
fi

echo "✓ Done — https://github.com/${REPO}/releases/tag/${TAG}"
