#!/usr/bin/env bash
# Commit the refreshed faiss_index/ and push it to a branch that may have moved.
#
# The weekly update takes 15+ minutes, so a merge to main during the run made
# the final `git push` fail ("rejected, fetch first") and threw the whole run's
# work (and its OpenAI spend) away. This job only ever changes faiss_index/, so
# instead of merging or rebasing it simply re-applies the new index on top of
# the latest remote branch and pushes again, retrying if the branch moves again.
#
# Usage: commit_index.sh [branch]   (default: main; run from the repo root)
set -euo pipefail

BRANCH="${1:-main}"
MESSAGE="chore: weekly knowledge base update [skip ci]"

git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

git add faiss_index/
if git diff --staged --quiet; then
  echo "No knowledge base changes to commit."
  exit 0
fi

# Keep a copy of the new index: `git reset --hard` below replaces the working tree.
STAGE="$(mktemp -d)"
cp -r faiss_index "$STAGE/faiss_index"

for attempt in 1 2 3 4 5; do
  git fetch --depth=1 origin "$BRANCH"
  git reset --hard "FETCH_HEAD" > /dev/null
  rm -rf faiss_index
  cp -r "$STAGE/faiss_index" faiss_index
  git add faiss_index/
  if git diff --staged --quiet; then
    echo "The remote already contains these changes."
    exit 0
  fi
  git commit -q -m "$MESSAGE"
  if git push origin "HEAD:$BRANCH"; then
    echo "Pushed the updated index (attempt $attempt)."
    exit 0
  fi
  echo "Push rejected (attempt $attempt): $BRANCH moved. Retrying on top of the latest commit..."
  sleep $((attempt * ${RETRY_SLEEP_SECONDS:-5}))
done

echo "::error::Could not push the updated index after 5 attempts."
exit 1
