#!/usr/bin/env bash
# Public-repo hygiene: this repository is publicly mirrored — internal
# hostnames, cluster paths, and deployment-tooling references must never
# appear. Runs in CI (hard fail) and locally before pushing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# One pattern per line. Extend when a new internal marker appears.
PATTERNS=(
  'gitlab\.innovius\.ai'
  'registry\.innovius\.ai'
  'taskman\.innovius'
  'credman'
  'chartman'
  '__DEPLOY'
  'portal\.chtsafe'
  'work-support-portal'
  'westai'
  '/p/project1'
  'svc\.cluster\.local'
  'JURECA'
  'JUPITER supercomput'
  'sbatch'
)

FAIL=0
for pattern in "${PATTERNS[@]}"; do
  # Plain grep (no git dependency — CI lint images are minimal); the hygiene
  # script itself carries the patterns and is excluded.
  hits=$(grep -RInE "$pattern" . \
    --exclude-dir=.git --exclude-dir=.venv --exclude-dir=__pycache__ \
    --exclude-dir=.pytest_cache --exclude-dir=.ruff_cache \
    --exclude-dir='*.egg-info' --exclude-dir=.pip-cache \
    --exclude=check-public-hygiene.sh || true)
  if [ -n "$hits" ]; then
    echo "HYGIENE FAIL: pattern '$pattern' found:" >&2
    echo "$hits" >&2
    FAIL=1
  fi
done

if [ "$FAIL" -ne 0 ]; then
  echo "This repo is publicly mirrored — remove the internal references above." >&2
  exit 1
fi
echo "hygiene: clean"
