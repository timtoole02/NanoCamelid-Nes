#!/usr/bin/env bash
# The release gate: 100+ random typed questions -> 100% visible answers.
# Run scripts/nano_nes_verify.sh first (builds the model + ROM).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/Volumes/Untitled/cargo-targets/nano_nes}"
(cd "$REPO/tools/nano_nes_model_builder" && cargo build --bin gauntlet --release --quiet)
FAIL=0
for f in known_questions nearby_questions random_questions nonsense technical_questions; do
  echo "==> eval: $f"
  "$CARGO_TARGET_DIR/release/gauntlet" \
    "$REPO/tools/nano_nes_rom/out/nanocamelid.nes" \
    "$REPO/tools/nano_nes_model_builder/out/model.bin" \
    "$REPO/docs/nano_nes/corpus.txt" \
    "$REPO/docs/nano_nes/eval/$f.txt" \
    "$REPO/docs/nano_nes/receipts/gauntlet_$f.json" || FAIL=1
done
test "$FAIL" -eq 0 && echo "==> RELEASE GATE: ALL EVAL FILES PASS" || { echo "==> RELEASE GATE: FAILURES"; exit 1; }
