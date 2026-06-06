#!/usr/bin/env bash
# Chat with NanoCamelid NES in your terminal — zero emulator/controller setup.
# Runs the real MMC5 ROM on the parity-verified headless core; your typing is
# translated into genuine NES controller presses.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/Volumes/Untitled/cargo-targets/nano_nes}"
ROM="$REPO/tools/nano_nes_rom/out/nanocamelid.nes"
test -f "$ROM" || { echo "ROM missing — run scripts/nano_nes_verify.sh first"; exit 2; }
(cd "$REPO/tools/nano_nes_model_builder" && cargo build --bin play --quiet)
exec "$CARGO_TARGET_DIR/debug/play" "$ROM"
