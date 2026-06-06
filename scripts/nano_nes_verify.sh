#!/usr/bin/env bash
# NanoCamelid NES v3 (MMC5) — one-command build + parity verification.
#
#   1. builds the word-trigram model from docs/nano_nes/corpus.txt
#   2. assembles the decode+engine banks (ca65/ld65) and stitches the final
#      1 MiB MMC5 ROM: header + tri banks 0-124 + pad + decode + engine + CHR
#   3. runs the Rust reference generator for each verification question
#   4. boots the ROM in the built-in headless NES core and TYPES each
#      question on the on-screen keyboard via the emulated controller
#   5. compares NES word IDs against the Rust reference, byte for byte
#   6. writes parity receipts to docs/nano_nes/receipts/
#
# Requires: cargo, cc65 (brew install cc65)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BUILDER="$REPO/tools/nano_nes_model_builder"
ROMDIR="$REPO/tools/nano_nes_rom"
DOCS="$REPO/docs/nano_nes"
RECEIPTS="$DOCS/receipts"
OUT="$BUILDER/out"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-/Volumes/Untitled/cargo-targets/nano_nes}"
BIN="$CARGO_TARGET_DIR/release"

echo "==> [1/6] building host tools"
(cd "$BUILDER" && cargo build --bins --release --quiet)

echo "==> [2/6] building the word model"
mkdir -p "$RECEIPTS"
"$BIN/build_model" "$DOCS/corpus.txt" "$DOCS/routes.txt" "$DOCS/topics.txt" "$DOCS/questions.txt" "$OUT" > /dev/null
cp "$OUT/model_receipt.json" "$RECEIPTS/model_receipt.json"

echo "==> [3/6] assembling MMC5 ROM (1 MiB PRG)"
mkdir -p "$ROMDIR/out"
ca65 "$ROMDIR/src/nanocamelid.s" -o "$ROMDIR/out/nanocamelid.o" \
     -I "$OUT" --bin-include-dir "$OUT" ${NANO_BUILD_MODE:+-D "$NANO_BUILD_MODE"}
ld65 -C "$ROMDIR/nes.cfg" "$ROMDIR/out/nanocamelid.o" -o "$ROMDIR/out/banks.bin"
python3 - "$OUT" "$ROMDIR/out" << 'EOF'
import sys
out, romdir = sys.argv[1], sys.argv[2]
tri = open(f"{out}/tri.bin", "rb").read()
neural = open(f"{out}/neural.bin", "rb").read()
cls = open(f"{out}/classifier.bin", "rb").read()
banks = open(f"{romdir}/banks.bin", "rb").read()
assert len(tri) == 1_015_808, len(tri)   # banks 0..123 (vocab 248)
assert len(neural) == 8192, len(neural)  # bank 124 (weights)
assert len(cls) == 8192, len(cls)        # bank 125 (classifier, $C000)
assert len(banks) == 24576, len(banks)   # decode + engine + chr
header = bytes([0x4E, 0x45, 0x53, 0x1A,  # "NES\x1a"
                64,                       # 64 x 16 KiB PRG = 1 MiB
                1,                        # 8 KiB CHR-ROM
                0x50, 0x00,               # mapper 5 (MMC5)
                0, 0, 0, 0, 0, 0, 0, 0])
rom = header + tri + neural + cls + banks[0:8192] + banks[8192:16384] + banks[16384:24576]
assert len(rom) == 16 + 1_048_576 + 8192, len(rom)
open(f"{romdir}/nanocamelid.nes", "wb").write(rom)
EOF
ROM="$ROMDIR/out/nanocamelid.nes"
echo "    ROM: $ROM ($(wc -c < "$ROM" | tr -d ' ') bytes, mapper 5)"

PASS=0
N=0
while IFS= read -r q; do
  [ -z "$q" ] && continue
  N=$((N+1))
  echo "==> [4-5/6] question: $q"
  "$BIN/refgen" "$OUT/model.bin" "$DOCS/corpus.txt" "$q" > "$OUT/ref_$N.json"
  if "$BIN/verify" "$ROM" "$OUT/model.bin" "$DOCS/corpus.txt" \
       "$OUT/ref_$N.json" "$RECEIPTS/parity_receipt_q$N.json" "$q" > /dev/null; then
    PASS=$((PASS+1))
    echo "    PASS"
  else
    echo "    FAIL (see $RECEIPTS/parity_receipt_q$N.json)"
  fi
done < "$DOCS/questions.txt"

echo "==> [6/6] $PASS/$N questions at parity. Receipts: $RECEIPTS/"
test "$PASS" -eq "$N"
