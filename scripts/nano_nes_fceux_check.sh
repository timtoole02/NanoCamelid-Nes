#!/usr/bin/env bash
# NanoCamelid NES v3 — third-party emulator parity check (FCEUX, MMC5).
#
# Complements scripts/nano_nes_verify.sh (built-in headless core): boots the
# ROM in FCEUX with a Lua probe that TYPES each verification question on the
# on-screen keyboard through the real controller protocol, then compares the
# word IDs captured from NES RAM against the Rust reference. Opens a GUI
# window briefly per question.
#
# Requires: brew install fceux ; run scripts/nano_nes_verify.sh first.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$REPO/tools/nano_nes_model_builder/out"
ROM="$REPO/tools/nano_nes_rom/out/nanocamelid.nes"
LUA="$REPO/tools/nano_nes_rom/test/fceux_probe.lua"
RECEIPTS="$REPO/docs/nano_nes/receipts"
QUESTIONS="$REPO/docs/nano_nes/questions.txt"

command -v fceux > /dev/null || { echo "fceux not found (brew install fceux)"; exit 2; }
test -f "$ROM" || { echo "ROM missing — run scripts/nano_nes_verify.sh first"; exit 2; }

FAIL=0
N=0
while IFS= read -r q; do
  [ -z "$q" ] && continue
  N=$((N+1))
  test -f "$OUT/ref_$N.json" || { echo "missing $OUT/ref_$N.json — run nano_nes_verify.sh first"; exit 2; }
  rm -f /tmp/nano_fceux_result.txt
  echo "==> FCEUX run: $q"
  NANO_QUESTION="$q" fceux --loadlua "$LUA" "$ROM" > /dev/null 2>&1 || true
  for _ in $(seq 1 60); do test -s /tmp/nano_fceux_result.txt && break; sleep 1; done
  python3 - "$N" "$REPO" << 'EOF' || FAIL=1
import hashlib, json, sys
n = int(sys.argv[1])
repo = sys.argv[2]
ref = json.load(open(f"{repo}/tools/nano_nes_model_builder/out/ref_{n}.json"))
probe = open("/tmp/nano_fceux_result.txt").read()
final = [l for l in probe.splitlines() if l.startswith("FINAL")][0]
ids_line = [l for l in probe.splitlines() if l.startswith("word_ids=")][0]
nes_ids = [int(t) for t in ids_line[9:].split(",") if t]
fields = dict(kv.split("=", 1) for kv in final.split(None, 4)[1:4])
sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()
ok = (nes_ids == ref["generated_word_ids"]
      and int(fields["status"]) == 2 and int(fields["prompt_id"]) == 255)
receipt = {
    "tool": "scripts/nano_nes_fceux_check.sh",
    "emulator": "FCEUX (third-party, MMC5) via Lua probe typing on the on-screen keyboard",
    "rom_sha256": sha(f"{repo}/tools/nano_nes_rom/out/nanocamelid.nes"),
    "model_sha256": ref["model_sha256"],
    "corpus_sha256": ref["corpus_sha256"],
    "question": ref["question"],
    "rust_generated_word_ids": ref["generated_word_ids"],
    "nes_generated_word_ids": nes_ids,
    "decoded_text": ref["decoded_text"],
    "parity": "PASS" if ok else "FAIL",
}
path = f"{repo}/docs/nano_nes/receipts/fceux_receipt_q{n}.json"
json.dump(receipt, open(path, "w"), indent=2)
print(f"    {'PASS' if ok else 'FAIL'} ({len(nes_ids)} words)")
sys.exit(0 if ok else 1)
EOF
done < "$QUESTIONS"
test "$FAIL" -eq 0 && echo "==> FCEUX parity: all questions PASS" || { echo "==> FCEUX parity: FAILURES"; exit 1; }
