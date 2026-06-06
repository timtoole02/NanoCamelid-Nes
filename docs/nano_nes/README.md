# NanoCamelid NES v5

**A tiny neural language model running inside a NES ROM.** It is not a
modern LLM, but it performs real local next-token generation on
NES-compatible 8-bit hardware: a network **trained with backprop** (hand-
rolled SGD in Rust, deterministic init), quantized to 8-bit integers, with
the **6502 executing the complete forward pass for every word** — embeddings
→ 16 ReLU hidden units → a logit for all 248 tokens — fused with n-gram
record bonuses and conditioning, then a full-vocabulary argmax. Deterministic
decoding, parity receipts.

Per word, the 6502 computes (sign-magnitude MACs through a u8 multiplier,
24-bit accumulators, power-of-2 requantization — ~4,200 multiplies):

```
h[i]     = relu((b1[i] + Σ_j W1[i][j]·x[j]) >> SH_H)         x = [emb(w1); emb(w2)]
neural[t]= clamp((b2[t] + Σ_i W2[t][i]·h[i]) >> SH_OUT, 48)   for ALL 248 tokens
score[t] = neural[t] + record_bonus[t] (n-gram ladder, 1..63)
         + topic_bias[t] + qtype_bias[t] + LEN_BONUS − REP_PEN   (u8 saturating)
```

and emits the argmax (token 0 initializes it — generation can never stall).
The network can promote words the count tables never saw; the counts keep
the grammar sharp. The forward pass is chunked across frames, so the visible
streaming pace IS the compute — no artificial delays. Conditioning (question type, topic, tone) is detected
from your typed words — including a stem matcher (`v4-stem1`: exact →
strip-S → longest-prefix≥4) so HUMMING conditions like HUM. Routes and
fallbacks still exist, but only as seeding and safety: the words themselves
come from the scored model. That is why nearby questions get *related*
answers and why the model sometimes blends corpus sentences into new ones
that read better than either.

## How it works

- **Word-level trigram** with bigram backoff, greedy decoding. A 2-word
  context carries real meaning, which is what makes Q&A work: typing
  `WHY DO LLAMAS HUM?` seeds the context `(HUM, ?)` and the model walks
  straight into the trained answer.
- **MMC5 does the data layout, the 6502 does the thinking.** 1 MiB PRG ROM:

  | Banks | Contents |
  |---|---|
  | 0–124 | dense trigram table: 16-byte records, 8 × (next word, frequency) |
  | 125 | pad |
  | 126 (fixed at `$A000`) | decode bank: word table + bigram backoff |
  | 127 (fixed at `$E000`) | engine: all code + vectors |

  Plus bank 125: the **classifier bank** at `$C000` — topic/qtype bias pages
  (page-aligned, so a bias read is one indexed load), the is-sentence-end
  table, word→group maps, and the conditioned fallback records. The hot path
  is address arithmetic + the scoring loop (~100 cycles/candidate, K=8), with
  a bank cache so consecutive lookups in the same bank never touch `$5114`
  twice (switches counted at `$07F7`).

  **Build modes** (disclosed in receipts, `$07F8`): `PURE_6502` (default)
  multiplies by shift-add — MMC5 is strictly a bank window. `MMC5_MAX`
  (`NANO_BUILD_MODE=MMC5_MAX`) runs the same MACs through the **MMC5
  hardware multiplier** (`$5205/$5206`) and enables ExRAM scratch — ~3×
  faster words, byte-identical products and tokens, same receipts. Bank 124
  holds the quantized network (~6.7 KiB: embeddings, W1/b1, W2/b2, shifts).
- Vocabulary ≤ 250 words (byte IDs are the honest cap), built from
  `docs/nano_nes/corpus.txt` — a hand-written camelid Q&A knowledge base
  whose answers each open with a unique word so greedy routing stays on
  rails. CHR tile == char ID, so rendering is direct nametable writes.
- Answers stop at a sentence end (after ≥ 4 words), on an unseen context, or
  at the 64-word budget — mirrored exactly in the Rust reference.

## The always-answer layer

**Every typed question produces a visible answer. Every time.** Unknown input
is routed, not failed:

```
typed chars -> normalize (uppercase, vocab chars, 24-char cap)
            -> parse words (unknowns skipped)
            -> classify: first question word (WHY/HOW/WHAT/...), last topic word
            -> seed ladder:
               A EXACT     typed context hits a trigram record
               B QT+TOPIC  route table (docs/nano_nes/routes.txt)
               C TOPIC     topic-only route
               D QTYPE     question-word-only route
               E FALLBACK  "HERD SAYS THIS QUESTION IS OUTSIDE THE TINY NES BRAIN..."
               F EMERGENCY watchdog reseed if the answer is under 6 words
            -> generate (trigram -> bigram backoff -> emergency jump)
```

The whole ladder runs on the 6502 (route table + attribute bits live in the
decode bank) and identically in Rust; the ROM reports its route decision at
`$07F4` and the receipts parity-check it. Answers are 6–32 words by
construction. The builder refuses to ship a model where any route seed, the
emergency seed, or any word's bigram row could stall.

**The release gate** (`scripts/nano_nes_gauntlet.sh`): the full eval suite —
25 known + 50 nearby + 102 random + 50 nonsense + 25 technical questions
(`docs/nano_nes/eval/`) — typed key-by-key through the controller protocol
on continuous NES sessions. Every answer must be 6–32 words, valid,
non-blank, and byte-identical to the Rust reference including the route
reason, backoff level, and tone. Receipts: `receipts/gauntlet_<file>.json`.

## The chat

Boot → on-screen keyboard. **D-pad** moves over the character grid, **A**
adds, **B** deletes, **Start** asks. The answer streams word by word with a
blinking cursor, a live **TOP** row showing the model's actual top
candidates *with their real corpus frequencies* (read from the same record
the 6502 just used), and a word counter. **B** starts the next question.

Try: `WHY DO LLAMAS HUM?` · `ARE YOU REAL?` · `WHAT IS YOUR NAME?` ·
`HOW DO YOU WORK?` · `WHAT IS THE NES?` — or anything; unknown words are
skipped and a fully unknown question gets the herd's fallback answer.

## One command

```sh
brew install cc65   # once
./scripts/nano_nes_verify.sh
```

Builds the model, assembles and stitches the 1 MiB MMC5 ROM, then for every
line in `docs/nano_nes/questions.txt`: runs the Rust reference, boots the ROM
in the built-in headless NES core (6502 + MMC5 PRG banking + controller),
**types the question on the on-screen keyboard via the emulated controller**,
captures the generated word IDs from NES RAM, and compares byte-for-byte.
Exit 0 means everything matched. `./scripts/nano_nes_fceux_check.sh` repeats
the whole thing in FCEUX (third-party MMC5 emulation) for independent
receipts.

## Play it in your terminal (zero setup, input always works)

```sh
./scripts/nano_nes_chat.sh
```

Runs the real ROM on the parity-verified headless core and renders the NES
screen live in your terminal. Just type your question and press Enter — each
keystroke is translated into genuine NES controller presses that walk the
ROM's on-screen keyboard ($4016 protocol and all; nothing is bypassed).
Backspace deletes / starts the next question, Ctrl-C quits. This path has no
emulator key bindings, no HID layer, and no macOS permissions to fight.

## Emulators

- **FCEUX** (`brew install fceux`): primary target, solid MMC5. Arrows =
  D-pad, **F** = A, **D** = B, **Enter** = Start.
- **Mesen2**: the MMC5 gold standard, recommended for visuals.
- **OpenEmu/Nestopia**: best effort — Nestopia's MMC5 is partial; if input
  or banking misbehaves there, use FCEUX/Mesen. (Also: OpenEmu copies ROMs
  into its library on import; replace
  `~/Library/Application Support/OpenEmu/Game Library/roms/Nintendo (NES)/nanocamelid.nes`
  after rebuilds, and grant Input Monitoring for keyboard play.)

## Parity hook contract

| Address | Meaning |
|---|---|
| `$0300..` | generated word IDs (answer only) |
| `$07F0` | status: 0 = chat input, 1 = generating, 2 = done |
| `$07F1` | generated word count |
| `$07F2` | `$FF` (every question is typed) |
| `$07F3` | word budget (32) |
| `$07F4` | route reason (0=EXACT, 1=QT+TOPIC, 2=TOPIC, 3=QTYPE, 4=FALLBACK, 5=EMERGENCY) |
| `$07F5` | max generation backoff level (0=trigram, 1=bigram, 2=topic, 3=qtype, 4=global) |
| `$07F6` | tone (0=normal, 1=empty, 2=nonsense, 3=greeting, 4=technical) |
| `$07F7` | model bank switches mod 256 (diagnostics) |
| `$07F8` | build mode (0=PURE_6502, 1=MMC5_MAX) |

Reason, level, and tone are **parity-checked** against the Rust reference on
every verified question.

## What this is not

- Not a useful modern LLM; no claim that GPT-scale models run on a NES.
- No networking, no host calls, no co-processor, no hidden hardware. The
  showcase layer (candidate display, pacing, thinking dots) is presentation
  only and provably never changes the word sequence.
- Greedy only; seeded sampling is future work. Going past 1 MiB honestly
  means 16-bit word IDs or sparse 4-grams — also future work, not faked now.

## Receipts

`docs/nano_nes/receipts/`: `model_receipt.json` (corpus/vocab/model SHA-256,
vocab size, context counts), `parity_receipt_q<N>.json` (headless core) and
`fceux_receipt_q<N>.json` (FCEUX) — each with ROM/model/corpus hashes, the
question, both word-ID sequences, decoded text, and the PASS/FAIL verdict.
