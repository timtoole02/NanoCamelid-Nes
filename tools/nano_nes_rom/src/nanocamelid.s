; ---------------------------------------------------------------------------
; NanoCamelid NES v3 — a word-level language model chat on the NES, built
; around MMC5 (mapper 5) from the start.
;
; The 6502 runs the whole next-word loop. MMC5 is a bank window, not a
; co-processor: the 1 MiB PRG holds a dense word-trigram table (16-byte
; records of 8 x (next_word, freq)), and the data layout does the hard work:
;
;   bank   = w1 >> 1              -> $5114 (model window at $8000-$9FFF)
;   record = $8000 + (w1 & 1) * 4096 + w2 * 16
;   next   = record[0]            (greedy), record[1] = its frequency
;
; Fixed banks: #126 decode (word table + bigram fallback) at $A000,
; #127 engine (all code + vectors) at $E000. MMC5 features deliberately NOT
; used: ExRAM, IRQ, split screen, multiplier, audio.
;
; Chat-only UI: boot -> on-screen keyboard; type a question (D-pad + A/B),
; Start asks; the answer streams with a live top-4 candidate display fed by
; the real record frequencies. B starts the next question.
;
; Parity hook contract (read by tools/nano_nes_model_builder verify):
;   $0300..  generated word IDs (answer only)
;   $07F0    status: 0=chat input, 1=generating, 2=done
;   $07F1    generated word count
;   $07F2    $FF (every question is typed)
;   $07F3    word budget
; ---------------------------------------------------------------------------

.include "model_consts.inc"     ; MODEL_VOCAB_SIZE, WORD_DOT/COMMA/EXCL/QM,
                                ; WORD_QMARK_SEED, WORD_BUDGET, MIN_WORDS_STOP

TILE_SEL       = 62
TILE_BLOCK     = 63
TILE_LLAMA     = 48
TILE_ZERO      = 27
TILE_DOT       = 37
TILE_DASH      = 40

STATE_KB       = 0
STATE_GEN      = 1
STATE_DONE     = 2
STATE_THINK    = 3

; gen_frame phases
PH_EMIT        = 0
PH_DASH1       = 1
PH_DASH2       = 2
PH_COUNT       = 3
PH_NHID        = 4              ; v5: hidden layer chunk
PH_NOUT        = 5              ; v5: output layer chunk
PH_NCOND       = 6              ; v5: conditioning sweep + argmax + emit
PH_STREAM      = 9

SCORES         = $0400          ; per-token score scratch (248 bytes)
NEURAL_BANK    = 124

.ifdef MMC5_MAX
HID_CHUNK      = 16             ; hardware multiplier: whole layer per frame
OUT_CHUNK      = 24
.else
HID_CHUNK      = 8              ; shift-add multiply: smaller chunks
OUT_CHUNK      = 8
.endif

; neural bank addresses ($8000 window while computing)
NW_EMB         = $8000          ; emb[wid*8], i8
NW_W1          = $8800          ; w1[i*16+j], i8
NW_B1          = $8900          ; b1[i], i16 LE
NW_B2          = $8A00          ; b2[t], i16 LE
NW_W2          = $8C00          ; w2[t*16+i], i8

MAX_TYPED      = 24
NO_WORD        = $FF

TYPED          = $0200          ; typed question (char IDs)
WORDBUF        = $0300          ; generated word IDs (parity buffer)

STATUS_ADDR    = $07F0
REASON_ADDR    = $07F4          ; route reason (0=EXACT..5=EMERGENCY), parity-checked
LEVEL_ADDR     = $07F5          ; max generation backoff level (0=tri..4=global), parity-checked
TONE_ADDR      = $07F6          ; detected tone (0=normal..4=technical), parity-checked
BANKSW_ADDR    = $07F7          ; model bank switches (mod 256, diagnostics)
MODE_ADDR      = $07F8          ; build mode: 0=PURE_6502, 1=MMC5_MAX
COUNT_ADDR     = $07F1
PROMPTID_ADDR  = $07F2
BUDGET_ADDR    = $07F3

PAD_A          = $80
PAD_B          = $40
PAD_SELECT     = $20
PAD_START      = $10
PAD_UP         = $08
PAD_DOWN       = $04
PAD_LEFT       = $02
PAD_RIGHT      = $01

; MMC5 registers
MMC5_PRG_MODE  = $5100
MMC5_CHR_MODE  = $5101
MMC5_EXRAM_MD  = $5104
MMC5_NT_MAP    = $5105
MMC5_PRG_8000  = $5114
MMC5_PRG_A000  = $5115
MMC5_PRG_C000  = $5116
MMC5_PRG_E000  = $5117

CLS_BANK       = 125            ; classifier bank at $C000 via $5116
DECODE_BANK    = 126
ENGINE_BANK    = 127

; classifier bank fixed addresses (mirror CLS_* in lib.rs)
CLS_TBIAS      = $C000          ; 16 topic bias pages
CLS_QBIAS      = $D000          ; 11 qtype bias pages
CLS_ISEND      = $DB00          ; is-sentence-end per token
CLS_TGROUP     = $DC00          ; word -> topic group + 1
CLS_QGROUP     = $DD00          ; word -> qtype group + 1
CLS_TRECS      = $DE00          ; topic fallback records
CLS_QRECS      = $DF00          ; qtype fallback records
CLS_GLOBAL     = $DFC0          ; global fallback record
CLS_TONE       = $DFE0          ; tone per topic group

; --- char-ID encoding for assembly-time strings ------------------------------
.macro tokch c
  .byte ((((c)>=$41) .and ((c)<=$5A)) * ((c)-$40) + (((c)>=$30) .and ((c)<=$39)) * ((c)-$15) + ((c)=$2E)*37 + ((c)=$2C)*38 + ((c)=$27)*39 + ((c)=$2D)*40 + ((c)=$21)*41 + ((c)=$3F)*42)
.endmacro

.macro vtext str
  .repeat .strlen(str), i
    tokch {.strat(str, i)}
  .endrepeat
.endmacro

.macro strrec addr, str
  .byte >(addr), <(addr), .strlen(str)
  vtext str
.endmacro

; ---------------------------------------------------------------------------
.segment "ZEROPAGE"
nmi_count:    .res 1
state:        .res 1
w1:           .res 1            ; rolling 2-word context ($FF = none)
w2:           .res 1
ptr_lo:       .res 1
ptr_hi:       .res 1
wptr_lo:      .res 1            ; current word string pointer (decode bank)
wptr_hi:      .res 1
tmp:          .res 1
tmp_hi:       .res 1            ; queue_tile target PPU address
tmp_lo:       .res 1
cursor_lo:    .res 1
cursor_hi:    .res 1
cur_col:      .res 1
cur_row:      .res 1
typed_len:    .res 1
pad_state:    .res 1
pad_prev:     .res 1
pad_pressed:  .res 1
q_count:      .res 1            ; vblank tile queue (max 16/frame)
soft2000:     .res 1
ti:           .res 1
think_timer:  .res 1
delay_timer:  .res 1
lfsr:         .res 1            ; pacing only — never touches the words
phase:        .res 1
blink_state:  .res 1
pad_disp:     .res 1
kb_row:       .res 1
kb_col:       .res 1
qw1:          .res 1            ; parsed question context
qw2:          .res 1
qw_n:         .res 1
qt_found:     .res 1            ; first known qtype word ($FE = none)
topic_found:  .res 1            ; last known topic word ($FE = none)
retried:      .res 1            ; watchdog: emergency regen done once
qsave_x:      .res 1            ; queue_tile X preservation
topic_g:      .res 1            ; topic group + 1 (0 = none)
qtype_g:      .res 1            ; qtype group + 1 (0 = none)
tb_hi:        .res 1            ; topic bias page ($C0+g) or 0
qb_hi:        .res 1            ; qtype bias page ($D0+g) or 0
bp_lo:        .res 1            ; bias page pointer (lo always 0)
bp_hi:        .res 1
had_chars:    .res 1            ; parser saw any word/punct chars
best_s:       .res 1            ; scoring loop state
best_t:       .res 1
sc:           .res 1
last_bank:    .res 1            ; model bank cache ($FF = none)
ring_i:       .res 1
ring:         .res 8            ; last 8 emitted tokens (repetition penalty)
x_arr:        .res 16           ; v5 input vector [emb(w1); emb(w2)], i8
h_arr:        .res 16           ; v5 hidden activations, u8
acc0:         .res 1            ; 24-bit signed accumulator
acc1:         .res 1
acc2:         .res 1
m1:           .res 1            ; multiply scratch
m2:           .res 1
prod_lo:      .res 1
prod_hi:      .res 1
msign:        .res 1
nt_i:         .res 1            ; hidden unit / token index
nt_t:         .res 1
chunk:        .res 1
wstart:       .res 1            ; submit-time word parser
wblen:        .res 1
sx:           .res 1
wid:          .res 1
wlen:         .res 1            ; streaming word
wpos:         .res 1
first_word:   .res 1
no_draw:      .res 1            ; answer area full: keep generating, stop drawing
stop_flag:    .res 1
emitted:      .res 1            ; the word just emitted (for stop check)
dig_t:        .res 1
dig_o:        .res 1
rec:          .res 16           ; current 16-byte transition record
q_hi:         .res 16
q_lo:         .res 16
q_tile:       .res 16

; ---------------------------------------------------------------------------
; Decode bank (#126, $A000): bigram fallback + word table. Laid out by ld65;
; the build script places this 8 KiB at ROM offset 126*8192.
.segment "BIGRAM"
bi_table:                       ; page-aligned: record = bi_table + w * 16
  .incbin "bi.bin"

.segment "WORDS"
word_attr:                      ; per-word ATTR bits: 1=qtype, 2=topic
  .incbin "attr.bin"
routes_tbl:                     ; [count][level,qtype,topic,seed1,seed2]...
  .incbin "routes.bin"
words_bin:                      ; woff_lo[250], woff_hi[250], len+chars...
  .incbin "words.bin"
woff_lo = words_bin
woff_hi = words_bin + 248
wdata   = words_bin + 496

; ---------------------------------------------------------------------------
.segment "CODE"

reset:
  sei
  cld
  ldx #$40
  stx $4017
  ldx #$FF
  txs
  inx
  stx $2000
  stx $2001
  stx $4010
  ; MMC5 init — engine bank is already visible at $E000 (power-on $5117=$FF)
  lda #3
  sta MMC5_PRG_MODE             ; four 8 KiB windows
  lda #3
  sta MMC5_CHR_MODE             ; 1 KiB CHR pages
  ldx #0
@chr:
  txa
  sta $5120,x                   ; CHR pages 0..7 -> linear 8 KiB font
  inx
  cpx #8
  bne @chr
  lda #0
  sta MMC5_EXRAM_MD
  sta MMC5_NT_MAP               ; all nametables -> CIRAM page 0
  lda #(DECODE_BANK | $80)
  sta MMC5_PRG_A000             ; decode bank fixed at $A000
  lda #(CLS_BANK | $80)
  sta MMC5_PRG_C000             ; classifier bank at $C000
  lda #(ENGINE_BANK | $80)
  sta MMC5_PRG_E000             ; engine at $E000 (explicit)
  lda #$80
  sta MMC5_PRG_8000             ; model window -> bank 0

  bit $2002
@vb1:
  bit $2002
  bpl @vb1
  lda #0
  tax
@clr:
  sta $0000,x
  sta $0100,x
  sta $0200,x
  sta $0300,x
  sta $0400,x
  sta $0500,x
  sta $0600,x
  sta $0700,x
  inx
  bne @clr
@vb2:
  bit $2002
  bpl @vb2

  ; palette
  bit $2002
  lda #$3F
  sta $2006
  lda #$00
  sta $2006
  ldx #0
@pal:
  lda palette_data,x
  sta $2007
  inx
  cpx #32
  bne @pal

  lda #$A5
  sta lfsr                      ; pacing only — never touches the words
  lda #0
  sta bp_lo                     ; bias pages are page-aligned
.ifdef MMC5_MAX
  lda #2
  sta MMC5_EXRAM_MD             ; ExRAM as read/write scratch (disclosed)
  lda #1
  sta MODE_ADDR                 ; build mode: MMC5_MAX
.endif
  lda #%10000000
  sta soft2000
  jsr enter_chat

main_loop:
  jsr wait_nmi
  jsr read_pad
  lda state
  beq @kb
  cmp #STATE_GEN
  beq @gen
  cmp #STATE_THINK
  beq @think
  jsr done_step
  jmp main_loop
@kb:
  jsr kb_step
  jmp main_loop
@gen:
  jsr gen_frame
  jmp main_loop
@think:
  jsr think_step
  jmp main_loop

; ---------------------------------------------------------------------------
; fetch_rec: v4 candidate ladder — tri -> bi -> topic prior -> qtype prior ->
; global. `sc` holds the level (0..4); LEVEL_ADDR tracks the max (receipts).
fetch_rec:
  lda #0
  sta sc
  jsr tri_probe
  cmp #NO_WORD
  bne @copy
  lda #1
  sta sc
  lda w2                        ; bigram record: bi_table + w2*16
  lsr
  lsr
  lsr
  lsr
  clc
  adc #>bi_table
  sta ptr_hi
  lda w2
  asl
  asl
  asl
  asl
  sta ptr_lo
  ldy #0
  lda (ptr_lo),y
  cmp #NO_WORD
  bne @copy
  lda topic_g
  beq @qr
  lda #2
  sta sc
  lda topic_g                   ; topic record: CLS_TRECS + (g-1)*16
  sec
  sbc #1
  asl
  asl
  asl
  asl
  sta ptr_lo
  lda #>CLS_TRECS
  sta ptr_hi
  ldy #0
  lda (ptr_lo),y
  cmp #NO_WORD
  bne @copy
@qr:
  lda qtype_g
  beq @glob
  lda #3
  sta sc
  lda qtype_g                   ; qtype record: CLS_QRECS + (g-1)*16
  sec
  sbc #1
  asl
  asl
  asl
  asl
  sta ptr_lo
  lda #>CLS_QRECS
  sta ptr_hi
  ldy #0
  lda (ptr_lo),y
  cmp #NO_WORD
  bne @copy
@glob:
  lda #4
  sta sc
  lda #<CLS_GLOBAL
  sta ptr_lo
  lda #>CLS_GLOBAL
  sta ptr_hi
@copy:
  lda sc
  cmp LEVEL_ADDR
  bcc @lv_ok
  sta LEVEL_ADDR
@lv_ok:
  ldy #15
@cl:
  lda (ptr_lo),y
  sta rec,y
  dey
  bpl @cl
  rts

; tri_probe: point ptr at the trigram record for (w1, w2) and return its
; first byte in A ($FF = empty or w1 = none). Leaves the bank switched.
tri_probe:
  lda w1
  cmp #NO_WORD
  beq @none
  lsr                           ; bank = w1 >> 1
  ora #$80                      ; bit7 = ROM
  jsr set_model_bank
  lda w1
  and #1
  asl
  asl
  asl
  asl                           ; (w1 & 1) << 4 -> high-byte nibble (4096)
  sta tmp
  lda w2
  lsr
  lsr
  lsr
  lsr                           ; w2 >> 4
  ora tmp
  ora #$80                      ; window base $8000
  sta ptr_hi
  lda w2
  asl
  asl
  asl
  asl                           ; (w2 << 4) & $FF
  sta ptr_lo
  ldy #0
  lda (ptr_lo),y
  rts
@none:
  lda #NO_WORD
  rts

; --- v5 neural engine ---------------------------------------------------
; set_model_bank: A = bank|$80 -> $5114 with cache + switch counter
set_model_bank:
  cmp last_bank
  beq @r
  sta last_bank
  sta MMC5_PRG_8000
  inc BANKSW_ADDR
@r:
  rts

; mul8u: prod = A * Y (u8 x u8 -> u16). MMC5_MAX uses the hardware
; multiplier; PURE_6502 uses shift-add. Identical products, different cycles.
mul8u:
.ifdef MMC5_MAX
  sta $5205
  sty $5206
  lda $5205
  sta prod_lo
  lda $5206
  sta prod_hi
  rts
.else
  sta m1
  sty m2
  lda #0
  sta prod_lo
  sta prod_hi
  ldx #8
@l:
  lsr m2
  bcc @no
  clc
  lda prod_hi
  adc m1
  sta prod_hi
@no:
  lsr prod_hi
  ror prod_lo
  dex
  bne @l
  rts
.endif

; mac_ss: acc24 += (i8 in A) * (i8 in X)   sign-magnitude via mul8u
mac_ss:
  sta m1
  txa
  sta m2
  lda m1
  eor m2
  and #$80
  sta msign
  lda m1
  bpl @a1
  eor #$FF
  clc
  adc #1
@a1:
  sta m1
  lda m2
  bpl @a2
  eor #$FF
  clc
  adc #1
@a2:
  tay
  lda m1
  jsr mul8u
  jmp acc_apply

; mac_su: acc24 += (i8 in A) * (u8 in X)
mac_su:
  sta m1
  and #$80
  sta msign
  lda m1
  bpl @p
  eor #$FF
  clc
  adc #1
@p:
  sta m1
  txa
  tay
  lda m1
  jsr mul8u
  ; fall through
acc_apply:
  lda msign
  bmi @sub
  clc
  lda acc0
  adc prod_lo
  sta acc0
  lda acc1
  adc prod_hi
  sta acc1
  lda acc2
  adc #0
  sta acc2
  rts
@sub:
  sec
  lda acc0
  sbc prod_lo
  sta acc0
  lda acc1
  sbc prod_hi
  sta acc1
  lda acc2
  sbc #0
  sta acc2
  rts

; relu_shift: acc24 -> u8 via (acc <= 0 ? 0 : min(acc >> shift, clamp))
; X = shift count, Y = clamp ($FF for the hidden layer)
relu_shift:
  lda acc2
  bmi @zero
  cpx #0
  beq @sdone
@s:
  lsr acc2
  ror acc1
  ror acc0
  dex
  bne @s
@sdone:
  lda acc2
  ora acc1
  bne @clamp
  sty m1
  lda acc0
  cmp m1
  bcc @ok
  beq @ok
@clamp:
  tya
  rts
@ok:
  lda acc0
  rts
@zero:
  lda #0
  rts

; emb_ptr: A = word id -> ptr = NW_EMB + id*8
emb_ptr:
  sta m1
  lsr
  lsr
  lsr
  lsr
  lsr
  clc
  adc #>NW_EMB
  sta ptr_hi
  lda m1
  asl
  asl
  asl
  sta ptr_lo
  rts

; neural_begin: map the weight bank, load shifts, build x = [emb(w1); emb(w2)]
neural_begin:
  lda #(NEURAL_BANK | $80)
  jsr set_model_bank
  lda w1
  cmp #NO_WORD
  bne @e1
  ldx #7
  lda #0
@z:
  sta x_arr,x
  dex
  bpl @z
  jmp @x2
@e1:
  lda w1
  jsr emb_ptr
  ldy #7
@c1:
  lda (ptr_lo),y
  sta x_arr,y
  dey
  bpl @c1
@x2:
  lda w2
  jsr emb_ptr
  ldy #7
@c2:
  lda (ptr_lo),y
  sta x_arr+8,y
  dey
  bpl @c2
  lda #0
  sta nt_i
  rts

; one hidden unit: h[nt_i] = relu((b1 + sum w1*x) >> SH_H)
hid_unit:
  lda nt_i
  asl
  tax
  lda NW_B1,x
  sta acc0
  lda NW_B1+1,x
  sta acc1
  bmi @neg
  lda #0
  beq @se
@neg:
  lda #$FF
@se:
  sta acc2
  lda nt_i
  asl
  asl
  asl
  asl
  sta ptr_lo
  lda #>NW_W1
  sta ptr_hi
  ldy #0
@j:
  lda (ptr_lo),y
  ldx x_arr,y
  sty sx
  jsr mac_ss
  ldy sx
  iny
  cpy #16
  bne @j
  ldx #NEURAL_SH_H
  ldy #$FF
  jsr relu_shift
  ldx nt_i
  sta h_arr,x
  inc nt_i
  rts

gf_nhid:
  lda #(NEURAL_BANK | $80)
  jsr set_model_bank
  lda #HID_CHUNK
  sta chunk
@l:
  jsr hid_unit
  lda nt_i
  cmp #16
  beq @done
  dec chunk
  bne @l
  rts
@done:
  lda #0
  sta nt_t
  lda #PH_NOUT
  sta phase
  rts

; one output token: SCORES[nt_t] = clamp((b2 + sum w2*h) >> SH_OUT, NEURAL_CLAMP)
out_token:
  lda nt_t
  asl
  sta ptr_lo
  lda #>NW_B2
  adc #0
  sta ptr_hi
  ldy #0
  lda (ptr_lo),y
  sta acc0
  iny
  lda (ptr_lo),y
  sta acc1
  bmi @neg
  lda #0
  beq @se
@neg:
  lda #$FF
@se:
  sta acc2
  lda nt_t
  lsr
  lsr
  lsr
  lsr
  clc
  adc #>NW_W2
  sta ptr_hi
  lda nt_t
  asl
  asl
  asl
  asl
  sta ptr_lo
  ldy #0
@i:
  lda (ptr_lo),y
  ldx h_arr,y
  sty sx
  jsr mac_su
  ldy sx
  iny
  cpy #16
  bne @i
  ldx #NEURAL_SH_OUT
  ldy #NEURAL_CLAMP
  jsr relu_shift
  ldx nt_t
  sta SCORES,x
  inc nt_t
  rts

gf_nout:
  lda #(NEURAL_BANK | $80)
  jsr set_model_bank
  lda #OUT_CHUNK
  sta chunk
@l:
  jsr out_token
  lda nt_t
  cmp #MODEL_VOCAB_SIZE
  beq @done
  dec chunk
  bne @l
  rts
@done:
  lda #PH_NCOND
  sta phase
  rts

; conditioning sweep + argmax -> emit. Mirrors gen_from in lib.rs exactly:
; per token: sc = neural_base (+record bonus, already added) + topic + qtype
;            + len bonus - rep penalty; strict-> first max wins, token 0 inits.
gf_ncond:
  jsr fetch_rec                 ; v4 ladder (switches tri banks; cache-safe)
  ldx #0
@b:
  lda rec,x
  cmp #NO_WORD
  beq @bdone
  tay
  lda rec+1,x
  clc
  adc SCORES,y
  bcc @bs
  lda #$FF
@bs:
  sta SCORES,y
  inx
  inx
  cpx #16
  bne @b
@bdone:
  lda #0
  sta best_s
  sta best_t
  ldx #0                        ; token index
@t:
  txa
  tay
  lda SCORES,x
  sta sc
  lda tb_hi
  beq @ntb
  sta bp_hi
  lda (bp_lo),y
  clc
  adc sc
  bcc @stb
  lda #$FF
@stb:
  sta sc
@ntb:
  lda qb_hi
  beq @nqb
  sta bp_hi
  lda (bp_lo),y
  clc
  adc sc
  bcc @sqb
  lda #$FF
@sqb:
  sta sc
@nqb:
  lda COUNT_ADDR
  clc
  adc #1
  cmp #SOFT_LEN
  bcc @nlen
  lda CLS_ISEND,y
  beq @nlen
  lda sc
  clc
  adc #LEN_BONUS
  bcc @slen
  lda #$FF
@slen:
  sta sc
@nlen:
  txa                           ; repetition (punct exempt), unrolled ring
  cmp #WORD_DOT
  beq @rdone
  cmp #WORD_COMMA
  beq @rdone
  cmp #WORD_EXCL
  beq @rdone
  cmp #WORD_QM
  beq @rdone
  cmp ring+0
  beq @rhit
  cmp ring+1
  beq @rhit
  cmp ring+2
  beq @rhit
  cmp ring+3
  beq @rhit
  cmp ring+4
  beq @rhit
  cmp ring+5
  beq @rhit
  cmp ring+6
  beq @rhit
  cmp ring+7
  beq @rhit
  bne @rdone
@rhit:
  lda sc
  cmp #(REP_PEN+1)
  bcs @rsub
  lda #1
  sta sc
  bne @rdone
@rsub:
  sec
  sbc #REP_PEN
  sta sc
@rdone:
  lda sc
  sta SCORES,x                  ; keep the final score (top-4 display pass)
  cpx #0
  beq @take                     ; token 0 initializes the argmax
  cmp best_s
  beq @skip
  bcc @skip
@take:
  sta best_s
  stx best_t
@skip:
  inx
  cpx #MODEL_VOCAB_SIZE
  beq @sweep_done
  jmp @t
@sweep_done:
  lda best_t
  jmp emit_token

; fill_top4: destructive top-4 extraction from SCORES into rec (display only)
fill_top4:
  ldx #0
@slot:
  lda #0
  sta best_s
  lda #NO_WORD
  sta best_t
  ldy #0
@scan:
  lda SCORES,y
  beq @nx
  cmp best_s
  beq @nx
  bcc @nx
  sta best_s
  sty best_t
@nx:
  iny
  cpy #MODEL_VOCAB_SIZE
  bne @scan
  lda best_t
  sta rec,x
  cmp #NO_WORD
  beq @pad
  tay
  lda #0
  sta SCORES,y
  lda best_s
@pad:
  sta rec+1,x
  inx
  inx
  cpx #8
  bne @slot
  rts

; score_pick: argmax over the 16-byte record in `rec`.
; score = base + topic_bias[tok] + qtype_bias[tok]
;         + LEN_BONUS if (count+1 >= SOFT_LEN and is_end[tok])
;         - REP_PEN if tok in the last-8 ring (punctuation exempt, floor 1)
; 8-bit saturating; first strict max wins (records sorted base desc).
score_pick:
  lda #0
  sta best_s
  lda #NO_WORD
  sta best_t
  ldx #0
@k:
  lda rec,x
  cmp #NO_WORD
  bne @live
  rts                           ; packed records: first $FF ends the list
@live:
  sta wid                       ; candidate token
  tay
  lda rec+1,x
  sta sc
  lda tb_hi                     ; + topic bias page
  beq @no_tb
  sta bp_hi
  lda (bp_lo),y
  clc
  adc sc
  bcs @sat1
  sta sc
  bcc @no_tb
@sat1:
  lda #$FF
  sta sc
@no_tb:
  lda qb_hi                     ; + qtype bias page
  beq @no_qb
  sta bp_hi
  lda (bp_lo),y
  clc
  adc sc
  bcs @sat2
  sta sc
  bcc @no_qb
@sat2:
  lda #$FF
  sta sc
@no_qb:
  lda COUNT_ADDR                ; + length bonus for sentence-end tokens
  clc
  adc #1
  cmp #SOFT_LEN
  bcc @no_len
  lda CLS_ISEND,y
  beq @no_len
  lda sc
  clc
  adc #LEN_BONUS
  bcs @sat3
  sta sc
  bcc @no_len
@sat3:
  lda #$FF
  sta sc
@no_len:
  lda wid                       ; - repetition penalty (punctuation exempt)
  cmp #WORD_DOT
  beq @cmp
  cmp #WORD_COMMA
  beq @cmp
  cmp #WORD_EXCL
  beq @cmp
  cmp #WORD_QM
  beq @cmp
  stx sx
  ldx #7
@rl:
  cmp ring,x
  beq @rep_hit
  dex
  bpl @rl
  ldx sx
  jmp @cmp
@rep_hit:
  ldx sx
  lda sc
  cmp #(REP_PEN+1)
  bcs @rep_sub
  lda #1
  sta sc
  bne @cmp
@rep_sub:
  sec
  sbc #REP_PEN
  sta sc
@cmp:
  lda sc
  cmp best_s
  beq @next                     ; strict >: first max wins
  bcc @next
  sta best_s
  lda wid
  sta best_t
@next:
  inx
  inx
  cpx #16
  beq @fin
  jmp @k
@fin:
  rts

; set_wptr: A = word ID -> wptr points at its len-prefixed char string
set_wptr:
  tax
  lda woff_lo,x
  clc
  adc #<wdata
  sta wptr_lo
  lda woff_hi,x
  adc #>wdata
  sta wptr_hi
  rts

; is_punct: A = word ID, returns carry set if . , ! ?
is_punct:
  cmp #WORD_DOT
  beq @yes
  cmp #WORD_COMMA
  beq @yes
  cmp #WORD_EXCL
  beq @yes
  cmp #WORD_QM
  beq @yes
  clc
  rts
@yes:
  sec
  rts

; is_stopper: A = word ID, carry set if . ! ?
is_stopper:
  cmp #WORD_DOT
  beq @yes
  cmp #WORD_EXCL
  beq @yes
  cmp #WORD_QM
  beq @yes
  clc
  rts
@yes:
  sec
  rts

; ---------------------------------------------------------------------------
; STATE_GEN frame dispatch
gen_frame:
  lda phase
  cmp #PH_STREAM
  bne @notstream
  jmp stream_step
@notstream:
  cmp #PH_NHID
  bne @nh
  jmp gf_nhid
@nh:
  cmp #PH_NOUT
  bne @no
  jmp gf_nout
@no:
  cmp #PH_NCOND
  bne @nc
  jmp gf_ncond
@nc:
  cmp #PH_DASH1
  beq @d1
  cmp #PH_DASH2
  beq @d2
  cmp #PH_COUNT
  beq @ct
  ; PH_EMIT: start the next word's forward pass immediately
  jmp @emit
@d1:
  jsr fill_top4                 ; v5: extract top-4 from the score sweep
  jsr dash_row1
  lda #PH_DASH2
  sta phase
  rts
@d2:
  jsr dash_row2
  lda #PH_COUNT
  sta phase
  rts
@ct:
  jsr queue_counter
  lda #PH_EMIT
  sta phase
  rts
@emit:
  lda stop_flag
  beq @not_stopped
  jmp gen_finish
@not_stopped:
  lda COUNT_ADDR
  cmp #WORD_BUDGET
  bcc @in_budget
  jmp gen_finish
@in_budget:
  jsr neural_begin              ; v5: load x = [emb(w1); emb(w2)]
  lda #PH_NHID
  sta phase
  rts
; emit_token: A = the chosen token (argmax of the conditioned sweep)
emit_token:
  sta emitted
  ldx ring_i                    ; repetition ring push
  sta ring,x
.ifdef MMC5_MAX
  sta $5C00,x                   ; MMC5_MAX: diagnostic ring copy in ExRAM
.endif
  inx
  txa
  and #7
  sta ring_i
  lda emitted
  ; parity buffer + count
  ldx COUNT_ADDR
  sta WORDBUF,x
  inc COUNT_ADDR
  ; roll the context
  lda w2
  sta w1
  lda emitted
  sta w2
  ; sentence-end stop (mirrors greedy_answer in lib.rs)
  lda COUNT_ADDR
  cmp #MIN_WORDS_STOP
  bcc @nostop
  lda emitted
  jsr is_stopper
  bcc @nostop
  lda #1
  sta stop_flag
@nostop:
  ; layout: clear any blink block, spacing, wrap
  jsr clear_blink
  lda emitted
  jsr set_wptr
  ldy #0
  lda (wptr_lo),y
  sta wlen
  lda first_word
  bne @nospace
  lda emitted
  jsr is_punct
  bcs @nospace
  ; needs a leading space: wrap check uses wlen+1
  lda cur_col
  clc
  adc wlen
  adc #1
  cmp #31
  bcs @wrap
  jsr cursor_advance            ; the space itself (background is blank)
  jmp @begin
@nospace:
  lda cur_col
  clc
  adc wlen
  cmp #31
  bcc @begin
@wrap:
  jsr cursor_newline
@begin:
  lda #0
  sta first_word
  lda #1
  sta wpos
  lda #PH_STREAM
  sta phase
  rts
gen_finish:
  ; ---- liveness watchdog: a short answer is replaced by the emergency
  ; answer, exactly once (mirrors answer_pipeline in lib.rs) ----
  lda COUNT_ADDR
  cmp #MIN_WORDS_STOP
  bcs @done_real
  lda retried
  bne @done_real
  lda #1
  sta retried
  lda #5                        ; R_EMERGENCY
  sta REASON_ADDR
  lda #EMERG_W1
  sta w1
  lda #EMERG_W2
  sta w2
  lda #NO_WORD
  ldx #7
@wri:
  sta ring,x
  dex
  bpl @wri
  lda #0
  sta ring_i
  sta COUNT_ADDR
  sta stop_flag
  sta no_draw
  sta blink_state
  sta phase
  sta think_timer
  lda #1
  sta first_word
  jsr draw_answer_screen        ; clean slate (question echo stays)
  lda #STATE_THINK
  sta state
  lda #1
  sta STATUS_ADDR
  rts
@done_real:
  lda #STATE_DONE
  sta state
  lda #2
  sta STATUS_ADDR
  ; "B NEW QUESTION" hint
  ldx #<str_again
  ldy #>str_again
  jsr queue_string
  rts

; stream up to 4 chars of the current word per frame
stream_step:
  ldx #4
@l:
  lda wpos
  cmp wlen
  beq @last
  bcs @done
@last:
  ldy wpos
  lda (wptr_lo),y
  jsr draw_char
  inc wpos
  dex
  bne @l
  rts
@done:
  lda #PH_DASH1
  sta phase
  rts

draw_char:
  pha
  lda no_draw
  bne @skip
  lda cursor_hi
  sta tmp_hi
  lda cursor_lo
  sta tmp_lo
  pla
  jsr queue_tile
  jmp cursor_advance
@skip:
  pla
  jmp cursor_advance_nodraw

cursor_advance:
cursor_advance_nodraw:
  inc cursor_lo
  bne @nc
  inc cursor_hi
@nc:
  inc cur_col
  rts

cursor_newline:
  lda cur_row
  cmp #18
  bcc @ok
  lda #1
  sta no_draw
  rts
@ok:
  inc cur_row
  ; cursor += 64 - (cur_col - 2)  (skip to col 2 of the row after next... no:
  ; rows advance by 32; from col c to next row col 2 = 34 - c
  lda #34
  sec
  sbc cur_col
  clc
  adc cursor_lo
  sta cursor_lo
  bcc @done
  inc cursor_hi
@done:
  lda #2
  sta cur_col
  rts

clear_blink:
  lda blink_state
  beq @r
  lda #0
  sta blink_state
  lda no_draw
  bne @r
  lda cursor_hi
  sta tmp_hi
  lda cursor_lo
  sta tmp_lo
  lda #0
  jmp queue_tile
@r:
  rts

blink_cursor:
  lda no_draw
  bne @r
  lda nmi_count
  and #$10
  cmp blink_state
  beq @r
  sta blink_state
  lda cursor_hi
  sta tmp_hi
  lda cursor_lo
  sta tmp_lo
  lda blink_state
  beq @off
  lda #TILE_BLOCK
  jmp queue_tile
@off:
  lda #0
  jmp queue_tile
@r:
  rts

; ---------------------------------------------------------------------------
; live candidate display ("the logits"): top-4 of the record just used.
; row 20: c0, c1   row 22: c2, c3 — word (5 chars) + freq (2 digits).
dash_row1:
  lda #$22
  sta tmp_hi
  lda #$86                      ; $2286 = row 20, col 6
  sta tmp_lo
  ldx #0                        ; rec offset 0 (cand 0)
  jsr dash_cand
  lda #$22
  sta tmp_hi
  lda #$90                      ; col 16
  sta tmp_lo
  ldx #2
  jmp dash_cand

dash_row2:
  lda #$22
  sta tmp_hi
  lda #$C6                      ; $22C6 = row 22, col 6
  sta tmp_lo
  ldx #4
  jsr dash_cand
  lda #$22
  sta tmp_hi
  lda #$D0                      ; col 16
  sta tmp_lo
  ldx #6
  jmp dash_cand

; dash_cand: X = rec offset (0/2/4/6), tmp = PPU addr. 8 tiles: 5 word chars
; (space-padded) + space + 2 freq digits.
dash_cand:
  lda rec,x
  cmp #NO_WORD
  bne @word
  ; empty slot: "----  -"
  ldy #0
@dash:
  lda #TILE_DASH
  jsr queue_tile
  inc tmp_lo
  iny
  cpy #5
  bne @dash
  inc tmp_lo                    ; gap
  lda #TILE_DASH
  jsr queue_tile
  inc tmp_lo
  lda #TILE_DASH
  jmp queue_tile
@word:
  stx ti                        ; keep rec offset
  jsr set_wptr
  ldy #0
  lda (wptr_lo),y
  sta tmp                       ; word length (tmp is free during dash phases)
  ldy #1
  ldx #0
@wc:
  cpx tmp
  bcs @pad
  lda (wptr_lo),y
  iny
  bne @put
@pad:
  lda #0
@put:
  jsr queue_tile
  inc tmp_lo
  inx
  cpx #5
  bne @wc
  inc tmp_lo                    ; gap column
  ; freq, capped at 99
  ldx ti
  lda rec+1,x
  cmp #100
  bcc @f
  lda #99
@f:
  ldx #TILE_ZERO
@tens:
  cmp #10
  bcc @td
  sbc #10
  inx
  bne @tens
@td:
  stx dig_t
  clc
  adc #TILE_ZERO
  sta dig_o
  lda dig_t
  jsr queue_tile
  inc tmp_lo
  lda dig_o
  jmp queue_tile

queue_counter:
  ; "WORDS NN" digits at $2308/$2309 (row 24, cols 8-9)
  lda COUNT_ADDR
  ldx #TILE_ZERO
@t:
  cmp #10
  bcc @td
  sbc #10
  inx
  bne @t
@td:
  stx dig_t
  clc
  adc #TILE_ZERO
  sta dig_o
  lda #$23
  sta tmp_hi
  lda #$08
  sta tmp_lo
  lda dig_t
  jsr queue_tile
  lda #$09
  sta tmp_lo
  lda dig_o
  jmp queue_tile

; ---------------------------------------------------------------------------
; STATE_THINK: dots, then start streaming
think_step:
  inc think_timer
  lda think_timer
  cmp #10
  beq @d0
  cmp #20
  beq @d1
  cmp #30
  beq @d2
  cmp #38
  beq @erase
  cmp #40
  beq @go
  rts
@d0:
  lda #0
  ldx #TILE_DOT
  jmp think_cell
@d1:
  lda #1
  ldx #TILE_DOT
  jmp think_cell
@d2:
  lda #2
  ldx #TILE_DOT
  jmp think_cell
@erase:
  lda #0
  ldx #0
  jsr think_cell
  lda #1
  ldx #0
  jsr think_cell
  lda #2
  ldx #0
  jmp think_cell
@go:
  lda #STATE_GEN
  sta state
  lda #1
  sta delay_timer
  lda #PH_EMIT
  sta phase
  rts

think_cell:
  clc
  adc #$02                      ; answer area starts at $2102 (row 8, col 2)
  sta tmp_lo
  lda #$21
  sta tmp_hi
  txa
  jmp queue_tile

; ---------------------------------------------------------------------------
; STATE_DONE: blink; B (or Start/A) -> new question
done_step:
  jsr blink_cursor
  lda pad_pressed
  and #(PAD_A | PAD_B | PAD_START)
  beq @r
  jmp enter_chat
@r:
  rts

; ---------------------------------------------------------------------------
; STATE_KB: the chat input screen
kb_step:
  lda pad_pressed
  and #PAD_UP
  beq @no_up
  jsr kb_erase_marker
  dec kb_row
  bpl @md
  lda #3
  sta kb_row
  bne @md
@no_up:
  lda pad_pressed
  and #PAD_DOWN
  beq @no_dn
  jsr kb_erase_marker
  inc kb_row
  lda kb_row
  cmp #4
  bcc @md
  lda #0
  sta kb_row
  beq @md
@no_dn:
  lda pad_pressed
  and #PAD_LEFT
  beq @no_lt
  jsr kb_erase_marker
  dec kb_col
  bpl @md
  lda #10
  sta kb_col
  bne @md
@no_lt:
  lda pad_pressed
  and #PAD_RIGHT
  beq @no_rt
  jsr kb_erase_marker
  inc kb_col
  lda kb_col
  cmp #11
  bcc @md
  lda #0
  sta kb_col
@md:
  jsr kb_marker_addr
  lda #TILE_SEL
  jsr queue_tile
@no_rt:
  lda pad_pressed
  and #PAD_A
  beq @no_a
  jsr kb_type_char
@no_a:
  lda pad_pressed
  and #PAD_B
  beq @no_b
  lda typed_len
  beq @no_b
  dec typed_len
  lda #$20
  sta tmp_hi
  lda #$A6
  clc
  adc typed_len
  sta tmp_lo
  lda #0
  jsr queue_tile
@no_b:
  lda pad_pressed
  and #PAD_START
  beq @no_st
  lda typed_len
  beq @no_st
  jmp submit_question
@no_st:
  ; live pad display (input diagnostic), row 26
  lda pad_state
  cmp pad_disp
  beq @r
  lda q_count
  cmp #8                        ; need 8 free queue slots
  bcs @r
  lda pad_state
  sta pad_disp
  sta tmp
  lda #0
  sta ti
@pl:
  asl tmp
  lda #TILE_ZERO
  adc #0
  pha
  lda #$23
  sta tmp_hi
  lda ti
  clc
  adc #$46
  sta tmp_lo
  pla
  jsr queue_tile
  inc ti
  lda ti
  cmp #8
  bne @pl
@r:
  rts

kb_marker_addr:
  ldx kb_row
  lda kbrow_mark_hi,x
  sta tmp_hi
  lda kb_col
  asl
  clc
  adc kbrow_mark_lo,x
  sta tmp_lo
  bcc @r
  inc tmp_hi
@r:
  rts

kb_erase_marker:
  jsr kb_marker_addr
  lda #0
  jmp queue_tile

kb_type_char:
  lda typed_len
  cmp #MAX_TYPED
  bcs @r
  ; char = kb_row * 11 + kb_col
  lda kb_row
  asl
  asl
  asl
  sta tmp
  lda kb_row
  asl
  clc
  adc tmp
  adc kb_row
  clc
  adc kb_col
  cmp #43
  bcs @r
  ldx typed_len
  sta TYPED,x
  pha
  lda #$20
  sta tmp_hi
  lda #$A6
  clc
  adc typed_len
  sta tmp_lo
  pla
  jsr queue_tile
  inc typed_len
@r:
  rts

; ---------------------------------------------------------------------------
; submit: parse the typed question into word IDs (decode-bank lookups),
; seed the context exactly like seed_context() in lib.rs, draw the answer
; screen, and start thinking.
submit_question:
  jsr parse_question
  ; ---- seed resolution ladder (mirrors resolve_seed in lib.rs) ----
  ; A: the typed context itself hits a non-empty trigram record
  lda qw_n
  cmp #2
  bcc @ladder
  lda qw1
  sta w1
  lda qw2
  sta w2
  jsr tri_probe                 ; A = tri record byte 0 for (w1, w2)
  cmp #NO_WORD
  beq @ladder
  lda #0                        ; R_EXACT
  jmp @have
@ladder:
  ; B/C/D: scan the route table (rows stored in priority order)
  lda #<(routes_tbl+1)
  sta ptr_lo
  lda #>(routes_tbl+1)
  sta ptr_hi
  ldx #0
@row:
  cpx routes_tbl                ; row count
  bcs @global
  ldy #1
  lda (ptr_lo),y                ; row qtype
  cmp #$FE
  beq @qt_ok
  cmp qt_found
  bne @next
@qt_ok:
  ldy #2
  lda (ptr_lo),y                ; row topic
  cmp #$FE
  beq @t_ok
  cmp topic_found
  bne @next
@t_ok:
  ldy #3
  lda (ptr_lo),y
  sta w1
  iny
  lda (ptr_lo),y
  sta w2
  ldy #0
  lda (ptr_lo),y                ; row level = reason
  jmp @have
@next:
  lda ptr_lo
  clc
  adc #5
  sta ptr_lo
  bcc @ni
  inc ptr_hi
@ni:
  inx
  bne @row
@global:
  ; E: global unknown -> the herd-says family
  lda #NO_WORD
  sta w1
  lda #WORD_QM
  sta w2
  lda #4                        ; R_FALLBACK
@have:
  sta REASON_ADDR
  ; ---- v4 conditioning state ----
  lda topic_g
  beq @no_tp
  clc
  adc #($C0-1)                  ; topic bias page = $C0 + (g-1)
  sta tb_hi
  bne @tp_done
@no_tp:
  lda #0
  sta tb_hi
@tp_done:
  lda qtype_g
  beq @no_qp
  clc
  adc #($D0-1)                  ; qtype bias page = $D0 + (g-1)
  sta qb_hi
  bne @qp_done
@no_qp:
  lda #0
  sta qb_hi
@qp_done:
  ; tone (receipts): empty/nonsense/greeting/technical/normal
  lda qw_n
  bne @tone_known
  lda had_chars
  beq @tone_empty
  lda #2                        ; NONSENSE
  bne @tone_set
@tone_empty:
  lda #1                        ; EMPTY
  bne @tone_set
@tone_known:
  ldx topic_g
  beq @tone_norm
  lda CLS_TONE-1,x              ; per-group tone (1-based index)
  jmp @tone_set
@tone_norm:
  lda #0
@tone_set:
  sta TONE_ADDR
  ; repetition ring + level/bank-cache reset
  lda #NO_WORD
  sta last_bank
  ldx #7
@ri:
  sta ring,x
  dex
  bpl @ri
  lda #0
  sta ring_i
  sta LEVEL_ADDR
  sta retried
@seeded:
  jsr draw_answer_screen
  lda #0
  sta COUNT_ADDR
  sta think_timer
  sta phase
  sta blink_state
  sta stop_flag
  sta no_draw
  lda #1
  sta first_word
  lda #NO_WORD
  sta PROMPTID_ADDR
  lda #WORD_BUDGET
  sta BUDGET_ADDR
  lda #STATE_THINK
  sta state
  lda #1
  sta STATUS_ADDR
  rts

; parse_question: TYPED[0..typed_len) -> rolling (qw1, qw2), qw_n known words
parse_question:
  lda #NO_WORD
  sta qw1
  sta qw2
  lda #$FE                      ; $FE = none (and conveniently = wildcard)
  sta qt_found
  sta topic_found
  lda #0
  sta qw_n
  sta topic_g
  sta qtype_g
  sta had_chars
  ldx #0
@scan:
  cpx typed_len
  bcs @done
  lda TYPED,x
  jsr char_class
  cmp #1
  beq @word
  cmp #2
  beq @punct
  inx
  bne @scan
@punct:
  stx wstart
  lda #1
  sta had_chars
  sta wblen
  jsr lookup_word
  ldx wstart
  inx
  bne @scan
@word:
  stx wstart
  lda #1
  sta had_chars
@wl:
  inx
  cpx typed_len
  bcs @wend
  lda TYPED,x
  jsr char_class
  cmp #1
  beq @wl
@wend:
  txa
  sec
  sbc wstart
  sta wblen
  stx sx
  jsr lookup_word
  ldx sx
  jmp @scan
@done:
  rts

; char_class: A = char ID -> 0 separator, 1 word char, 2 punct word
char_class:
  cmp #0                        ; space
  beq @sep
  cmp #40                       ; '-' acts as a separator
  beq @sep
  cmp #39                       ; apostrophe joins words
  beq @wc
  cmp #37                       ; . , -> punct words
  beq @p
  cmp #38
  beq @p
  cmp #41                       ; ! ?
  beq @p
  cmp #42
  beq @p
  cmp #43
  bcs @sep                      ; out of vocab (shouldn't happen)
  lda #1
  rts
@wc:
  lda #1
  rts
@p:
  lda #2
  rts
@sep:
  lda #0
  rts

; lookup_exact: scan the word table for TYPED[wstart..wstart+wblen).
; Returns carry set + wid on a hit; carry clear on a miss. No side effects.
lookup_exact:
  lda wblen
  cmp #12
  bcs @miss
  lda #0
  sta wid
@next:
  lda wid
  cmp #MODEL_VOCAB_SIZE
  bcs @miss
  lda wid
  jsr set_wptr
  ldy #0
  lda (wptr_lo),y
  cmp wblen
  bne @advance
  lda wblen
  sta tmp
  ldx wstart
  ldy #1
@c:
  lda (wptr_lo),y
  cmp TYPED,x
  bne @advance
  inx
  iny
  dec tmp
  bne @c
  sec
  rts
@advance:
  inc wid
  jmp @next
@miss:
  clc
  rts

; lookup_word: tokenizer v4-stem1 — exact, then strip one trailing S, then
; the longest vocab word (len >= 4) that PREFIXES the typed word. On a hit:
; roll (qw1,qw2), bump qw_n, capture route attrs + conditioning groups.
lookup_word:
  jsr lookup_exact
  bcs @hit
  lda wblen                     ; strip-S retry
  cmp #3
  bcc @prefix
  lda wstart
  clc
  adc wblen
  tax
  dex
  lda TYPED,x
  cmp #19                       ; 'S'
  bne @prefix
  dec wblen
  jsr lookup_exact
  bcs @hit
  inc wblen
@prefix:
  lda #0                        ; best length so far
  sta best_s
  lda #NO_WORD
  sta best_t
  lda #0
  sta wid
@pl:
  lda wid
  cmp #MODEL_VOCAB_SIZE
  bcs @pdone
  lda wid
  jsr set_wptr
  ldy #0
  lda (wptr_lo),y
  cmp #4
  bcc @pnext                    ; prefix must be >= 4 chars
  cmp wblen
  bcs @pnext                    ; and strictly shorter than the typed word
  cmp best_s
  beq @pnext
  bcc @pnext                    ; only strictly longer than current best
  sta sc                        ; candidate length
  sta tmp
  ldx wstart
  ldy #1
@pc:
  lda (wptr_lo),y
  cmp TYPED,x
  bne @pnext
  inx
  iny
  dec tmp
  bne @pc
  lda sc
  sta best_s
  lda wid
  sta best_t
@pnext:
  inc wid
  jmp @pl
@pdone:
  lda best_t
  cmp #NO_WORD
  beq @nope
  sta wid
@hit:
  lda qw2
  sta qw1
  lda wid
  sta qw2
  inc qw_n
  ldx wid                       ; route-attr capture (seed ladder)
  lda word_attr,x
  lsr
  bcc @no_qt
  lda qt_found
  cmp #$FE
  bne @no_qt
  stx qt_found
@no_qt:
  lda word_attr,x
  and #2
  beq @no_tw
  stx topic_found
@no_tw:
  lda CLS_TGROUP,x              ; conditioning groups (classifier bank)
  beq @no_tg
  sta topic_g
@no_tg:
  lda CLS_QGROUP,x
  beq @nope
  lda qtype_g
  bne @nope
  lda CLS_QGROUP,x
  sta qtype_g
@nope:
  rts

; ---------------------------------------------------------------------------
; screens
enter_chat:
  lda #0
  sta $2000
  sta $2001
  sta q_count
  sta typed_len
  sta kb_row
  sta kb_col
  sta state
  sta STATUS_ADDR
  lda #$FF
  sta pad_disp
  jsr clear_nametable
  ldx #<str_title
  ldy #>str_title
  jsr draw_string
  ldx #<str_ask
  ldy #>str_ask
  jsr draw_string
  ldx #<str_kbkeys
  ldy #>str_kbkeys
  jsr draw_string
  ldx #<str_kbhint
  ldy #>str_kbhint
  jsr draw_string
  ldx #<str_pad
  ldy #>str_pad
  jsr draw_string
  ; the 4x11 character grid (cell index == char ID; space key = block tile)
  lda #0
  sta tmp
  ldx #0
@grow:
  bit $2002
  lda kbrow_grid_hi,x
  sta $2006
  lda kbrow_grid_lo,x
  sta $2006
  ldy #11
@gcol:
  lda tmp
  cmp #43
  bcs @blank
  cmp #0
  bne @glyph
  lda #TILE_BLOCK
  bne @put
@blank:
  lda #0
  beq @put
@glyph:
  lda tmp
@put:
  sta $2007
  lda #0
  sta $2007
  inc tmp
  dey
  bne @gcol
  inx
  cpx #4
  bne @grow
  ; marker on (0,0)
  jsr kb_marker_addr
  bit $2002
  lda tmp_hi
  sta $2006
  lda tmp_lo
  sta $2006
  lda #TILE_SEL
  sta $2007
  ; llama, rows 18-20 cols 26-28
  lda #TILE_LLAMA
  sta tmp
  ldx #0
@ll:
  bit $2002
  lda llama_hi,x
  sta $2006
  lda llama_lo,x
  sta $2006
  lda tmp
  sta $2007
  clc
  adc #1
  sta $2007
  adc #1
  sta $2007
  adc #1
  sta tmp
  inx
  cpx #3
  bne @ll
  jmp screen_on

draw_answer_screen:
  lda #0
  sta $2000
  sta $2001
  sta q_count
  jsr clear_nametable
  ldx #<str_title
  ldy #>str_title
  jsr draw_string
  ldx #<str_q
  ldy #>str_q
  jsr draw_string
  ldx #<str_top
  ldy #>str_top
  jsr draw_string
  ldx #<str_words
  ldy #>str_words
  jsr draw_string
  ; echo the typed question on row 4 (col 4..)
  bit $2002
  lda #$20
  sta $2006
  lda #$84
  sta $2006
  ldx #0
@qe:
  cpx typed_len
  bcs @qdone
  lda TYPED,x
  sta $2007
  inx
  bne @qe
@qdone:
  ; cursor at the answer area start ($2102 = row 8, col 2)
  lda #$02
  sta cursor_lo
  sta cur_col
  lda #$21
  sta cursor_hi
  lda #8
  sta cur_row
  jmp screen_on

screen_on:
  bit $2002
  lda #0
  sta $2005
  sta $2005
  lda soft2000
  sta $2000
  lda #%00001010
  sta $2001
  rts

clear_nametable:
  bit $2002
  lda #$20
  sta $2006
  lda #$00
  sta $2006
  ldx #4
  ldy #0
  lda #0
@l:
  sta $2007
  iny
  bne @l
  dex
  bne @l
  rts

; draw_string: X/Y = lo/hi of record (ppu hi, ppu lo, len, chars...)
draw_string:
  stx ptr_lo
  sty ptr_hi
  bit $2002
  ldy #0
  lda (ptr_lo),y
  sta $2006
  iny
  lda (ptr_lo),y
  sta $2006
  iny
  lda (ptr_lo),y
  sta tmp
  iny
@l:
  lda (ptr_lo),y
  sta $2007
  iny
  dec tmp
  bne @l
  rts

; queue_string: same record, but via the NMI tile queue (rendering on)
queue_string:
  stx ptr_lo
  sty ptr_hi
  ldy #0
  lda (ptr_lo),y
  sta tmp_hi
  iny
  lda (ptr_lo),y
  sta tmp_lo
  iny
  lda (ptr_lo),y
  sta ti
  iny
@l:
  lda (ptr_lo),y
  sty sx
  jsr queue_tile
  ldy sx
  inc tmp_lo
  iny
  dec ti
  bne @l
  rts

; queue_tile: A = tile, tmp_hi/lo = PPU address (max 16/frame).
; PRESERVES X (several callers loop on X), and hard-clamps the queue so an
; over-eager frame can drop tiles but can never corrupt memory.
queue_tile:
  stx qsave_x
  ldx q_count
  cpx #16
  bcs @full
  sta q_tile,x
  lda tmp_hi
  sta q_hi,x
  lda tmp_lo
  sta q_lo,x
  inc q_count
@full:
  ldx qsave_x
  rts

step_lfsr:
  lda lfsr
  lsr
  bcc @s
  eor #$B8
@s:
  sta lfsr
  rts

wait_nmi:
  lda nmi_count
@w:
  cmp nmi_count
  beq @w
  rts

read_pad:
  lda pad_state
  sta pad_prev
  lda #1
  sta $4016
  lda #0
  sta $4016
  ldx #8
@l:
  lda $4016
  lsr a
  rol pad_state
  dex
  bne @l
  lda pad_prev
  eor #$FF
  and pad_state
  sta pad_pressed
  rts

; --- NMI: drain the tile queue, reset scroll --------------------------------
nmi:
  pha
  txa
  pha
  tya
  pha
  inc nmi_count
  lda q_count
  beq @no_q
  ldx #0
@q:
  bit $2002
  lda q_hi,x
  sta $2006
  lda q_lo,x
  sta $2006
  lda q_tile,x
  sta $2007
  inx
  cpx q_count
  bne @q
  lda #0
  sta q_count
@no_q:
  bit $2002
  lda #0
  sta $2005
  sta $2005
  lda soft2000
  sta $2000
  pla
  tay
  pla
  tax
  pla
irq:
  rti

; ---------------------------------------------------------------------------
.segment "RODATA"

palette_data:
  .repeat 8
    .byte $0F, $30, $30, $30
  .endrepeat

; keyboard grid rows 10/12/14/16 (glyphs col 5, marker col-1)
kbrow_grid_hi: .byte $21, $21, $21, $22
kbrow_grid_lo: .byte $45, $85, $C5, $05
kbrow_mark_hi: .byte $21, $21, $21, $22
kbrow_mark_lo: .byte $44, $84, $C4, $04
; llama rows 18-20, col 26
llama_hi:   .byte $22, $22, $22
llama_lo:   .byte $5A, $7A, $9A

str_title:  strrec $2048, "NANOCAMELID NES"
str_ask:    strrec $20A2, "ASK"
str_kbkeys: strrec $22C2, "A ADD  B DEL  START ASK"
str_kbhint: strrec $2302, "THE HERD ANSWERS QUESTIONS"
str_pad:    strrec $2342, "PAD"
str_q:      strrec $2082, "Q"
str_top:    strrec $2282, "TOP"
str_words:  strrec $2302, "WORDS 00"
str_again:  strrec $230E, "B NEW QUESTION"

; ---------------------------------------------------------------------------
.segment "VECTORS"
  .addr nmi, reset, irq

.segment "CHARS"
  .incbin "chr.bin"
