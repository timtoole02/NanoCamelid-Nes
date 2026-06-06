//! NanoCamelid NES — shared model format, tokenizers, font, and generator.
//!
//! v2: a WORD-level trigram chat model. Word IDs are bytes (vocab ≤ 224),
//! the dense trigram table is (w1 * 1024 + w2 * 4) = 224 KiB across 14
//! UOROM banks, with a bigram fallback table in the fixed bank. A 2-word
//! context carries real meaning, which is what makes Q&A work: the question
//! "WHY DO LLAMAS HUM?" seeds on (HUM, ?) and the model walks straight into
//! the trained answer. The claim stays narrow: the 6502 runs the whole
//! next-word loop locally; this crate is the source of truth that proves it.

use std::collections::HashMap;

pub mod cpu6502;
pub mod nes;
pub mod neural;

/// Display characters. CHAR id == CHR tile index (typing grid uses these).
pub const VOCAB: &str = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'-!?";

/// Hard caps baked into the ROM layout.
pub const MAX_WORDS: usize = 248; // 124 trigram banks; bank 124 = neural weights
pub const MAX_WORD_LEN: usize = 11;
pub const TOP_K: usize = 8; // (next, freq) pairs per record
pub const REC_LEN: usize = 16; // 8 x (next_word, freq), $FF/0 padded
pub const WORD_BUDGET: usize = 32; // generated words per answer (hard max)
pub const MIN_WORDS_BEFORE_STOP: usize = 6; // sentence-end stops only after this
pub const MIN_ANSWER_WORDS: usize = 6; // watchdog: shorter -> emergency reseed
pub const MAX_TYPED_CHARS: usize = 24; // on-screen keyboard line length

/// Route resolution reasons (parity-checked: the ROM reports its pick at $07F4).
pub const R_EXACT: u8 = 0; //   A: typed context hits a non-empty trigram record
pub const R_QT_TOPIC: u8 = 1; // B: qtype + topic route
pub const R_TOPIC: u8 = 2; //   C: topic-only route
pub const R_QTYPE: u8 = 3; //   D: qtype-only route
pub const R_FALLBACK: u8 = 4; // E: global unknown -> bigram("?") family
pub const R_EMERGENCY: u8 = 5; // F: watchdog reseed (answer was too short)

pub const ATTR_QTYPE: u8 = 0x01;
pub const ATTR_TOPIC: u8 = 0x02;
/// MMC5 layout: 16-byte record per (w1, w2); 4 KiB per w1; two w1 per 8 KiB
/// bank; banks 0..124 hold the table. record = bank(w1>>1) : (w1&1)*4096 + w2*16.
pub const TRI_LEN: usize = MAX_WORDS * 256 * REC_LEN; // 1,024,000 = 125 x 8 KiB
pub const BI_LEN: usize = MAX_WORDS * REC_LEN; // 4000, decode bank
pub const NO_WORD: u8 = 0xFF;

pub const HEADER_LEN: usize = 32;
pub const MAGIC: &[u8; 4] = b"NNW5";
pub const TOKENIZER_VERSION: &str = "v4-stem1";
pub const MODEL_FORMAT_VERSION: &str = "NNW5-neural1";

/// v4 scored decoding parameters (mirrored byte-for-byte on the 6502).
pub const N_TOPICS_MAX: usize = 16;
pub const N_QTYPES_MAX: usize = 11;
pub const REP_WINDOW: usize = 8; // repetition ring size
pub const REP_PEN: u8 = 24; // score penalty for recently used tokens
pub const LEN_BONUS: u8 = 20; // bonus for sentence-end tokens past SOFT_LEN
pub const SOFT_LEN: usize = 14; // soft answer-length target (words)
pub const BASE_MAX: u8 = 63; // quantized record base scores 1..=63
pub const BIAS_MAX: u8 = 15; // quantized conditioning biases 0..=15
pub const CLS_LEN: usize = 8192; // classifier bank image (MMC5 bank 125, $C000)

/// Generation-time backoff levels (max over the answer reported at $07F5).
pub const L_TRI: u8 = 0;
pub const L_BI: u8 = 1;
pub const L_TOPIC: u8 = 2;
pub const L_QTYPE: u8 = 3;
pub const L_GLOBAL: u8 = 4;
pub const L_NEURAL: u8 = 5; // v5: full-vocab neural decode (the only gen level)

/// Tone (receipts + fallback flavor), reported at $07F6.
pub const TONE_NORMAL: u8 = 0;
pub const TONE_EMPTY: u8 = 1;
pub const TONE_NONSENSE: u8 = 2;
pub const TONE_GREETING: u8 = 3;
pub const TONE_TECHNICAL: u8 = 4;

/// Classifier bank layout (offsets inside the 8 KiB bank at $C000):
pub const CLS_TOPIC_BIAS: usize = 0x0000; // 16 pages x 256
pub const CLS_QTYPE_BIAS: usize = 0x1000; // 11 pages x 256
pub const CLS_IS_END: usize = 0x1B00;
pub const CLS_TGROUP: usize = 0x1C00; // word -> topic group + 1 (0 = none)
pub const CLS_QGROUP: usize = 0x1D00; // word -> qtype group + 1 (0 = none)
pub const CLS_TOPIC_RECS: usize = 0x1E00; // 16 x 16-byte records
pub const CLS_QTYPE_RECS: usize = 0x1F00; // 11 x 16-byte records
pub const CLS_GLOBAL_REC: usize = 0x1FC0; // 1 x 16-byte record
pub const CLS_TONE: usize = 0x1FE0; // tone per topic group (0/3/4)

/// One route-table row. qtype/topic $FE = wildcard. Rows are stored in
/// resolution priority order (level B rows, then C, then D).
#[derive(Clone, Copy, Debug)]
pub struct RouteRow {
    pub level: u8, // R_QT_TOPIC / R_TOPIC / R_QTYPE
    pub qtype: u8,
    pub topic: u8,
    pub seed1: u8,
    pub seed2: u8,
}

pub const WILDCARD: u8 = 0xFE;

/// CHR tiles (same as v1).
pub const TILE_SELECTOR: u8 = 62;
pub const TILE_BLOCK: u8 = 63;
pub const TILE_LLAMA: u8 = 48;

// ---------------------------------------------------------------------------
// Character tokenizer (for the typing grid + display)
// ---------------------------------------------------------------------------

pub fn char_ids(text: &str) -> Vec<u8> {
    let mut out = Vec::new();
    for ch in text.chars() {
        let ch = if ch.is_whitespace() { ' ' } else { ch.to_ascii_uppercase() };
        if let Some(id) = VOCAB.find(ch) {
            out.push(id as u8);
        }
    }
    out
}

pub fn chars_from_ids(ids: &[u8]) -> String {
    let v: Vec<char> = VOCAB.chars().collect();
    ids.iter().filter_map(|&t| v.get(t as usize)).collect()
}

// ---------------------------------------------------------------------------
// Word tokenizer
// ---------------------------------------------------------------------------

/// Split normalized text into word strings. Punctuation `. , ! ?` are their
/// own words; words are A-Z0-9' runs. This must match the 6502 parser.
pub fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        let ch = ch.to_ascii_uppercase();
        if ch.is_ascii_alphanumeric() || ch == '\'' {
            cur.push(ch);
        } else {
            if !cur.is_empty() {
                words.push(std::mem::take(&mut cur));
            }
            if matches!(ch, '.' | ',' | '!' | '?') {
                words.push(ch.to_string());
            }
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

pub struct WordVocab {
    pub words: Vec<String>, // id == index, sorted lexicographically
}

impl WordVocab {
    pub fn id(&self, w: &str) -> Option<u8> {
        self.words.binary_search(&w.to_string()).ok().map(|i| i as u8)
    }

    /// Serialized form for both hashing and the ROM fixed bank:
    /// woff_lo[224], woff_hi[224], then per word: len, char_ids...
    /// Offsets are relative to the start of the data area; unused slots
    /// point at offset 0 with the convention that ids >= len(words) never occur.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();
        let mut offs = Vec::new();
        for w in &self.words {
            offs.push(data.len() as u16);
            let ids = char_ids(w);
            data.push(ids.len() as u8);
            data.extend_from_slice(&ids);
        }
        offs.resize(MAX_WORDS, 0);
        let mut out = Vec::new();
        for o in &offs {
            out.push((*o & 0xFF) as u8);
        }
        for o in &offs {
            out.push((*o >> 8) as u8);
        }
        out.extend_from_slice(&data);
        out
    }
}

/// Tokenize text to word IDs, silently dropping unknown words (mirrors the
/// NES submit parser, which skips words it can't find in the table).
pub fn word_ids(vocab: &WordVocab, text: &str) -> Vec<u8> {
    split_words(text).iter().filter_map(|w| vocab.id(w)).collect()
}

pub fn words_to_text(vocab: &WordVocab, ids: &[u8]) -> String {
    let mut out = String::new();
    for &id in ids {
        let w = match vocab.words.get(id as usize) {
            Some(w) => w,
            None => continue,
        };
        let is_punct = matches!(w.as_str(), "." | "," | "!" | "?");
        if !out.is_empty() && !is_punct {
            out.push(' ');
        }
        out.push_str(w);
    }
    out
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

pub struct Model {
    pub vocab_size: u8,
    pub k: u8,
    pub qmark_id: u8, // seed of last resort
    pub emergency: (u8, u8), // watchdog reseed context
    pub tri: Vec<u8>, // TRI_LEN
    pub bi: Vec<u8>,  // BI_LEN
    pub vocab: WordVocab,
    pub attr: [u8; 256], // per-word ATTR_QTYPE / ATTR_TOPIC bits (route ladder)
    pub routes: Vec<RouteRow>,
    pub cls: Vec<u8>, // the 8 KiB classifier bank image (see CLS_* offsets)
    pub neural: neural::NeuralInt, // v5: the trained quantized network (bank 124)
    pub topic_names: Vec<String>,
    pub topic_tech: Vec<bool>,
    pub qtype_names: Vec<String>,
}

impl Model {
    pub fn routes_bytes(&self) -> Vec<u8> {
        let mut out = vec![self.routes.len() as u8];
        for r in &self.routes {
            out.extend_from_slice(&[r.level, r.qtype, r.topic, r.seed1, r.seed2]);
        }
        out
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let words = self.vocab.to_bytes();
        let routes = self.routes_bytes();
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.push(5); // version
        out.push(self.vocab_size);
        out.push(self.k);
        out.push(0); // mode greedy
        out.push(self.qmark_id);
        out.push(WORD_BUDGET as u8);
        out.push(MIN_WORDS_BEFORE_STOP as u8);
        out.push(self.emergency.0);
        out.push(self.emergency.1);
        out.extend_from_slice(&[0u8; 3]);
        out.extend_from_slice(&(self.tri.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.bi.len() as u32).to_le_bytes());
        out.extend_from_slice(&(words.len() as u32).to_le_bytes());
        out.extend_from_slice(&[0u8; HEADER_LEN - 28]);
        assert_eq!(out.len(), HEADER_LEN);
        out.extend_from_slice(&self.tri);
        out.extend_from_slice(&self.bi);
        out.extend_from_slice(&words);
        out.extend_from_slice(&self.attr);
        out.extend_from_slice(&routes);
        out.extend_from_slice(&self.cls);
        out.extend_from_slice(&self.neural.bank_image());
        // names (for receipts after roundtrip): count + len-prefixed strings
        out.push(self.topic_names.len() as u8);
        for (n, t) in self.topic_names.iter().zip(&self.topic_tech) {
            out.push(n.len() as u8 | if *t { 0x80 } else { 0 });
            out.extend_from_slice(n.as_bytes());
        }
        out.push(self.qtype_names.len() as u8);
        for n in &self.qtype_names {
            out.push(n.len() as u8);
            out.extend_from_slice(n.as_bytes());
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Model> {
        anyhow::ensure!(bytes.len() > HEADER_LEN, "model.bin too short");
        anyhow::ensure!(&bytes[0..4] == MAGIC, "bad magic (want NNW5)");
        anyhow::ensure!(bytes[4] == 5, "unsupported version");
        let vocab_size = bytes[5];
        let qmark_id = bytes[8];
        let emergency = (bytes[11], bytes[12]);
        let tri_len = u32::from_le_bytes(bytes[16..20].try_into()?) as usize;
        let bi_len = u32::from_le_bytes(bytes[20..24].try_into()?) as usize;
        let words_len = u32::from_le_bytes(bytes[24..28].try_into()?) as usize;
        anyhow::ensure!(tri_len == TRI_LEN && bi_len == BI_LEN, "table size mismatch");
        let tri = bytes[HEADER_LEN..HEADER_LEN + tri_len].to_vec();
        let bi = bytes[HEADER_LEN + tri_len..HEADER_LEN + tri_len + bi_len].to_vec();
        let wb = &bytes[HEADER_LEN + tri_len + bi_len..HEADER_LEN + tri_len + bi_len + words_len];
        // decode the word table
        let mut words = Vec::new();
        let data = &wb[MAX_WORDS * 2..];
        for i in 0..vocab_size as usize {
            let off = wb[i] as usize | ((wb[MAX_WORDS + i] as usize) << 8);
            let len = data[off] as usize;
            let ids: Vec<u8> = data[off + 1..off + 1 + len].to_vec();
            words.push(chars_from_ids(&ids));
        }
        // attr + routes
        let rest = &bytes[HEADER_LEN + tri_len + bi_len + words_len..];
        anyhow::ensure!(rest.len() >= 257, "missing attr/routes sections");
        let mut attr = [0u8; 256];
        attr.copy_from_slice(&rest[0..256]);
        let n_routes = rest[256] as usize;
        let mut routes = Vec::new();
        for i in 0..n_routes {
            let r = &rest[257 + i * 5..257 + i * 5 + 5];
            routes.push(RouteRow { level: r[0], qtype: r[1], topic: r[2], seed1: r[3], seed2: r[4] });
        }
        let mut off = 257 + n_routes * 5;
        let cls = rest[off..off + CLS_LEN].to_vec();
        off += CLS_LEN;
        let neural = neural::NeuralInt::from_bank(&rest[off..off + neural::NEURAL_LEN], vocab_size as usize);
        off += neural::NEURAL_LEN;
        let mut topic_names = Vec::new();
        let mut topic_tech = Vec::new();
        let nt = rest[off] as usize;
        off += 1;
        for _ in 0..nt {
            let lt = rest[off];
            let l = (lt & 0x7F) as usize;
            topic_tech.push(lt & 0x80 != 0);
            topic_names.push(String::from_utf8_lossy(&rest[off + 1..off + 1 + l]).into_owned());
            off += 1 + l;
        }
        let mut qtype_names = Vec::new();
        let nq = rest[off] as usize;
        off += 1;
        for _ in 0..nq {
            let l = rest[off] as usize;
            qtype_names.push(String::from_utf8_lossy(&rest[off + 1..off + 1 + l]).into_owned());
            off += 1 + l;
        }
        Ok(Model {
            vocab_size,
            k: bytes[6],
            qmark_id,
            emergency,
            tri,
            bi,
            vocab: WordVocab { words },
            attr,
            routes,
            cls,
            neural,
            topic_names,
            topic_tech,
            qtype_names,
        })
    }

    // --- classifier bank accessors (identical addressing to the 6502) ---
    pub fn topic_bias(&self, topic: Option<u8>, tok: u8) -> u8 {
        match topic {
            Some(t) => self.cls[CLS_TOPIC_BIAS + t as usize * 256 + tok as usize],
            None => 0,
        }
    }
    pub fn qtype_bias(&self, qt: Option<u8>, tok: u8) -> u8 {
        match qt {
            Some(q) => self.cls[CLS_QTYPE_BIAS + q as usize * 256 + tok as usize],
            None => 0,
        }
    }
    pub fn is_end_tok(&self, tok: u8) -> bool {
        self.cls[CLS_IS_END + tok as usize] != 0
    }
    pub fn tgroup(&self, wid: u8) -> Option<u8> {
        match self.cls[CLS_TGROUP + wid as usize] {
            0 => None,
            g => Some(g - 1),
        }
    }
    pub fn qgroup(&self, wid: u8) -> Option<u8> {
        match self.cls[CLS_QGROUP + wid as usize] {
            0 => None,
            g => Some(g - 1),
        }
    }
    pub fn topic_rec(&self, t: u8) -> &[u8] {
        &self.cls[CLS_TOPIC_RECS + t as usize * 16..CLS_TOPIC_RECS + t as usize * 16 + 16]
    }
    pub fn qtype_rec(&self, q: u8) -> &[u8] {
        &self.cls[CLS_QTYPE_RECS + q as usize * 16..CLS_QTYPE_RECS + q as usize * 16 + 16]
    }
    pub fn global_rec(&self) -> &[u8] {
        &self.cls[CLS_GLOBAL_REC..CLS_GLOBAL_REC + 16]
    }

    /// 16-byte transition record: 8 x (next_word, freq), exactly the bytes
    /// the 6502 sees at $8000 + (w1&1)*4096 + w2*16 after switching $5114.
    pub fn tri_rec(&self, w1: u8, w2: u8) -> &[u8] {
        let base = w1 as usize * 4096 + w2 as usize * REC_LEN;
        &self.tri[base..base + REC_LEN]
    }

    pub fn bi_rec(&self, w: u8) -> &[u8] {
        let base = w as usize * REC_LEN;
        &self.bi[base..base + REC_LEN]
    }
}

/// Build the word model from a corpus + routes + topic groups. Returns (model, stats).
pub fn build_model(corpus: &str, routes_src: &str, topics_src: &str) -> anyhow::Result<(Model, usize, usize)> {
    let toks = split_words(corpus);
    anyhow::ensure!(toks.len() >= 3, "corpus too small");
    let mut uniq: Vec<String> = toks.clone();
    uniq.sort();
    uniq.dedup();
    anyhow::ensure!(
        uniq.len() <= MAX_WORDS,
        "vocabulary too large: {} words (max {MAX_WORDS})",
        uniq.len()
    );
    for w in &uniq {
        anyhow::ensure!(
            w.chars().count() <= MAX_WORD_LEN,
            "word too long for the display: {w:?} (max {MAX_WORD_LEN} chars)"
        );
        for ch in w.chars() {
            anyhow::ensure!(VOCAB.contains(ch), "char {ch:?} of {w:?} not in display vocab");
        }
    }
    let vocab = WordVocab { words: uniq };
    let qmark_id = vocab
        .id("?")
        .ok_or_else(|| anyhow::anyhow!("corpus must contain at least one question mark"))?;

    let ids: Vec<u8> = toks.iter().map(|w| vocab.id(w).unwrap()).collect();

    let mut tri_counts: HashMap<(u8, u8), HashMap<u8, u32>> = HashMap::new();
    let mut bi_counts: HashMap<u8, HashMap<u8, u32>> = HashMap::new();
    for w in ids.windows(3) {
        *tri_counts.entry((w[0], w[1])).or_default().entry(w[2]).or_default() += 1;
    }
    for w in ids.windows(2) {
        *bi_counts.entry(w[0]).or_default().entry(w[1]).or_default() += 1;
    }

    // 16-byte record: 8 x (next_word, freq u8 saturating), padded (0xFF, 0).
    // Deterministic order: freq desc, then PUNCTUATION LOSES TIES (a tied
    // '?' or '.' never beats a real word — this kills leading-'?' artifacts
    // on truncated questions), then id asc. Byte 0 is the greedy pick.
    let punct_ids: Vec<u8> = [".", ",", "!", "?"]
        .iter()
        .filter_map(|p| vocab.id(p))
        .collect();
    let record = |m: &HashMap<u8, u32>| -> [u8; REC_LEN] {
        let mut c: Vec<(u8, u32)> = m.iter().map(|(&t, &n)| (t, n)).collect();
        c.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then(punct_ids.contains(&a.0).cmp(&punct_ids.contains(&b.0)))
                .then(a.0.cmp(&b.0))
        });
        let maxn = c.first().map(|&(_, n)| n).unwrap_or(1).max(1);
        let mut out = [0u8; REC_LEN];
        for i in 0..TOP_K {
            match c.get(i) {
                Some(&(t, n)) => {
                    out[i * 2] = t;
                    // base score quantized 1..=63 relative to the record max
                    out[i * 2 + 1] = (1 + (BASE_MAX as u32 - 1) * n / maxn).min(BASE_MAX as u32) as u8;
                }
                None => {
                    out[i * 2] = NO_WORD;
                    out[i * 2 + 1] = 0;
                }
            }
        }
        out
    };

    let mut tri = Vec::with_capacity(TRI_LEN);
    for _ in 0..MAX_WORDS * 256 {
        tri.extend_from_slice(&[NO_WORD, 0, NO_WORD, 0, NO_WORD, 0, NO_WORD, 0, NO_WORD, 0, NO_WORD, 0, NO_WORD, 0, NO_WORD, 0]);
    }
    for (&(w1, w2), nexts) in &tri_counts {
        let base = w1 as usize * 4096 + w2 as usize * REC_LEN;
        tri[base..base + REC_LEN].copy_from_slice(&record(nexts));
    }
    let mut bi = vec![0u8; BI_LEN];
    for chunk in bi.chunks_mut(2) {
        chunk[0] = NO_WORD;
    }
    for (&w, nexts) in &bi_counts {
        let base = w as usize * REC_LEN;
        bi[base..base + REC_LEN].copy_from_slice(&record(nexts));
    }

    let n_tri = tri_counts.len();
    let n_bi = bi_counts.len();
    let vocab_size = vocab.words.len() as u8;

    // --- always-answer invariant: every vocab word has a bigram successor ---
    for wid in 0..vocab_size {
        anyhow::ensure!(
            bi[wid as usize * REC_LEN] != NO_WORD,
            "word {:?} has no bigram successor — generation could stall",
            vocab.words[wid as usize]
        );
    }

    // --- parse the route table (single source of truth: routes.txt) ---
    let mut attr = [0u8; 256];
    let mut rows_b = Vec::new();
    let mut rows_c = Vec::new();
    let mut rows_d = Vec::new();
    let mut emergency: Option<(u8, u8)> = None;
    for (ln, raw) in routes_src.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let (lhs, rhs) = line
            .split_once("->")
            .ok_or_else(|| anyhow::anyhow!("routes.txt:{}: missing ->", ln + 1))?;
        let l: Vec<&str> = lhs.split_whitespace().collect();
        let r: Vec<&str> = rhs.split_whitespace().collect();
        let id_of = |w: &str| -> anyhow::Result<u8> {
            vocab
                .id(&w.to_ascii_uppercase())
                .ok_or_else(|| anyhow::anyhow!("routes.txt:{}: {w:?} not in vocabulary", ln + 1))
        };
        if l == ["EMERGENCY"] {
            anyhow::ensure!(r.len() == 2, "routes.txt:{}: EMERGENCY needs 2 seed words", ln + 1);
            emergency = Some((id_of(r[0])?, id_of(r[1])?));
            continue;
        }
        anyhow::ensure!(l.len() == 2 && r.len() == 2, "routes.txt:{}: want QT TOPIC -> S1 S2", ln + 1);
        let qt = if l[0] == "*" { WILDCARD } else { id_of(l[0])? };
        let topic = if l[1] == "*" { WILDCARD } else { id_of(l[1])? };
        let (seed1, seed2) = (id_of(r[0])?, id_of(r[1])?);
        let (level, bucket) = match (qt, topic) {
            (WILDCARD, WILDCARD) => anyhow::bail!("routes.txt:{}: double wildcard", ln + 1),
            (WILDCARD, _) => (R_TOPIC, &mut rows_c),
            (_, WILDCARD) => (R_QTYPE, &mut rows_d),
            _ => (R_QT_TOPIC, &mut rows_b),
        };
        if qt != WILDCARD {
            attr[qt as usize] |= ATTR_QTYPE;
        }
        if topic != WILDCARD {
            attr[topic as usize] |= ATTR_TOPIC;
        }
        bucket.push(RouteRow { level, qtype: qt, topic, seed1, seed2 });
    }
    let emergency = emergency
        .ok_or_else(|| anyhow::anyhow!("routes.txt must define an EMERGENCY seed"))?;
    let mut routes = rows_b;
    routes.extend(rows_c);
    routes.extend(rows_d);
    anyhow::ensure!(routes.len() <= 255, "too many route rows");

    // ---- v4 classifier bank: topic groups, qtype groups, biases, records ----
    let qtype_list = ["WHY", "HOW", "WHAT", "WHO", "WHERE", "WHEN", "CAN", "DO", "DOES", "ARE", "IS"];
    let mut topic_names = Vec::new();
    let mut topic_tech = Vec::new();
    let mut topic_members: Vec<Vec<u8>> = Vec::new();
    for (ln, raw) in topics_src.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let rest = line
            .strip_prefix("TOPIC ")
            .ok_or_else(|| anyhow::anyhow!("topics.txt:{}: want TOPIC name [TECH]: words", ln + 1))?;
        let (head, members) = rest
            .split_once(':')
            .ok_or_else(|| anyhow::anyhow!("topics.txt:{}: missing ':'", ln + 1))?;
        let hp: Vec<&str> = head.split_whitespace().collect();
        let tech = hp.len() > 1 && hp[1] == "TECH";
        topic_names.push(hp[0].to_string());
        topic_tech.push(tech);
        let mut ids = Vec::new();
        for w in members.split_whitespace() {
            ids.push(
                vocab
                    .id(w)
                    .ok_or_else(|| anyhow::anyhow!("topics.txt:{}: {w:?} not in vocabulary", ln + 1))?,
            );
        }
        topic_members.push(ids);
    }
    anyhow::ensure!(topic_members.len() <= N_TOPICS_MAX, "too many topic groups");

    let mut cls = vec![0u8; CLS_LEN];
    // group maps
    for (g, members) in topic_members.iter().enumerate() {
        for &w in members {
            cls[CLS_TGROUP + w as usize] = g as u8 + 1;
        }
    }
    let mut qtype_names = Vec::new();
    for (g, qw) in qtype_list.iter().enumerate() {
        qtype_names.push(qw.to_string());
        if let Some(id) = vocab.id(qw) {
            cls[CLS_QGROUP + id as usize] = g as u8 + 1;
        }
    }
    // sentence segmentation over the token stream
    let mut sentences: Vec<Vec<u8>> = Vec::new();
    let mut cur = Vec::new();
    let enders: Vec<u8> = [".", "!", "?"].iter().filter_map(|p| vocab.id(p)).collect();
    for &t in &ids {
        cur.push(t);
        if enders.contains(&t) {
            sentences.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        sentences.push(cur);
    }
    let quant_bias = |counts: &HashMap<u8, u32>, page: &mut [u8]| {
        let maxn = counts.values().copied().max().unwrap_or(1).max(1);
        for (&t, &n) in counts {
            page[t as usize] = ((BIAS_MAX as u32 * n + maxn - 1) / maxn).min(BIAS_MAX as u32) as u8;
        }
    };
    // topic biases: token stats over sentences containing a member word
    let mut topic_counts_all: Vec<HashMap<u8, u32>> = vec![HashMap::new(); topic_members.len()];
    for sent in &sentences {
        for (g, members) in topic_members.iter().enumerate() {
            if sent.iter().any(|t| members.contains(t)) {
                for &t in sent {
                    *topic_counts_all[g].entry(t).or_default() += 1;
                }
            }
        }
    }
    for (g, counts) in topic_counts_all.iter().enumerate() {
        let (a, b) = (CLS_TOPIC_BIAS + g * 256, CLS_TOPIC_BIAS + g * 256 + 256);
        quant_bias(counts, &mut cls[a..b]);
    }
    // qtype biases: token stats over the sentence FOLLOWING a '?' sentence
    // whose first three words contain the qtype word
    let mut qtype_counts_all: Vec<HashMap<u8, u32>> = vec![HashMap::new(); qtype_list.len()];
    let qm = vocab.id("?");
    for win in sentences.windows(2) {
        let (q, a) = (&win[0], &win[1]);
        if q.last().copied() != qm {
            continue;
        }
        for (g, qw) in qtype_list.iter().enumerate() {
            if let Some(id) = vocab.id(qw) {
                if q.iter().take(3).any(|&t| t == id) {
                    for &t in a {
                        *qtype_counts_all[g].entry(t).or_default() += 1;
                    }
                }
            }
        }
    }
    for (g, counts) in qtype_counts_all.iter().enumerate() {
        let (a, b) = (CLS_QTYPE_BIAS + g * 256, CLS_QTYPE_BIAS + g * 256 + 256);
        quant_bias(counts, &mut cls[a..b]);
    }
    // is_end + conditioned fallback records + global record
    for &t in &enders {
        cls[CLS_IS_END + t as usize] = 1;
    }
    for (g, counts) in topic_counts_all.iter().enumerate() {
        let r = record(counts);
        cls[CLS_TOPIC_RECS + g * 16..CLS_TOPIC_RECS + g * 16 + 16].copy_from_slice(&r);
    }
    for (g, counts) in qtype_counts_all.iter().enumerate() {
        let r = record(counts);
        cls[CLS_QTYPE_RECS + g * 16..CLS_QTYPE_RECS + g * 16 + 16].copy_from_slice(&r);
    }
    let mut uni: HashMap<u8, u32> = HashMap::new();
    for &t in &ids {
        *uni.entry(t).or_default() += 1;
    }
    let gr = record(&uni);
    cls[CLS_GLOBAL_REC..CLS_GLOBAL_REC + 16].copy_from_slice(&gr);
    anyhow::ensure!(gr[0] != NO_WORD, "global record empty");
    for (g, name) in topic_names.iter().enumerate() {
        cls[CLS_TONE + g] = if name == "GREETING" {
            TONE_GREETING
        } else if topic_tech[g] {
            TONE_TECHNICAL
        } else {
            TONE_NORMAL
        };
    }

    // ---- v5: train the neural next-word model (deterministic) ----
    // Examples: trigram (w1,w2)->w3 plus backoff-style ($FF,w)->next pairs,
    // so seeded contexts with no first word still get real predictions.
    let mut triples: Vec<(u8, u8, u8)> = Vec::new();
    for w in ids.windows(3) {
        triples.push((w[0], w[1], w[2]));
    }
    for w in ids.windows(2) {
        triples.push((NO_WORD, w[0], w[1]));
    }
    let (netf, train_loss) = neural::train(vocab.words.len(), &triples, 1200, 0.08);
    let neural = neural::quantize(&netf, &triples);
    eprintln!(
        "neural: final CE loss {:.3}, sh_h={}, sh_out={}",
        train_loss, neural.sh_h, neural.sh_out
    );

    let model = Model {
        vocab_size,
        k: TOP_K as u8,
        qmark_id,
        emergency,
        tri,
        bi,
        vocab,
        attr,
        routes,
        cls,
        neural,
        topic_names,
        topic_tech,
        qtype_names,
    };

    // --- every route seed (and the emergency seed) must answer well ---
    for row in &model.routes {
        let (words, _) = gen_from(&model, (row.seed1, row.seed2), None, None);
        anyhow::ensure!(
            words.len() >= MIN_ANSWER_WORDS,
            "route seed ({}, {}) generates only {} words",
            model.vocab.words[row.seed1 as usize],
            model.vocab.words[row.seed2 as usize],
            words.len()
        );
    }
    let (ew, _) = gen_from(&model, model.emergency, None, None);
    anyhow::ensure!(ew.len() >= MIN_ANSWER_WORDS, "emergency seed answers too short");
    anyhow::ensure!(
        model.tri_rec(model.emergency.0, model.emergency.1)[0] != NO_WORD,
        "emergency seed must hit a trigram record directly"
    );

    Ok((model, n_tri, n_bi))
}

// ---------------------------------------------------------------------------
// The always-answer pipeline — the exact algorithm the 6502 runs.
// normalize -> parse typed chars -> classify -> route -> generate -> watchdog
// ---------------------------------------------------------------------------

/// Everything the pipeline decided, for receipts and parity checks.
#[derive(Debug)]
pub struct Trace {
    pub typed: Vec<u8>,          // normalized char IDs (the NES TYPED buffer)
    pub normalized_text: String, // human-readable form of `typed`
    pub known_word_ids: Vec<u8>,
    pub qtype: Option<u8>,       // qtype GROUP id (index into qtype pages)
    pub topic: Option<u8>,       // topic GROUP id (index into topic pages)
    pub tone: u8,                // TONE_* (ROM mirrors at $07F6)
    pub seed: (u8, u8),
    pub reason: u8,              // R_EXACT..R_EMERGENCY (ROM mirrors at $07F4)
    pub gen_level_max: u8,       // worst backoff level used (ROM at $07F5)
    pub words: Vec<u8>,
}

/// Normalization = exactly what the on-screen keyboard can hold: uppercase,
/// vocab chars only, capped at MAX_TYPED_CHARS.
pub fn normalize_chars(text: &str) -> Vec<u8> {
    let mut ids = char_ids(text);
    ids.truncate(MAX_TYPED_CHARS);
    ids
}

/// Char classes, mirroring char_class in nanocamelid.s:
/// 0 = separator (space, '-'), 1 = word char, 2 = punctuation word.
fn class_of(id: u8) -> u8 {
    match id {
        0 | 40 => 0,
        39 => 1,                  // apostrophe joins words
        37 | 38 | 41 | 42 => 2,   // . , ! ?
        1..=36 => 1,
        _ => 0,
    }
}

/// v4 stem matcher (tokenizer "v4-stem1"), mirrored on the 6502:
/// 1. exact vocab match
/// 2. strip one trailing S and retry  (HUMS -> HUM if HUMS unknown)
/// 3. longest vocab word (len >= 4) that is a PREFIX of the typed word
///    (HUMMINGBIRD -> HUMMING)
pub fn stem_lookup(model: &Model, chars: &[u8]) -> Option<u8> {
    let exact = |c: &[u8]| -> Option<u8> {
        if c.len() > MAX_WORD_LEN {
            return None;
        }
        (0..model.vocab.words.len())
            .find(|&i| char_ids(&model.vocab.words[i]) == c)
            .map(|i| i as u8)
    };
    if let Some(w) = exact(chars) {
        return Some(w);
    }
    let s_id = VOCAB.find('S').unwrap() as u8;
    if chars.len() > 2 && *chars.last().unwrap() == s_id {
        if let Some(w) = exact(&chars[..chars.len() - 1]) {
            return Some(w);
        }
    }
    // longest prefix >= 4 chars; scan in id order, prefer longer
    let mut best: Option<(usize, u8)> = None;
    for i in 0..model.vocab.words.len() {
        let wc = char_ids(&model.vocab.words[i]);
        if wc.len() >= 4 && wc.len() < chars.len() && chars[..wc.len()] == wc[..] {
            if best.map(|(l, _)| wc.len() > l).unwrap_or(true) {
                best = Some((wc.len(), i as u8));
            }
        }
    }
    best.map(|(_, w)| w)
}

/// Parse the typed buffer: known word IDs + conditioning (qtype group =
/// first hit, topic group = last hit) + tone. Mirrors the 6502 parser.
pub fn parse_typed(model: &Model, typed: &[u8]) -> (Vec<u8>, Option<u8>, Option<u8>, u8) {
    let mut known = Vec::new();
    let mut qtype = None;
    let mut topic = None;
    let mut had_any_chars = false;
    {
        let mut hit = |wid: u8, known: &mut Vec<u8>, qtype: &mut Option<u8>, topic: &mut Option<u8>| {
            known.push(wid);
            if qtype.is_none() {
                if let Some(g) = model.qgroup(wid) {
                    *qtype = Some(g);
                }
            }
            if let Some(g) = model.tgroup(wid) {
                *topic = Some(g);
            }
        };
        let mut i = 0;
        while i < typed.len() {
            match class_of(typed[i]) {
                2 => {
                    had_any_chars = true;
                    if let Some(wid) = stem_lookup(model, &typed[i..i + 1]) {
                        hit(wid, &mut known, &mut qtype, &mut topic);
                    }
                    i += 1;
                }
                1 => {
                    had_any_chars = true;
                    let start = i;
                    while i < typed.len() && class_of(typed[i]) == 1 {
                        i += 1;
                    }
                    if let Some(wid) = stem_lookup(model, &typed[start..i]) {
                        hit(wid, &mut known, &mut qtype, &mut topic);
                    }
                }
                _ => i += 1,
            }
        }
    }
    let tone = if known.is_empty() && !had_any_chars {
        TONE_EMPTY
    } else if known.is_empty() {
        TONE_NONSENSE
    } else if topic
        .map(|t| model.topic_names.get(t as usize).map(|n| n == "GREETING").unwrap_or(false))
        .unwrap_or(false)
    {
        TONE_GREETING
    } else if topic.map(|t| *model.topic_tech.get(t as usize).unwrap_or(&false)).unwrap_or(false) {
        TONE_TECHNICAL
    } else {
        TONE_NORMAL
    };
    (known, qtype, topic, tone)
}

/// Seed resolution ladder A-E (F happens after generation). The route table
/// keys on WORDS (attr bits), unchanged from v3 — biases do the rest.
pub fn resolve_seed(
    model: &Model,
    known: &[u8],
    qtype_word: Option<u8>,
    topic_word: Option<u8>,
) -> ((u8, u8), u8) {
    if let [.., a, b] = known {
        if model.tri_rec(*a, *b)[0] != NO_WORD {
            return ((*a, *b), R_EXACT);
        }
    }
    for row in &model.routes {
        let qt_ok = row.qtype == WILDCARD || Some(row.qtype) == qtype_word;
        let topic_ok = row.topic == WILDCARD || Some(row.topic) == topic_word;
        if qt_ok && topic_ok {
            return ((row.seed1, row.seed2), row.level);
        }
    }
    ((NO_WORD, model.qmark_id), R_FALLBACK)
}

pub fn is_sentence_end(model: &Model, id: u8) -> bool {
    model.is_end_tok(id)
}

fn is_punct_word(model: &Model, id: u8) -> bool {
    matches!(
        model.vocab.words.get(id as usize).map(|s| s.as_str()),
        Some(".") | Some(",") | Some("!") | Some("?")
    )
}

/// Pick the candidate record exactly like v4: tri -> bi -> topic prior ->
/// qtype prior -> global. Returns (record, level).
fn ladder_rec<'a>(
    model: &'a Model,
    w1: u8,
    w2: u8,
    topic: Option<u8>,
    qtype: Option<u8>,
) -> (&'a [u8], u8) {
    if w1 != NO_WORD {
        let r = model.tri_rec(w1, w2);
        if r[0] != NO_WORD {
            return (r, L_TRI);
        }
    }
    let r = model.bi_rec(w2);
    if r[0] != NO_WORD {
        return (r, L_BI);
    }
    if let Some(t) = topic {
        let r = model.topic_rec(t);
        if r[0] != NO_WORD {
            return (r, L_TOPIC);
        }
    }
    if let Some(q) = qtype {
        let r = model.qtype_rec(q);
        if r[0] != NO_WORD {
            return (r, L_QTYPE);
        }
    }
    (model.global_rec(), L_GLOBAL)
}

/// v5 NEURAL-ENSEMBLE decoding — the exact loop the 6502 runs. Per word:
///   1. integer forward pass -> neural base score for EVERY token (0..48)
///   2. the v4 candidate ladder record adds its count-based bonus (1..63)
///      to the (at most 8) tokens it lists
///   3. conditioning: + topic_bias + qtype_bias + len_bonus - rep_penalty
/// all u8 saturating; argmax over ALL tokens, initialized with token 0, so
/// generation can never stall. The network can promote tokens the records
/// never saw; the records keep the grammar sharp.
pub fn gen_from(
    model: &Model,
    seed: (u8, u8),
    topic: Option<u8>,
    qtype: Option<u8>,
) -> (Vec<u8>, u8) {
    let (mut w1, mut w2) = seed;
    let mut out: Vec<u8> = Vec::new();
    let mut level_max = L_TRI;
    let mut ring = [NO_WORD; REP_WINDOW];
    let mut ring_i = 0usize;
    while out.len() < WORD_BUDGET {
        let base = model.neural.forward_int(w1, w2);
        let (rec, level) = ladder_rec(model, w1, w2, topic, qtype);
        level_max = level_max.max(level);
        let mut bonus = [0u8; 256];
        for k in 0..TOP_K {
            let tok = rec[k * 2];
            if tok == NO_WORD {
                break;
            }
            bonus[tok as usize] = rec[k * 2 + 1];
        }
        let mut best_tok = 0u8;
        let mut best_score = 0u8;
        let mut first = true;
        for t in 0..model.vocab.words.len() {
            let tok = t as u8;
            let mut sc = base[t];
            sc = sc.saturating_add(bonus[t]);
            sc = sc.saturating_add(model.topic_bias(topic, tok));
            sc = sc.saturating_add(model.qtype_bias(qtype, tok));
            if out.len() + 1 >= SOFT_LEN && model.is_end_tok(tok) {
                sc = sc.saturating_add(LEN_BONUS);
            }
            if !is_punct_word(model, tok) && ring.contains(&tok) {
                sc = sc.saturating_sub(REP_PEN).max(1);
            }
            if first || sc > best_score {
                best_score = sc;
                best_tok = tok;
                first = false;
            }
        }
        out.push(best_tok);
        ring[ring_i] = best_tok;
        ring_i = (ring_i + 1) % REP_WINDOW;
        w1 = w2;
        w2 = best_tok;
        if out.len() >= MIN_WORDS_BEFORE_STOP && model.is_end_tok(best_tok) {
            break;
        }
    }
    (out, level_max)
}

/// The whole pipeline. The answer is never empty and never under
/// MIN_ANSWER_WORDS: the watchdog reseeds from the emergency context.
pub fn answer_pipeline(model: &Model, text: &str) -> Trace {
    let typed = normalize_chars(text);
    let (known, qtype, topic, tone) = parse_typed(model, &typed);
    // route ladder keys on the WORD that hit (attr bits), so re-derive those
    let qtype_word = known.iter().copied().find(|&w| model.attr[w as usize] & ATTR_QTYPE != 0);
    let topic_word = known.iter().copied().rev().find(|&w| model.attr[w as usize] & ATTR_TOPIC != 0);
    let (seed, mut reason) = resolve_seed(model, &known, qtype_word, topic_word);
    let (mut words, mut level_max) = gen_from(model, seed, topic, qtype);
    if words.len() < MIN_ANSWER_WORDS {
        let r = gen_from(model, model.emergency, topic, qtype);
        words = r.0;
        level_max = level_max.max(r.1);
        reason = R_EMERGENCY;
    }
    Trace {
        normalized_text: chars_from_ids(&typed),
        typed,
        known_word_ids: known,
        qtype,
        topic,
        tone,
        seed,
        reason,
        gen_level_max: level_max,
        words,
    }
}

pub fn tone_name(t: u8) -> &'static str {
    match t {
        TONE_NORMAL => "NORMAL",
        TONE_EMPTY => "EMPTY",
        TONE_NONSENSE => "NONSENSE",
        TONE_GREETING => "GREETING",
        TONE_TECHNICAL => "TECHNICAL",
        _ => "?",
    }
}

pub fn level_name(l: u8) -> &'static str {
    match l {
        L_TRI => "TRIGRAM",
        L_BI => "BIGRAM",
        L_TOPIC => "TOPIC_PRIOR",
        L_QTYPE => "QTYPE_PRIOR",
        L_GLOBAL => "GLOBAL_PRIOR",
        L_NEURAL => "NEURAL",
        _ => "?",
    }
}

pub fn reason_name(r: u8) -> &'static str {
    match r {
        R_EXACT => "EXACT",
        R_QT_TOPIC => "ROUTED_QTYPE_TOPIC",
        R_TOPIC => "ROUTED_TOPIC",
        R_QTYPE => "ROUTED_QTYPE",
        R_FALLBACK => "FALLBACK",
        R_EMERGENCY => "EMERGENCY",
        _ => "?",
    }
}

// ---------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// CHR font: tile index == char ID for the 43 glyphs, plus selector, block
// cursor, and the llama. Emitted as 8 KiB chr.bin AND as the first 1024
// bytes (64 tiles) for the CHR-RAM init copy in the fixed bank.
// ---------------------------------------------------------------------------

#[rustfmt::skip]
const GLYPHS_5X7: &[(char, [u8; 7])] = &[
    (' ', [0b00000,0b00000,0b00000,0b00000,0b00000,0b00000,0b00000]),
    ('A', [0b01110,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001]),
    ('B', [0b11110,0b10001,0b10001,0b11110,0b10001,0b10001,0b11110]),
    ('C', [0b01110,0b10001,0b10000,0b10000,0b10000,0b10001,0b01110]),
    ('D', [0b11100,0b10010,0b10001,0b10001,0b10001,0b10010,0b11100]),
    ('E', [0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b11111]),
    ('F', [0b11111,0b10000,0b10000,0b11110,0b10000,0b10000,0b10000]),
    ('G', [0b01110,0b10001,0b10000,0b10111,0b10001,0b10001,0b01111]),
    ('H', [0b10001,0b10001,0b10001,0b11111,0b10001,0b10001,0b10001]),
    ('I', [0b01110,0b00100,0b00100,0b00100,0b00100,0b00100,0b01110]),
    ('J', [0b00111,0b00010,0b00010,0b00010,0b00010,0b10010,0b01100]),
    ('K', [0b10001,0b10010,0b10100,0b11000,0b10100,0b10010,0b10001]),
    ('L', [0b10000,0b10000,0b10000,0b10000,0b10000,0b10000,0b11111]),
    ('M', [0b10001,0b11011,0b10101,0b10101,0b10001,0b10001,0b10001]),
    ('N', [0b10001,0b11001,0b10101,0b10011,0b10001,0b10001,0b10001]),
    ('O', [0b01110,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110]),
    ('P', [0b11110,0b10001,0b10001,0b11110,0b10000,0b10000,0b10000]),
    ('Q', [0b01110,0b10001,0b10001,0b10001,0b10101,0b10010,0b01101]),
    ('R', [0b11110,0b10001,0b10001,0b11110,0b10100,0b10010,0b10001]),
    ('S', [0b01111,0b10000,0b10000,0b01110,0b00001,0b00001,0b11110]),
    ('T', [0b11111,0b00100,0b00100,0b00100,0b00100,0b00100,0b00100]),
    ('U', [0b10001,0b10001,0b10001,0b10001,0b10001,0b10001,0b01110]),
    ('V', [0b10001,0b10001,0b10001,0b10001,0b10001,0b01010,0b00100]),
    ('W', [0b10001,0b10001,0b10001,0b10101,0b10101,0b11011,0b10001]),
    ('X', [0b10001,0b10001,0b01010,0b00100,0b01010,0b10001,0b10001]),
    ('Y', [0b10001,0b10001,0b01010,0b00100,0b00100,0b00100,0b00100]),
    ('Z', [0b11111,0b00001,0b00010,0b00100,0b01000,0b10000,0b11111]),
    ('0', [0b01110,0b10001,0b10011,0b10101,0b11001,0b10001,0b01110]),
    ('1', [0b00100,0b01100,0b00100,0b00100,0b00100,0b00100,0b01110]),
    ('2', [0b01110,0b10001,0b00001,0b00010,0b00100,0b01000,0b11111]),
    ('3', [0b11111,0b00010,0b00100,0b00010,0b00001,0b10001,0b01110]),
    ('4', [0b00010,0b00110,0b01010,0b10010,0b11111,0b00010,0b00010]),
    ('5', [0b11111,0b10000,0b11110,0b00001,0b00001,0b10001,0b01110]),
    ('6', [0b00110,0b01000,0b10000,0b11110,0b10001,0b10001,0b01110]),
    ('7', [0b11111,0b00001,0b00010,0b00100,0b01000,0b01000,0b01000]),
    ('8', [0b01110,0b10001,0b10001,0b01110,0b10001,0b10001,0b01110]),
    ('9', [0b01110,0b10001,0b10001,0b01111,0b00001,0b00010,0b01100]),
    ('.', [0b00000,0b00000,0b00000,0b00000,0b00000,0b01100,0b01100]),
    (',', [0b00000,0b00000,0b00000,0b00000,0b01100,0b00100,0b01000]),
    ('\'',[0b01100,0b00100,0b01000,0b00000,0b00000,0b00000,0b00000]),
    ('-', [0b00000,0b00000,0b00000,0b11111,0b00000,0b00000,0b00000]),
    ('!', [0b00100,0b00100,0b00100,0b00100,0b00100,0b00000,0b00100]),
    ('?', [0b01110,0b10001,0b00001,0b00010,0b00100,0b00000,0b00100]),
];

pub fn font_chr() -> Vec<u8> {
    let mut chr = vec![0u8; 8192];
    let mut put = |tile: usize, rows: &[u8; 7]| {
        let base = tile * 16;
        for (r, &bits) in rows.iter().enumerate() {
            chr[base + r] = bits << 2;
        }
    };
    let vocab: Vec<char> = VOCAB.chars().collect();
    let glyphs: HashMap<char, &[u8; 7]> = GLYPHS_5X7.iter().map(|(c, g)| (*c, g)).collect();
    for (i, ch) in vocab.iter().enumerate() {
        put(i, glyphs.get(ch).unwrap_or_else(|| panic!("no glyph for {ch:?}")));
    }
    put(
        TILE_SELECTOR as usize,
        &[0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000],
    );
    for r in 0..8 {
        chr[TILE_BLOCK as usize * 16 + r] = 0xFF;
    }
    const LLAMA: [&str; 24] = [
        "..##....................",
        ".###.#..................",
        ".#####..................",
        "..####..................",
        "..###...................",
        "..###...................",
        "..###...................",
        "..###...................",
        "..###..................#",
        "..####################.#",
        "..######################",
        ".#######################",
        ".#######################",
        ".#######################",
        ".#######################",
        ".######################.",
        "..###..###......###..###",
        "..###..###......###..###",
        "..###..###......###..###",
        "..###..###......###..###",
        "..###..###......###..###",
        "..##....##......##...##.",
        "........................",
        "........................",
    ];
    for tr in 0..3 {
        for tc in 0..3 {
            let tile = TILE_LLAMA as usize + tr * 3 + tc;
            for r in 0..8 {
                let mut bits = 0u8;
                for (i, ch) in LLAMA[tr * 8 + r][tc * 8..tc * 8 + 8].chars().enumerate() {
                    if ch == '#' {
                        bits |= 0x80 >> i;
                    }
                }
                chr[tile * 16 + r] = bits;
            }
        }
    }
    chr
}

#[cfg(test)]
mod tests {
    use super::*;

    const CORPUS: &str = "Why do llamas hum? Humming calms the herd. \
        What is a llama? A llama is a camelid. The herd hums at dawn. \
        Why do llamas hum? Humming calms the herd. Wow? Beep beep, a tiny \
        brain missed that, but the model is still alive.";
    const ROUTES: &str = "WHY HUM -> HUM ?\n* LLAMA -> LLAMA ?\nWHY * -> HUM ?\nEMERGENCY -> WOW ?\n";
    const TOPICS: &str = "TOPIC LLAMA: LLAMA LLAMAS HUM\nTOPIC SELF TECH: MODEL\n";

    fn model() -> Model {
        build_model(CORPUS, ROUTES, TOPICS).unwrap().0
    }

    #[test]
    fn roundtrip() {
        let m = model();
        let m2 = Model::from_bytes(&m.to_bytes()).unwrap();
        assert_eq!(m.vocab.words, m2.vocab.words);
        assert_eq!(m.routes.len(), m2.routes.len());
        assert_eq!(m.emergency, m2.emergency);
        let a = answer_pipeline(&m, "why do llamas hum?");
        let b = answer_pipeline(&m2, "why do llamas hum?");
        assert_eq!(a.words, b.words);
        assert_eq!(a.reason, b.reason);
    }

    #[test]
    fn exact_route() {
        let m = model();
        let t = answer_pipeline(&m, "WHY DO LLAMAS HUM?");
        assert_eq!(t.reason, R_EXACT);
        assert!(words_to_text(&m.vocab, &t.words).starts_with("HUMMING CALMS THE HERD"));
    }

    #[test]
    fn topic_route() {
        let m = model();
        // "LLAMA JAZZ" -> (LLAMA, JAZZ-unknown): no exact context, topic LLAMA
        let t = answer_pipeline(&m, "LLAMA JAZZ");
        assert_eq!(t.reason, R_TOPIC);
        assert!(t.words.len() >= MIN_ANSWER_WORDS);
    }

    #[test]
    fn unknown_never_blank() {
        let m = model();
        for q in ["", "zzz qqq", "xkcd!", "....", "a", "lol lol lol"] {
            let t = answer_pipeline(&m, q);
            assert!(
                t.words.len() >= MIN_ANSWER_WORDS,
                "{q:?} gave {} words (reason {})",
                t.words.len(),
                reason_name(t.reason)
            );
            assert!(t.words.iter().all(|&w| (w as usize) < m.vocab.words.len()));
        }
    }
}
