//! v5: a tiny neural next-word model, trained with hand-rolled SGD and
//! shipped as quantized integers. The f32 network exists only at build time;
//! the artifact is the integer model, and `forward_int` here is the
//! byte-exact reference for what the 6502 computes.
//!
//! Architecture: x = [emb(w1); emb(w2)] (2 x E i8) -> H ReLU -> V logits.
//! E = 8, H = 16, V = vocab (<= 248). ~6.7 KiB quantized = MMC5 bank 124.
//!
//! Integer pipeline (mirrored on the 6502, no negative shifts anywhere):
//!   h_acc[i]  = sum_j W1q[i][j] * x[j]  + b1q[i]            (i24)
//!   h[i]      = 0 if h_acc <= 0 else min(h_acc >> SH_H, 255)  (ReLU u8)
//!   l_acc[t]  = sum_i W2q[t][i] * h[i]  + b2q[t]            (i24)
//!   base[t]   = 0 if l_acc <= 0 else min(l_acc >> SH_OUT, BASE_CLAMP)
//! then v4 conditioning: + topic_bias + qtype_bias + len bonus - rep penalty,
//! u8 saturating, argmax over ALL tokens (token 0 initializes the argmax, so
//! generation can never stall).

pub const EMB_DIM: usize = 8;
pub const HID_DIM: usize = 16;
pub const IN_DIM: usize = EMB_DIM * 2;
pub const BASE_CLAMP: u8 = 48; // neural term is comparable to record bases (1..63)
pub const NEURAL_LEN: usize = 8192; // bank image size

/// Deterministic LCG so model.bin is bit-reproducible.
struct Lcg(u64);
impl Lcg {
    fn next_f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0 // -1..1
    }
}

pub struct NeuralF32 {
    pub v: usize,
    pub emb: Vec<f32>, // v * EMB_DIM
    pub w1: Vec<f32>,  // HID_DIM * IN_DIM
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,  // v * HID_DIM (row per token)
    pub b2: Vec<f32>,
}

pub struct NeuralInt {
    pub v: usize,
    pub emb: Vec<i8>,
    pub w1: Vec<i8>,
    pub b1: Vec<i16>, // accumulator units
    pub w2: Vec<i8>,
    pub b2: Vec<i16>,
    pub sh_h: u8,
    pub sh_out: u8,
}

/// Train on (w1, w2) -> w3 triples. `no_word` contexts use a zero first slot.
pub fn train(
    v: usize,
    triples: &[(u8, u8, u8)],
    epochs: usize,
    lr0: f32,
) -> (NeuralF32, f32) {
    let mut rng = Lcg(0xCAFE_F00D_D00D_5EED);
    let r = 0.5;
    let mut net = NeuralF32 {
        v,
        emb: (0..v * EMB_DIM).map(|_| rng.next_f() * r).collect(),
        w1: (0..HID_DIM * IN_DIM).map(|_| rng.next_f() * r).collect(),
        b1: vec![0.0; HID_DIM],
        w2: (0..v * HID_DIM).map(|_| rng.next_f() * r).collect(),
        b2: vec![0.0; v],
    };
    let mut loss_avg = 0.0;
    for ep in 0..epochs {
        let lr = lr0 / (1.0 + ep as f32 / epochs as f32 * 3.0);
        let mut loss_sum = 0.0;
        for &(c1, c2, target) in triples {
            // forward
            let mut x = [0.0f32; IN_DIM];
            if c1 != 0xFF {
                x[..EMB_DIM].copy_from_slice(&net.emb[c1 as usize * EMB_DIM..][..EMB_DIM]);
            }
            x[EMB_DIM..].copy_from_slice(&net.emb[c2 as usize * EMB_DIM..][..EMB_DIM]);
            let mut hpre = [0.0f32; HID_DIM];
            let mut h = [0.0f32; HID_DIM];
            for i in 0..HID_DIM {
                let mut a = net.b1[i];
                for j in 0..IN_DIM {
                    a += net.w1[i * IN_DIM + j] * x[j];
                }
                hpre[i] = a;
                h[i] = a.max(0.0);
            }
            let mut logits = vec![0.0f32; v];
            let mut maxl = f32::NEG_INFINITY;
            for t in 0..v {
                let mut a = net.b2[t];
                for i in 0..HID_DIM {
                    a += net.w2[t * HID_DIM + i] * h[i];
                }
                logits[t] = a;
                maxl = maxl.max(a);
            }
            let mut z = 0.0;
            for t in 0..v {
                logits[t] = (logits[t] - maxl).exp();
                z += logits[t];
            }
            let p_t = logits[target as usize] / z;
            loss_sum += -p_t.max(1e-9).ln();
            // backward (softmax CE)
            let mut dh = [0.0f32; HID_DIM];
            for t in 0..v {
                let dl = logits[t] / z - if t == target as usize { 1.0 } else { 0.0 };
                for i in 0..HID_DIM {
                    dh[i] += dl * net.w2[t * HID_DIM + i];
                    net.w2[t * HID_DIM + i] -= lr * dl * h[i];
                }
                net.b2[t] -= lr * dl;
            }
            let mut dx = [0.0f32; IN_DIM];
            for i in 0..HID_DIM {
                if hpre[i] <= 0.0 {
                    continue;
                }
                for j in 0..IN_DIM {
                    dx[j] += dh[i] * net.w1[i * IN_DIM + j];
                    net.w1[i * IN_DIM + j] -= lr * dh[i] * x[j];
                }
                net.b1[i] -= lr * dh[i];
            }
            if c1 != 0xFF {
                for j in 0..EMB_DIM {
                    net.emb[c1 as usize * EMB_DIM + j] -= lr * dx[j];
                }
            }
            for j in 0..EMB_DIM {
                net.emb[c2 as usize * EMB_DIM + j] -= lr * dx[EMB_DIM + j];
            }
        }
        loss_avg = loss_sum / triples.len() as f32;
    }
    (net, loss_avg)
}

fn quant_i8(v: &[f32]) -> (Vec<i8>, f32) {
    let m = v.iter().fold(0.0f32, |a, &x| a.max(x.abs())).max(1e-6);
    let s = 127.0 / m;
    (v.iter().map(|&x| (x * s).round().clamp(-127.0, 127.0) as i8).collect(), s)
}

/// Quantize + calibrate shifts on the training contexts.
pub fn quantize(net: &NeuralF32, triples: &[(u8, u8, u8)]) -> NeuralInt {
    let (emb_q, se) = quant_i8(&net.emb);
    let (w1_q, s1) = quant_i8(&net.w1);
    let (w2_q, s2) = quant_i8(&net.w2);
    let b1_q: Vec<i16> = net
        .b1
        .iter()
        .map(|&b| (b * se * s1).round().clamp(-32000.0, 32000.0) as i16)
        .collect();

    // calibrate SH_H: largest positive hidden accumulator on training data
    let hidden_acc = |c1: u8, c2: u8, i: usize| -> i32 {
        let mut acc = b1_q[i] as i32;
        for j in 0..IN_DIM {
            let x = if j < EMB_DIM {
                if c1 == 0xFF { 0 } else { emb_q[c1 as usize * EMB_DIM + j] as i32 }
            } else {
                emb_q[c2 as usize * EMB_DIM + (j - EMB_DIM)] as i32
            };
            acc += w1_q[i * IN_DIM + j] as i32 * x;
        }
        acc
    };
    let mut max_h = 1i32;
    for &(c1, c2, _) in triples {
        for i in 0..HID_DIM {
            max_h = max_h.max(hidden_acc(c1, c2, i));
        }
    }
    let mut sh_h = 0u8;
    while (max_h >> sh_h) > 255 {
        sh_h += 1;
    }

    let sh_eff = se * s1 / (1 << sh_h) as f32; // u8 hidden units per f32
    let b2_q: Vec<i16> = net
        .b2
        .iter()
        .map(|&b| (b * sh_eff * s2).round().clamp(-32000.0, 32000.0) as i16)
        .collect();

    // calibrate SH_OUT against BASE_CLAMP
    let mut tmp = NeuralInt {
        v: net.v,
        emb: emb_q,
        w1: w1_q,
        b1: b1_q,
        w2: w2_q,
        b2: b2_q,
        sh_h,
        sh_out: 0,
    };
    let mut max_l = 1i32;
    for &(c1, c2, _) in triples {
        let h = tmp.hidden(c1, c2);
        for t in 0..tmp.v {
            max_l = max_l.max(tmp.logit_acc(t, &h));
        }
    }
    let mut sh_out = 0u8;
    while (max_l >> sh_out) > BASE_CLAMP as i32 {
        sh_out += 1;
    }
    tmp.sh_out = sh_out;
    tmp
}

impl NeuralInt {
    pub fn hidden(&self, c1: u8, c2: u8) -> [u8; HID_DIM] {
        let mut h = [0u8; HID_DIM];
        for i in 0..HID_DIM {
            let mut acc = self.b1[i] as i32;
            for j in 0..IN_DIM {
                let x = if j < EMB_DIM {
                    if c1 == 0xFF { 0 } else { self.emb[c1 as usize * EMB_DIM + j] as i32 }
                } else {
                    self.emb[c2 as usize * EMB_DIM + (j - EMB_DIM)] as i32
                };
                acc += self.w1[i * IN_DIM + j] as i32 * x;
            }
            h[i] = if acc <= 0 { 0 } else { (acc >> self.sh_h).min(255) as u8 };
        }
        h
    }

    pub fn logit_acc(&self, t: usize, h: &[u8; HID_DIM]) -> i32 {
        let mut acc = self.b2[t] as i32;
        for i in 0..HID_DIM {
            acc += self.w2[t * HID_DIM + i] as i32 * h[i] as i32;
        }
        acc
    }

    /// The byte-exact forward pass: u8 base scores for every token.
    pub fn forward_int(&self, c1: u8, c2: u8) -> Vec<u8> {
        let h = self.hidden(c1, c2);
        (0..self.v)
            .map(|t| {
                let acc = self.logit_acc(t, &h);
                if acc <= 0 { 0 } else { (acc >> self.sh_out).min(BASE_CLAMP as i32) as u8 }
            })
            .collect()
    }

    /// Bank image for MMC5 bank 124 ($8000 window during generation).
    /// Layout (page-friendly):
    ///   $0000: emb        (v * 8, i8)
    ///   $0800: w1         (16 * 16, i8)
    ///   $0900: b1         (16 * i16 LE)
    ///   $0A00: b2         (v * i16 LE)
    ///   $0C00: w2         (v * 16, i8, row per token)
    ///   $1FF0: sh_h, sh_out, v
    pub fn bank_image(&self) -> Vec<u8> {
        let mut b = vec![0u8; NEURAL_LEN];
        for (i, &e) in self.emb.iter().enumerate() {
            b[i] = e as u8;
        }
        for (i, &w) in self.w1.iter().enumerate() {
            b[0x0800 + i] = w as u8;
        }
        for (i, &x) in self.b1.iter().enumerate() {
            b[0x0900 + i * 2] = (x as u16 & 0xFF) as u8;
            b[0x0900 + i * 2 + 1] = (x as u16 >> 8) as u8;
        }
        for (i, &x) in self.b2.iter().enumerate() {
            b[0x0A00 + i * 2] = (x as u16 & 0xFF) as u8;
            b[0x0A00 + i * 2 + 1] = (x as u16 >> 8) as u8;
        }
        for (i, &w) in self.w2.iter().enumerate() {
            b[0x0C00 + i] = w as u8;
        }
        b[0x1FF0] = self.sh_h;
        b[0x1FF1] = self.sh_out;
        b[0x1FF2] = self.v as u8;
        b
    }

    pub fn from_bank(b: &[u8], v: usize) -> NeuralInt {
        NeuralInt {
            v,
            emb: b[0..v * EMB_DIM].iter().map(|&x| x as i8).collect(),
            w1: b[0x0800..0x0800 + HID_DIM * IN_DIM].iter().map(|&x| x as i8).collect(),
            b1: (0..HID_DIM)
                .map(|i| i16::from_le_bytes([b[0x0900 + i * 2], b[0x0900 + i * 2 + 1]]))
                .collect(),
            b2: (0..v)
                .map(|i| i16::from_le_bytes([b[0x0A00 + i * 2], b[0x0A00 + i * 2 + 1]]))
                .collect(),
            w2: b[0x0C00..0x0C00 + v * HID_DIM].iter().map(|&x| x as i8).collect(),
            sh_h: b[0x1FF0],
            sh_out: b[0x1FF1],
        }
    }
}
