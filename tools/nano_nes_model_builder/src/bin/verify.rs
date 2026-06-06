//! Emulator parity verifier for the chat ROM.
//!
//! Usage: verify <rom.nes> <model.bin> <corpus.txt> <ref.json> <receipt_out.json> <question...>
//!
//! Boots the ROM in the built-in headless NES core, TYPES the question on the
//! on-screen keyboard through the real controller protocol, presses Start,
//! waits for the done flag, reads the generated word IDs out of NES RAM, and
//! compares them byte-for-byte against the Rust reference generator.
//!
//! Parity hook contract (see tools/nano_nes_rom/src/nanocamelid.s):
//!   $0300..  generated word IDs (answer only)
//!   $07F0    status: 0=chat input, 1=generating, 2=done
//!   $07F1    generated word count
//!   $07F2    $FF (all prompts are typed)
//!   $07F3    word budget

use nano_nes_model_builder::nes::{parse_ines, Driver, Nes};
use nano_nes_model_builder::*;
use std::fs;
use std::process::ExitCode;

const STATUS_ADDR: usize = 0x07F0;
const COUNT_ADDR: usize = 0x07F1;
const BUDGET_ADDR: usize = 0x07F3;
const WORDS_ADDR: usize = 0x0300;
const REASON_ADDR: usize = 0x07F4;
const LEVEL_ADDR: usize = 0x07F5;
const TONE_ADDR: usize = 0x07F6;
const BANKSW_ADDR: usize = 0x07F7;
const MODE_ADDR: usize = 0x07F8;
const STATUS_DONE: u8 = 2;
const MAX_FRAMES: u64 = 8000;

fn main() -> anyhow::Result<ExitCode> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() >= 7,
        "usage: verify <rom.nes> <model.bin> <corpus.txt> <ref.json> <receipt_out.json> <question...>"
    );
    let rom_bytes = fs::read(&args[1])?;
    let model_bytes = fs::read(&args[2])?;
    let corpus_bytes = fs::read(&args[3])?;
    let ref_json: serde_json::Value = serde_json::from_str(&fs::read_to_string(&args[4])?)?;
    let receipt_path = &args[5];
    let question = args[6..].join(" ");

    let model_sha = sha256_hex(&model_bytes);
    let corpus_sha = sha256_hex(&corpus_bytes);
    anyhow::ensure!(ref_json["model_sha256"] == model_sha.as_str(), "ref.json model hash mismatch");
    anyhow::ensure!(ref_json["corpus_sha256"] == corpus_sha.as_str(), "ref.json corpus hash mismatch");
    anyhow::ensure!(ref_json["question"] == question.as_str(), "ref.json question mismatch");
    let ref_words: Vec<u8> = ref_json["generated_word_ids"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("ref.json missing generated_word_ids"))?
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect();

    let model = Model::from_bytes(&model_bytes)?;

    // The trigram table must be byte-identical in the ROM (banks 0..13).
    let rom = parse_ines(&rom_bytes)?;
    anyhow::ensure!(rom.prg.len() == 1_048_576, "expected 1 MiB PRG (MMC5)");
    anyhow::ensure!(&rom.prg[0..TRI_LEN] == model.tri.as_slice(), "trigram table in ROM differs");
    // bigram table sits at the start of the decode bank (#126)
    let bi_off = 126 * 8192;
    anyhow::ensure!(
        &rom.prg[bi_off..bi_off + BI_LEN] == model.bi.as_slice(),
        "bigram table in ROM differs"
    );

    // Boot and type the question through the real input path.
    let mut d = Driver::new(Nes::new(rom.prg.clone()));
    d.idle(60);
    d.type_and_ask(&question);
    while d.nes.frame < MAX_FRAMES && d.nes.bus.ram[STATUS_ADDR] != STATUS_DONE {
        d.nes.run_frame();
    }
    let done = d.nes.bus.ram[STATUS_ADDR] == STATUS_DONE;
    if done {
        d.idle(2); // drain the final tile queue for the screen dump
    }
    let count = d.nes.bus.ram[COUNT_ADDR] as usize;
    let nes_words: Vec<u8> = d.nes.bus.ram[WORDS_ADDR..WORDS_ADDR + count].to_vec();
    let nes_reason = d.nes.bus.ram[REASON_ADDR];
    let nes_level = d.nes.bus.ram[LEVEL_ADDR];
    let nes_tone = d.nes.bus.ram[TONE_ADDR];
    let ref_reason = ref_json["route_reason_code"].as_u64().unwrap_or(99) as u8;
    let ref_level = ref_json["gen_level_max_code"].as_u64().unwrap_or(99) as u8;
    let valid_ids = nes_words.iter().all(|&w| (w as usize) < model.vocab.words.len());
    let parity = done
        && nes_words == ref_words
        && nes_reason == ref_reason
        && nes_level == ref_level
        && valid_ids
        && count >= MIN_ANSWER_WORDS
        && count <= WORD_BUDGET;

    let receipt = serde_json::json!({
        "tool": "nano_nes_model_builder/verify",
        "claim": "the NES ROM ran the greedy next-word loop locally for a typed question and matches the Rust reference exactly",
        "rom_sha256": sha256_hex(&rom_bytes),
        "model_sha256": model_sha,
        "corpus_sha256": corpus_sha,
        "question": question,
        "normalized_input": ref_json["normalized_input"],
        "question_type": ref_json["question_type"],
        "topic": ref_json["topic"],
        "route_reason_rust": ref_json["route_reason"],
        "route_reason_nes": nes_reason,
        "gen_level_max_rust": ref_json["gen_level_max"],
        "gen_level_max_nes": level_name(nes_level),
        "tone_nes": tone_name(nes_tone),
        "bank_switches_mod256": d.nes.bus.ram[BANKSW_ADDR],
        "build_mode": if d.nes.bus.ram[MODE_ADDR] == 1 { "MMC5_MAX" } else { "PURE_6502" },
        "tokenizer_version": TOKENIZER_VERSION,
        "min_answer_words": MIN_ANSWER_WORDS,
        "input_path": "question typed on the on-screen keyboard via emulated controller",
        "rust_generated_word_ids": ref_words,
        "nes_generated_word_ids": nes_words,
        "nes_decoded_text": words_to_text(&model.vocab, &nes_words),
        "nes_reported_budget": d.nes.bus.ram[BUDGET_ADDR],
        "nes_done_flag_seen": done,
        "frames_emulated": d.nes.frame,
        "emulator": "built-in headless 6502/UxROM core (tools/nano_nes_model_builder/src/{cpu6502,nes}.rs)",
        "parity": if parity { "PASS" } else { "FAIL" },
        "nes_screen_text": d.nes.screen_text(VOCAB),
    });
    let receipt_str = serde_json::to_string_pretty(&receipt)?;
    if let Some(dir) = std::path::Path::new(receipt_path).parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(receipt_path, &receipt_str)?;
    println!("{receipt_str}");
    eprintln!("parity: {}", if parity { "PASS" } else { "FAIL" });
    Ok(if parity { ExitCode::SUCCESS } else { ExitCode::from(1) })
}
