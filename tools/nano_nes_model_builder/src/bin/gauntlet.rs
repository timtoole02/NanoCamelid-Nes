//! The release gate: every question in random_questions.txt is TYPED into
//! the ROM through the emulated controller, and must produce a visible,
//! valid, reference-identical answer. 0 blanks, 0 hangs, 0 invalid words,
//! 0 route mismatches — or the build fails.
//!
//! Usage: gauntlet <rom.nes> <model.bin> <corpus.txt> <questions.txt> <receipt_out.json>

use nano_nes_model_builder::nes::{parse_ines, Driver, Nes, BTN_B};
use nano_nes_model_builder::*;
use std::fs;
use std::process::ExitCode;

const STATUS: usize = 0x07F0;
const COUNT: usize = 0x07F1;
const REASON: usize = 0x07F4;
const LEVEL: usize = 0x07F5;
const TONE: usize = 0x07F6;
const WORDS: usize = 0x0300;
const FRAME_LIMIT_PER_Q: u64 = 8000; // hang detector

fn main() -> anyhow::Result<ExitCode> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 6,
        "usage: gauntlet <rom.nes> <model.bin> <corpus.txt> <questions.txt> <receipt_out.json>"
    );
    let rom_bytes = fs::read(&args[1])?;
    let model_bytes = fs::read(&args[2])?;
    let corpus_bytes = fs::read(&args[3])?;
    let questions: Vec<String> = fs::read_to_string(&args[4])?
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    let receipt_path = &args[5];

    let model = Model::from_bytes(&model_bytes)?;
    let rom = parse_ines(&rom_bytes)?;

    // ONE long-running NES session: B returns to the chat screen between
    // questions, exactly like a human at a demo booth.
    let mut d = Driver::new(Nes::new(rom.prg.clone()));
    d.idle(60);

    let mut results = Vec::new();
    let mut fails = 0usize;
    let mut reason_hist = std::collections::BTreeMap::<String, usize>::new();

    for (i, q) in questions.iter().enumerate() {
        let trace = answer_pipeline(&model, q);
        let start_frame = d.nes.frame;
        d.type_and_ask(q);
        while d.nes.frame - start_frame < FRAME_LIMIT_PER_Q && d.nes.bus.ram[STATUS] != 2 {
            d.nes.run_frame();
        }
        let done = d.nes.bus.ram[STATUS] == 2;
        let count = d.nes.bus.ram[COUNT] as usize;
        let nes_words: Vec<u8> = d.nes.bus.ram[WORDS..WORDS + count].to_vec();
        let nes_reason = d.nes.bus.ram[REASON];
        let valid = nes_words.iter().all(|&w| (w as usize) < model.vocab.words.len());

        let ok = done                                  // no hang
            && count >= MIN_ANSWER_WORDS               // no blank / short answer
            && count <= WORD_BUDGET                    // max length respected
            && valid                                   // no invalid word IDs
            && nes_words == trace.words                // reference-identical
            && nes_reason == trace.reason              // same route decision
            && d.nes.bus.ram[LEVEL] == trace.gen_level_max  // same backoff level
            && d.nes.bus.ram[TONE] == trace.tone;      // same tone
        if !ok {
            fails += 1;
            eprintln!(
                "FAIL {:>3} {:?}: done={} count={} valid={} reason nes={} rust={}",
                i + 1, q, done, count, valid, nes_reason, trace.reason
            );
            eprintln!("  nes : {:?}", words_to_text(&model.vocab, &nes_words));
            eprintln!("  rust: {:?}", words_to_text(&model.vocab, &trace.words));
        }
        *reason_hist.entry(reason_name(trace.reason).to_string()).or_default() += 1;
        results.push(serde_json::json!({
            "question": q,
            "normalized_input": trace.normalized_text,
            "question_type": trace.qtype.map(|w| model.vocab.words[w as usize].clone()),
            "topic": trace.topic.map(|w| model.vocab.words[w as usize].clone()),
            "route_reason": reason_name(trace.reason),
            "tone": tone_name(trace.tone),
            "gen_level_max": level_name(trace.gen_level_max),
            "answer": words_to_text(&model.vocab, &nes_words),
            "word_count": count,
            "frames": d.nes.frame - start_frame,
            "pass": ok,
        }));
        // back to the chat screen for the next question (grid cursor resets)
        d.press(BTN_B);
        d.idle(8);
        d.reset_kb();
    }

    let receipt = serde_json::json!({
        "tool": "nano_nes_model_builder/gauntlet",
        "claim": "every typed question produces a visible answer of >= 6 words, on one continuous NES session, byte-identical to the Rust reference including the route decision",
        "rom_sha256": sha256_hex(&rom_bytes),
        "model_sha256": sha256_hex(&model_bytes),
        "corpus_sha256": sha256_hex(&corpus_bytes),
        "questions": questions.len(),
        "passed": questions.len() - fails,
        "failed": fails,
        "route_reason_histogram": reason_hist,
        "min_answer_words": MIN_ANSWER_WORDS,
        "word_budget": WORD_BUDGET,
        "results": results,
    });
    if let Some(dir) = std::path::Path::new(receipt_path).parent() {
        fs::create_dir_all(dir)?;
    }
    fs::write(receipt_path, serde_json::to_string_pretty(&receipt)?)?;
    println!(
        "gauntlet: {}/{} PASS ({} fails) — receipt: {}",
        questions.len() - fails,
        questions.len(),
        fails,
        receipt_path
    );
    Ok(if fails == 0 { ExitCode::SUCCESS } else { ExitCode::from(1) })
}
