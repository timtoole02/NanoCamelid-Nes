//! Rust reference generator — the source of truth the NES must match.
//!
//! Usage: refgen <model.bin> <corpus.txt> <question text...>

use nano_nes_model_builder::*;
use std::fs;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 4, "usage: refgen <model.bin> <corpus.txt> <question...>");
    let model_bytes = fs::read(&args[1])?;
    let corpus_bytes = fs::read(&args[2])?;
    let question = args[3..].join(" ");

    let model = Model::from_bytes(&model_bytes)?;
    let t = answer_pipeline(&model, &question);

    let out = serde_json::json!({
        "tool": "nano_nes_model_builder/refgen",
        "question": question,
        "normalized_input": t.normalized_text,
        "known_word_ids": t.known_word_ids,
        "question_type": t.qtype.map(|g| model.qtype_names[g as usize].clone()),
        "topic": t.topic.map(|g| model.topic_names[g as usize].clone()),
        "tone": tone_name(t.tone),
        "seed": [t.seed.0, t.seed.1],
        "route_reason": reason_name(t.reason),
        "route_reason_code": t.reason,
        "gen_level_max": level_name(t.gen_level_max),
        "gen_level_max_code": t.gen_level_max,
        "tokenizer_version": TOKENIZER_VERSION,
        "generated_word_ids": t.words,
        "decoded_text": words_to_text(&model.vocab, &t.words),
        "generated_count": t.words.len(),
        "word_budget": WORD_BUDGET,
        "model_sha256": sha256_hex(&model_bytes),
        "corpus_sha256": sha256_hex(&corpus_bytes),
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
