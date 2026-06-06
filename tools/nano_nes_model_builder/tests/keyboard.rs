//! Chat-flow regression test: boot the MMC5 ROM, type a question on the
//! on-screen keyboard through the real controller protocol, press Start,
//! and the NES must produce exactly the Rust reference answer.

use nano_nes_model_builder::nes::{parse_ines, Driver, Nes};
use nano_nes_model_builder::*;
use std::fs;
use std::path::Path;

#[test]
fn typed_question_matches_reference() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let rom_path = base.join("../nano_nes_rom/out/nanocamelid.nes");
    let model_path = base.join("out/model.bin");
    if !rom_path.exists() || !model_path.exists() {
        eprintln!("skipping: run scripts/nano_nes_verify.sh first to build the ROM");
        return;
    }
    let rom = parse_ines(&fs::read(rom_path).unwrap()).unwrap();
    let model = Model::from_bytes(&fs::read(model_path).unwrap()).unwrap();

    let question = "ARE YOU REAL?";
    let trace = answer_pipeline(&model, question);
    let expected = trace.words.clone();
    assert!(expected.len() >= MIN_ANSWER_WORDS);

    let mut d = Driver::new(Nes::new(rom.prg));
    d.idle(60);
    d.type_and_ask(question);
    while d.nes.frame < 8000 && d.nes.bus.ram[0x07F0] != 2 {
        d.nes.run_frame();
    }
    assert_eq!(d.nes.bus.ram[0x07F0], 2, "generation never finished");
    assert_eq!(d.nes.bus.ram[0x07F2], 0xFF, "typed prompts report id $FF");
    let count = d.nes.bus.ram[0x07F1] as usize;
    assert_eq!(
        &d.nes.bus.ram[0x0300..0x0300 + count],
        expected.as_slice(),
        "typed-question words diverge from the reference"
    );
    let text = words_to_text(&model.vocab, &d.nes.bus.ram[0x0300..0x0300 + count].to_vec());
    assert!(text.contains("REAL MODEL"), "expected the receipts answer, got: {text}");
}

#[test]
fn unknown_question_gets_fallback_answer() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let rom_path = base.join("../nano_nes_rom/out/nanocamelid.nes");
    let model_path = base.join("out/model.bin");
    if !rom_path.exists() || !model_path.exists() {
        return;
    }
    let rom = parse_ines(&fs::read(rom_path).unwrap()).unwrap();
    let model = Model::from_bytes(&fs::read(model_path).unwrap()).unwrap();

    let question = "ZORP GLEEP?"; // only "?" is in vocab -> FALLBACK route
    let trace = answer_pipeline(&model, question);
    let expected = trace.words.clone();
    assert_eq!(trace.reason, R_FALLBACK);

    let mut d = Driver::new(Nes::new(rom.prg));
    d.idle(60);
    d.type_and_ask(question);
    while d.nes.frame < 8000 && d.nes.bus.ram[0x07F0] != 2 {
        d.nes.run_frame();
    }
    let count = d.nes.bus.ram[0x07F1] as usize;
    assert_eq!(&d.nes.bus.ram[0x0300..0x0300 + count], expected.as_slice());
    assert_eq!(d.nes.bus.ram[0x07F4], trace.reason, "route reason parity");
}
