//! Interactive terminal player — chat with the NES ROM with zero input
//! drama. Runs the real ROM on the built-in (parity-verified) 6502 + MMC5
//! core; your keystrokes are translated into genuine NES controller presses
//! that navigate the ROM's on-screen keyboard, and the NES nametable is
//! rendered live in the terminal.
//!
//! Nothing is bypassed: the ROM still reads $4016, still parses the typed
//! words, still runs every inference step on the 6502. This is just a
//! controller you already know how to use.
//!
//! Usage: play <rom.nes> [--turbo]
//!   type letters/space/'?/!/,/.  -> typed on the on-screen keyboard
//!   Backspace                    -> B (delete)   Enter -> Start (ask)
//!   Ctrl-C or Esc                -> quit

use nano_nes_model_builder::nes::{parse_ines, Nes, BTN_A, BTN_B, BTN_DOWN, BTN_LEFT, BTN_RIGHT, BTN_START, BTN_UP};
use nano_nes_model_builder::VOCAB;
use std::collections::VecDeque;
use std::fs;
use std::io::{Read, Write};
use std::time::{Duration, Instant};

struct RawTerm {
    orig: libc::termios,
    is_tty: bool,
}

impl RawTerm {
    fn enter() -> RawTerm {
        unsafe {
            let is_tty = libc::isatty(0) == 1;
            let mut orig: libc::termios = std::mem::zeroed();
            if is_tty {
                libc::tcgetattr(0, &mut orig);
                let mut raw = orig;
                // ISIG off too: Ctrl-C arrives as byte 0x03 and quits
                // cleanly (restoring the terminal) instead of SIGINT-killing
                // the process mid-raw-mode.
                raw.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG);
                raw.c_cc[libc::VMIN] = 1; // block in the reader thread
                raw.c_cc[libc::VTIME] = 0;
                libc::tcsetattr(0, libc::TCSANOW, &raw);
            }
            // NOTE: no O_NONBLOCK here — stdin and stdout share the tty file
            // description, so non-blocking stdin makes print! panic with
            // EAGAIN. Input is read on a dedicated blocking thread instead.
            RawTerm { orig, is_tty }
        }
    }
}

impl Drop for RawTerm {
    fn drop(&mut self) {
        if self.is_tty {
            unsafe {
                libc::tcsetattr(0, libc::TCSANOW, &self.orig);
            }
        }
        print!("\x1b[?25h"); // cursor back on
        let _ = std::io::stdout().flush();
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(args.len() >= 2, "usage: play <rom.nes> [--turbo]");
    let turbo = args.iter().any(|a| a == "--turbo");
    let rom = parse_ines(&fs::read(&args[1])?)?;
    let mut nes = Nes::new(rom.prg);

    let _raw = RawTerm::enter();
    print!("\x1b[2J\x1b[?25l"); // clear screen, hide cursor

    // queue of per-frame controller states (each logical press = 2 on + 2 off)
    let mut presses: VecDeque<u8> = VecDeque::new();
    // grid cursor tracking mirrors the ROM (enter_chat resets it to 0,0)
    let (mut row, mut col): (i32, i32) = (0, 0);

    let push = |presses: &mut VecDeque<u8>, btn: u8| {
        presses.push_back(btn);
        presses.push_back(btn);
        presses.push_back(0);
        presses.push_back(0);
    };
    let type_char = |presses: &mut VecDeque<u8>, row: &mut i32, col: &mut i32, id: i32| {
        let (tr, tc) = (id / 11, id % 11);
        while *row != tr {
            push(presses, if *row < tr { BTN_DOWN } else { BTN_UP });
            *row += if *row < tr { 1 } else { -1 };
        }
        while *col != tc {
            push(presses, if *col < tc { BTN_RIGHT } else { BTN_LEFT });
            *col += if *col < tc { 1 } else { -1 };
        }
        push(presses, BTN_A);
    };

    let is_tty = _raw.is_tty;
    // blocking reader thread -> channel; empty Vec marks EOF
    let (tx, rx) = std::sync::mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        let mut stdin = std::io::stdin();
        let mut buf = [0u8; 64];
        loop {
            match stdin.read(&mut buf) {
                Ok(0) | Err(_) => {
                    let _ = tx.send(Vec::new());
                    break;
                }
                Ok(n) => {
                    if tx.send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
            }
        }
    });
    let mut quit = false;
    let mut eof = false;
    let frame_time = Duration::from_millis(16);
    let mut frame_n: u64 = 0;
    // boot
    for _ in 0..60 {
        nes.run_frame();
    }

    while !quit {
        let t0 = Instant::now();
        // keyboard -> queued controller presses
        while let Ok(chunk) = rx.try_recv() {
            if chunk.is_empty() {
                eof = true;
                break;
            }
            for &b in &chunk {
                match b {
                    b'\r' | b'\n' => {
                        push(&mut presses, BTN_START);
                    }
                    0x7F | 0x08 => {
                        // backspace -> B. In DONE state B also returns to
                        // the chat screen, which resets the grid cursor.
                        push(&mut presses, BTN_B);
                        if nes.bus.ram[0x07F0] == 2 {
                            row = 0;
                            col = 0;
                        }
                    }
                    0x03 | 0x1B => quit = true, // Ctrl-C / Esc
                    c => {
                        let ch = (c as char).to_ascii_uppercase();
                        if let Some(id) = VOCAB.find(ch) {
                            // only meaningful on the chat screen
                            if nes.bus.ram[0x07F0] == 0 {
                                type_char(&mut presses, &mut row, &mut col, id as i32);
                            }
                        }
                    }
                }
            }
        }
        // piped mode: exit once everything played out and the answer is done
        if !is_tty && eof && presses.is_empty() && (nes.bus.ram[0x07F0] == 2 || nes.frame > 20_000) {
            quit = true;
        }
        nes.bus.buttons = presses.pop_front().unwrap_or(0);
        nes.run_frame();
        frame_n += 1;

        // render every 4th frame
        if frame_n % 4 == 0 {
            let screen = nes.screen_text(VOCAB);
            let status = match nes.bus.ram[0x07F0] {
                0 => "type your question, ENTER asks, BACKSPACE deletes, CTRL-C quits",
                1 => "the 6502 is generating...",
                _ => "done — BACKSPACE for a new question, CTRL-C quits",
            };
            let mut out = String::from("\x1b[H");
            out.push_str("┌────────────────────────────────┐\n");
            for line in screen.lines() {
                out.push_str(&format!("│{:<32}│\n", line));
            }
            out.push_str("└────────────────────────────────┘\n");
            out.push_str(&format!(
                "  NanoCamelid NES on the verified headless core — {status}\x1b[K\n"
            ));
            out.push_str(&format!(
                "  words {:>3}  status {}  frame {}\x1b[K\n",
                nes.bus.ram[0x07F1], nes.bus.ram[0x07F0], nes.frame
            ));
            print!("{out}");
            std::io::stdout().flush()?;
        }
        if !turbo {
            let spent = t0.elapsed();
            if spent < frame_time {
                std::thread::sleep(frame_time - spent);
            }
        }
    }
    Ok(())
}
