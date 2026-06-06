//! Headless NES system for parity verification: CPU + RAM + MMC5-banked PRG
//! ROM + a PPU stub (vblank flag, NMI, $2006/$2007 VRAM writes so the
//! nametable text can be dumped as evidence) + one controller, plus a Driver
//! that types on the ROM's on-screen keyboard through the real controller
//! protocol.
//!
//! MMC5 subset modeled (all the ROM uses): PRG mode 3 (four 8 KiB windows,
//! $5114-$5117, bit7 = ROM). ExRAM, IRQ, split, multiplier, audio: unused
//! and unmodeled. The mapper is a bank window, not a co-processor — the
//! inference loop itself is pure CPU + ROM + RAM.

use crate::cpu6502::{Bus, Cpu};

pub const CYCLES_PER_FRAME: u64 = 29781; // NTSC CPU cycles per frame
pub const VBLANK_CYCLES: u64 = 2273; // ~20 scanlines of vblank

pub const BTN_A: u8 = 0x01;
pub const BTN_B: u8 = 0x02;
pub const BTN_SELECT: u8 = 0x04;
pub const BTN_START: u8 = 0x08;
pub const BTN_UP: u8 = 0x10;
pub const BTN_DOWN: u8 = 0x20;
pub const BTN_LEFT: u8 = 0x40;
pub const BTN_RIGHT: u8 = 0x80;

pub struct NesBus {
    pub ram: [u8; 0x800],
    pub prg: Vec<u8>, // n x 8 KiB banks (1 MiB for NanoCamelid)
    pub prg_bank: [u8; 4], // $5114-$5117 (bit7 stripped), windows at $8000/$A000/$C000/$E000
    pub vram: [u8; 0x800],
    pub palette: [u8; 32],
    ppu_ctrl: u8,
    ppu_status: u8,
    ppu_addr: u16,
    ppu_latch_hi: bool,
    pub buttons: u8, // bit0=A,1=B,2=Select,3=Start,4=Up,5=Down,6=Left,7=Right
    strobe: bool,
    shift: u8,
    pub exram: [u8; 1024], // MMC5 ExRAM ($5C00-$5FFF), scratch in MMC5_MAX builds
}

impl NesBus {
    pub fn new(prg: Vec<u8>) -> NesBus {
        assert!(prg.len() % 0x2000 == 0 && !prg.is_empty(), "PRG must be 8 KiB banks");
        let last = (prg.len() / 0x2000 - 1) as u8;
        NesBus {
            ram: [0; 0x800],
            prg,
            // MMC5 power-on: all windows map the last bank (so vectors work)
            prg_bank: [last; 4],
            vram: [0; 0x800],
            palette: [0; 32],
            ppu_ctrl: 0,
            ppu_status: 0,
            ppu_addr: 0,
            ppu_latch_hi: true,
            buttons: 0,
            strobe: false,
            shift: 0,
            exram: [0; 1024],
        }
    }

    fn n_banks(&self) -> usize {
        self.prg.len() / 0x2000
    }

    pub fn nmi_enabled(&self) -> bool {
        self.ppu_ctrl & 0x80 != 0
    }

    pub fn set_vblank(&mut self, on: bool) {
        if on {
            self.ppu_status |= 0x80;
        } else {
            self.ppu_status &= !0x80;
        }
    }
}

impl Bus for NesBus {
    fn read(&mut self, addr: u16) -> u8 {
        match addr {
            0x0000..=0x1FFF => self.ram[(addr & 0x07FF) as usize],
            0x2000..=0x3FFF => match addr & 0x2007 {
                0x2002 => {
                    let v = self.ppu_status;
                    self.ppu_status &= !0x80;
                    self.ppu_latch_hi = true;
                    v
                }
                _ => 0,
            },
            0x4016 => {
                let bit = if self.strobe {
                    self.buttons & 1
                } else {
                    let b = self.shift & 1;
                    self.shift = (self.shift >> 1) | 0x80;
                    b
                };
                0x40 | bit
            }
            0x5C00..=0x5FFF => self.exram[(addr - 0x5C00) as usize],
            0x8000..=0xFFFF => {
                let window = ((addr - 0x8000) / 0x2000) as usize;
                let bank = self.prg_bank[window] as usize % self.n_banks();
                self.prg[bank * 0x2000 + (addr as usize & 0x1FFF)]
            }
            _ => 0,
        }
    }

    fn write(&mut self, addr: u16, val: u8) {
        match addr {
            0x0000..=0x1FFF => self.ram[(addr & 0x07FF) as usize] = val,
            0x2000..=0x3FFF => match addr & 0x2007 {
                0x2000 => self.ppu_ctrl = val,
                0x2006 => {
                    if self.ppu_latch_hi {
                        self.ppu_addr = (self.ppu_addr & 0x00FF) | ((val as u16) << 8);
                    } else {
                        self.ppu_addr = (self.ppu_addr & 0xFF00) | val as u16;
                    }
                    self.ppu_latch_hi = !self.ppu_latch_hi;
                }
                0x2007 => {
                    let a = self.ppu_addr & 0x3FFF;
                    match a {
                        0x2000..=0x2FFF => self.vram[(a & 0x07FF) as usize] = val,
                        0x3F00..=0x3FFF => self.palette[(a & 0x1F) as usize] = val,
                        _ => {} // CHR-RAM writes: not modeled (no rendering)
                    }
                    let inc = if self.ppu_ctrl & 0x04 != 0 { 32 } else { 1 };
                    self.ppu_addr = self.ppu_addr.wrapping_add(inc);
                }
                _ => {}
            },
            0x4016 => {
                let new_strobe = val & 1 != 0;
                if self.strobe && !new_strobe {
                    self.shift = self.buttons;
                }
                self.strobe = new_strobe;
            }
            0x5C00..=0x5FFF => self.exram[(addr - 0x5C00) as usize] = val,
            0x5114..=0x5117 => {
                // MMC5 PRG bank registers (bit7 = ROM; we model ROM only)
                self.prg_bank[(addr - 0x5114) as usize] = val & 0x7F;
            }
            // other MMC5 regs ($5100 mode, $5101 CHR, $5105 NT, $5127 CHR
            // bank...) are accepted and ignored: the ROM only uses mode-3
            // semantics, which is what this bus hard-codes.
            _ => {}
        }
    }
}

pub struct Nes {
    pub cpu: Cpu,
    pub bus: NesBus,
    pub frame: u64,
    cycle_in_frame: u64,
}

pub struct Rom {
    pub prg: Vec<u8>,
    pub mapper: u8,
}

/// Parse an iNES file (mapper 5 / MMC5 expected).
pub fn parse_ines(bytes: &[u8]) -> anyhow::Result<Rom> {
    anyhow::ensure!(bytes.len() >= 16 && &bytes[0..4] == b"NES\x1a", "not an iNES file");
    let prg_units = bytes[4] as usize;
    let mapper = (bytes[6] >> 4) | (bytes[7] & 0xF0);
    anyhow::ensure!(mapper == 5, "expected mapper 5 (MMC5), got {mapper}");
    let prg_len = prg_units * 16384;
    anyhow::ensure!(bytes.len() >= 16 + prg_len, "truncated ROM");
    Ok(Rom { prg: bytes[16..16 + prg_len].to_vec(), mapper })
}

impl Nes {
    pub fn new(prg: Vec<u8>) -> Nes {
        let mut bus = NesBus::new(prg);
        let mut cpu = Cpu::new();
        cpu.reset(&mut bus);
        Nes { cpu, bus, frame: 0, cycle_in_frame: 0 }
    }

    pub fn run_frame(&mut self) {
        self.bus.set_vblank(true);
        if self.bus.nmi_enabled() {
            self.cpu.nmi(&mut self.bus);
        }
        let mut vblank_over = false;
        while self.cycle_in_frame < CYCLES_PER_FRAME {
            if !vblank_over && self.cycle_in_frame >= VBLANK_CYCLES {
                self.bus.set_vblank(false);
                vblank_over = true;
            }
            self.cycle_in_frame += self.cpu.step(&mut self.bus) as u64;
        }
        self.cycle_in_frame -= CYCLES_PER_FRAME;
        self.frame += 1;
    }

    /// Decode the nametable to text using tile == char ID (plus UI tiles).
    pub fn screen_text(&self, vocab: &str) -> String {
        let chars: Vec<char> = vocab.chars().collect();
        let mut out = String::new();
        for row in 0..30 {
            let mut line = String::new();
            for col in 0..32 {
                let tile = self.bus.vram[row * 32 + col] as usize;
                line.push(match tile {
                    62 => '>',
                    63 => '#',
                    48..=56 => '@',
                    t => *chars.get(t).unwrap_or(&' '),
                });
            }
            out.push_str(line.trim_end());
            out.push('\n');
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Driver: scripted controller input over the real $4016 protocol, including
// typing on the ROM's 4x11 on-screen keyboard (cell index == char ID).
// ---------------------------------------------------------------------------

pub struct Driver {
    pub nes: Nes,
    kb_row: i32,
    kb_col: i32,
}

impl Driver {
    pub fn new(nes: Nes) -> Driver {
        Driver { nes, kb_row: 0, kb_col: 0 }
    }

    pub fn press(&mut self, buttons: u8) {
        self.nes.bus.buttons = buttons;
        self.nes.run_frame();
        self.nes.run_frame();
        self.nes.bus.buttons = 0;
        self.nes.run_frame();
        self.nes.run_frame();
    }

    pub fn idle(&mut self, frames: u64) {
        self.nes.bus.buttons = 0;
        for _ in 0..frames {
            self.nes.run_frame();
        }
    }

    /// Call after the ROM returns to the chat screen (enter_chat resets the
    /// grid cursor to 0,0) so the tracked position stays in lockstep.
    pub fn reset_kb(&mut self) {
        self.kb_row = 0;
        self.kb_col = 0;
    }

    /// Type text on the on-screen keyboard and press Start.
    pub fn type_and_ask(&mut self, text: &str) {
        for id in crate::char_ids(text) {
            let tr = id as i32 / 11;
            let tc = id as i32 % 11;
            while self.kb_row != tr {
                if self.kb_row < tr {
                    self.press(BTN_DOWN);
                    self.kb_row += 1;
                } else {
                    self.press(BTN_UP);
                    self.kb_row -= 1;
                }
            }
            while self.kb_col != tc {
                if self.kb_col < tc {
                    self.press(BTN_RIGHT);
                    self.kb_col += 1;
                } else {
                    self.press(BTN_LEFT);
                    self.kb_col -= 1;
                }
            }
            self.press(BTN_A);
        }
        self.press(BTN_START);
    }
}
