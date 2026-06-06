//! Minimal 6502 core (official opcodes only) for the headless parity
//! verifier. Cycle counts are per-opcode base values (page-cross penalties
//! ignored) — accurate enough to pace vblank/NMI for this ROM, which has no
//! cycle-timed raster effects.

pub trait Bus {
    fn read(&mut self, addr: u16) -> u8;
    fn write(&mut self, addr: u16, val: u8);
}

const C: u8 = 0x01;
const Z: u8 = 0x02;
const I: u8 = 0x04;
const D: u8 = 0x08;
const B: u8 = 0x10;
const U: u8 = 0x20;
const V: u8 = 0x40;
const N: u8 = 0x80;

pub struct Cpu {
    pub a: u8,
    pub x: u8,
    pub y: u8,
    pub sp: u8,
    pub pc: u16,
    pub p: u8,
}

impl Cpu {
    pub fn new() -> Cpu {
        Cpu { a: 0, x: 0, y: 0, sp: 0xFD, pc: 0, p: I | U }
    }

    pub fn reset(&mut self, bus: &mut impl Bus) {
        let lo = bus.read(0xFFFC) as u16;
        let hi = bus.read(0xFFFD) as u16;
        self.pc = (hi << 8) | lo;
        self.sp = 0xFD;
        self.p = I | U;
    }

    pub fn nmi(&mut self, bus: &mut impl Bus) {
        let pc = self.pc;
        self.push(bus, (pc >> 8) as u8);
        self.push(bus, pc as u8);
        self.push(bus, (self.p & !B) | U);
        self.p |= I;
        let lo = bus.read(0xFFFA) as u16;
        let hi = bus.read(0xFFFB) as u16;
        self.pc = (hi << 8) | lo;
    }

    fn push(&mut self, bus: &mut impl Bus, v: u8) {
        bus.write(0x0100 + self.sp as u16, v);
        self.sp = self.sp.wrapping_sub(1);
    }

    fn pop(&mut self, bus: &mut impl Bus) -> u8 {
        self.sp = self.sp.wrapping_add(1);
        bus.read(0x0100 + self.sp as u16)
    }

    fn set_zn(&mut self, v: u8) {
        self.p = (self.p & !(Z | N)) | if v == 0 { Z } else { 0 } | (v & N);
    }

    fn fetch(&mut self, bus: &mut impl Bus) -> u8 {
        let v = bus.read(self.pc);
        self.pc = self.pc.wrapping_add(1);
        v
    }

    fn fetch16(&mut self, bus: &mut impl Bus) -> u16 {
        let lo = self.fetch(bus) as u16;
        let hi = self.fetch(bus) as u16;
        (hi << 8) | lo
    }

    // Addressing modes -> effective address
    fn zp(&mut self, bus: &mut impl Bus) -> u16 {
        self.fetch(bus) as u16
    }
    fn zpx(&mut self, bus: &mut impl Bus) -> u16 {
        self.fetch(bus).wrapping_add(self.x) as u16
    }
    fn zpy(&mut self, bus: &mut impl Bus) -> u16 {
        self.fetch(bus).wrapping_add(self.y) as u16
    }
    fn abs(&mut self, bus: &mut impl Bus) -> u16 {
        self.fetch16(bus)
    }
    fn absx(&mut self, bus: &mut impl Bus) -> u16 {
        self.fetch16(bus).wrapping_add(self.x as u16)
    }
    fn absy(&mut self, bus: &mut impl Bus) -> u16 {
        self.fetch16(bus).wrapping_add(self.y as u16)
    }
    fn indx(&mut self, bus: &mut impl Bus) -> u16 {
        let zp = self.fetch(bus).wrapping_add(self.x);
        let lo = bus.read(zp as u16) as u16;
        let hi = bus.read(zp.wrapping_add(1) as u16) as u16;
        (hi << 8) | lo
    }
    fn indy(&mut self, bus: &mut impl Bus) -> u16 {
        let zp = self.fetch(bus);
        let lo = bus.read(zp as u16) as u16;
        let hi = bus.read(zp.wrapping_add(1) as u16) as u16;
        ((hi << 8) | lo).wrapping_add(self.y as u16)
    }

    fn adc(&mut self, v: u8) {
        let sum = self.a as u16 + v as u16 + (self.p & C) as u16;
        let r = sum as u8;
        self.p &= !(C | V);
        if sum > 0xFF {
            self.p |= C;
        }
        if (self.a ^ r) & (v ^ r) & 0x80 != 0 {
            self.p |= V;
        }
        self.a = r;
        self.set_zn(r);
    }

    fn sbc(&mut self, v: u8) {
        self.adc(v ^ 0xFF);
    }

    fn cmp_op(&mut self, reg: u8, v: u8) {
        let r = reg.wrapping_sub(v);
        self.p = (self.p & !C) | if reg >= v { C } else { 0 };
        self.set_zn(r);
    }

    fn branch(&mut self, bus: &mut impl Bus, cond: bool) -> u32 {
        let off = self.fetch(bus) as i8;
        if cond {
            self.pc = self.pc.wrapping_add(off as u16);
            3
        } else {
            2
        }
    }

    fn asl_v(&mut self, v: u8) -> u8 {
        self.p = (self.p & !C) | (v >> 7);
        let r = v << 1;
        self.set_zn(r);
        r
    }
    fn lsr_v(&mut self, v: u8) -> u8 {
        self.p = (self.p & !C) | (v & 1);
        let r = v >> 1;
        self.set_zn(r);
        r
    }
    fn rol_v(&mut self, v: u8) -> u8 {
        let c = self.p & C;
        self.p = (self.p & !C) | (v >> 7);
        let r = (v << 1) | c;
        self.set_zn(r);
        r
    }
    fn ror_v(&mut self, v: u8) -> u8 {
        let c = self.p & C;
        self.p = (self.p & !C) | (v & 1);
        let r = (v >> 1) | (c << 7);
        self.set_zn(r);
        r
    }

    /// Execute one instruction; returns cycles consumed.
    pub fn step(&mut self, bus: &mut impl Bus) -> u32 {
        let op = self.fetch(bus);
        macro_rules! rmw {
            ($addr:expr, $f:ident) => {{
                let a = $addr;
                let v = bus.read(a);
                let r = self.$f(v);
                bus.write(a, r);
            }};
        }
        match op {
            // LDA
            0xA9 => { let v = self.fetch(bus); self.a = v; self.set_zn(v); 2 }
            0xA5 => { let a = self.zp(bus); let v = bus.read(a); self.a = v; self.set_zn(v); 3 }
            0xB5 => { let a = self.zpx(bus); let v = bus.read(a); self.a = v; self.set_zn(v); 4 }
            0xAD => { let a = self.abs(bus); let v = bus.read(a); self.a = v; self.set_zn(v); 4 }
            0xBD => { let a = self.absx(bus); let v = bus.read(a); self.a = v; self.set_zn(v); 4 }
            0xB9 => { let a = self.absy(bus); let v = bus.read(a); self.a = v; self.set_zn(v); 4 }
            0xA1 => { let a = self.indx(bus); let v = bus.read(a); self.a = v; self.set_zn(v); 6 }
            0xB1 => { let a = self.indy(bus); let v = bus.read(a); self.a = v; self.set_zn(v); 5 }
            // LDX
            0xA2 => { let v = self.fetch(bus); self.x = v; self.set_zn(v); 2 }
            0xA6 => { let a = self.zp(bus); let v = bus.read(a); self.x = v; self.set_zn(v); 3 }
            0xB6 => { let a = self.zpy(bus); let v = bus.read(a); self.x = v; self.set_zn(v); 4 }
            0xAE => { let a = self.abs(bus); let v = bus.read(a); self.x = v; self.set_zn(v); 4 }
            0xBE => { let a = self.absy(bus); let v = bus.read(a); self.x = v; self.set_zn(v); 4 }
            // LDY
            0xA0 => { let v = self.fetch(bus); self.y = v; self.set_zn(v); 2 }
            0xA4 => { let a = self.zp(bus); let v = bus.read(a); self.y = v; self.set_zn(v); 3 }
            0xB4 => { let a = self.zpx(bus); let v = bus.read(a); self.y = v; self.set_zn(v); 4 }
            0xAC => { let a = self.abs(bus); let v = bus.read(a); self.y = v; self.set_zn(v); 4 }
            0xBC => { let a = self.absx(bus); let v = bus.read(a); self.y = v; self.set_zn(v); 4 }
            // STA
            0x85 => { let a = self.zp(bus); bus.write(a, self.a); 3 }
            0x95 => { let a = self.zpx(bus); bus.write(a, self.a); 4 }
            0x8D => { let a = self.abs(bus); bus.write(a, self.a); 4 }
            0x9D => { let a = self.absx(bus); bus.write(a, self.a); 5 }
            0x99 => { let a = self.absy(bus); bus.write(a, self.a); 5 }
            0x81 => { let a = self.indx(bus); bus.write(a, self.a); 6 }
            0x91 => { let a = self.indy(bus); bus.write(a, self.a); 6 }
            // STX / STY
            0x86 => { let a = self.zp(bus); bus.write(a, self.x); 3 }
            0x96 => { let a = self.zpy(bus); bus.write(a, self.x); 4 }
            0x8E => { let a = self.abs(bus); bus.write(a, self.x); 4 }
            0x84 => { let a = self.zp(bus); bus.write(a, self.y); 3 }
            0x94 => { let a = self.zpx(bus); bus.write(a, self.y); 4 }
            0x8C => { let a = self.abs(bus); bus.write(a, self.y); 4 }
            // Transfers
            0xAA => { self.x = self.a; self.set_zn(self.x); 2 }
            0xA8 => { self.y = self.a; self.set_zn(self.y); 2 }
            0x8A => { self.a = self.x; self.set_zn(self.a); 2 }
            0x98 => { self.a = self.y; self.set_zn(self.a); 2 }
            0xBA => { self.x = self.sp; self.set_zn(self.x); 2 }
            0x9A => { self.sp = self.x; 2 }
            // Stack
            0x48 => { let v = self.a; self.push(bus, v); 3 }
            0x08 => { let v = self.p | B | U; self.push(bus, v); 3 }
            0x68 => { let v = self.pop(bus); self.a = v; self.set_zn(v); 4 }
            0x28 => { let v = self.pop(bus); self.p = (v | U) & !B; 4 }
            // AND
            0x29 => { let v = self.fetch(bus); self.a &= v; self.set_zn(self.a); 2 }
            0x25 => { let a = self.zp(bus); self.a &= bus.read(a); self.set_zn(self.a); 3 }
            0x35 => { let a = self.zpx(bus); self.a &= bus.read(a); self.set_zn(self.a); 4 }
            0x2D => { let a = self.abs(bus); self.a &= bus.read(a); self.set_zn(self.a); 4 }
            0x3D => { let a = self.absx(bus); self.a &= bus.read(a); self.set_zn(self.a); 4 }
            0x39 => { let a = self.absy(bus); self.a &= bus.read(a); self.set_zn(self.a); 4 }
            0x21 => { let a = self.indx(bus); self.a &= bus.read(a); self.set_zn(self.a); 6 }
            0x31 => { let a = self.indy(bus); self.a &= bus.read(a); self.set_zn(self.a); 5 }
            // ORA
            0x09 => { let v = self.fetch(bus); self.a |= v; self.set_zn(self.a); 2 }
            0x05 => { let a = self.zp(bus); self.a |= bus.read(a); self.set_zn(self.a); 3 }
            0x15 => { let a = self.zpx(bus); self.a |= bus.read(a); self.set_zn(self.a); 4 }
            0x0D => { let a = self.abs(bus); self.a |= bus.read(a); self.set_zn(self.a); 4 }
            0x1D => { let a = self.absx(bus); self.a |= bus.read(a); self.set_zn(self.a); 4 }
            0x19 => { let a = self.absy(bus); self.a |= bus.read(a); self.set_zn(self.a); 4 }
            0x01 => { let a = self.indx(bus); self.a |= bus.read(a); self.set_zn(self.a); 6 }
            0x11 => { let a = self.indy(bus); self.a |= bus.read(a); self.set_zn(self.a); 5 }
            // EOR
            0x49 => { let v = self.fetch(bus); self.a ^= v; self.set_zn(self.a); 2 }
            0x45 => { let a = self.zp(bus); self.a ^= bus.read(a); self.set_zn(self.a); 3 }
            0x55 => { let a = self.zpx(bus); self.a ^= bus.read(a); self.set_zn(self.a); 4 }
            0x4D => { let a = self.abs(bus); self.a ^= bus.read(a); self.set_zn(self.a); 4 }
            0x5D => { let a = self.absx(bus); self.a ^= bus.read(a); self.set_zn(self.a); 4 }
            0x59 => { let a = self.absy(bus); self.a ^= bus.read(a); self.set_zn(self.a); 4 }
            0x41 => { let a = self.indx(bus); self.a ^= bus.read(a); self.set_zn(self.a); 6 }
            0x51 => { let a = self.indy(bus); self.a ^= bus.read(a); self.set_zn(self.a); 5 }
            // BIT
            0x24 => { let a = self.zp(bus); let v = bus.read(a); self.p = (self.p & !(Z | V | N)) | (v & (V | N)) | if self.a & v == 0 { Z } else { 0 }; 3 }
            0x2C => { let a = self.abs(bus); let v = bus.read(a); self.p = (self.p & !(Z | V | N)) | (v & (V | N)) | if self.a & v == 0 { Z } else { 0 }; 4 }
            // ADC
            0x69 => { let v = self.fetch(bus); self.adc(v); 2 }
            0x65 => { let a = self.zp(bus); let v = bus.read(a); self.adc(v); 3 }
            0x75 => { let a = self.zpx(bus); let v = bus.read(a); self.adc(v); 4 }
            0x6D => { let a = self.abs(bus); let v = bus.read(a); self.adc(v); 4 }
            0x7D => { let a = self.absx(bus); let v = bus.read(a); self.adc(v); 4 }
            0x79 => { let a = self.absy(bus); let v = bus.read(a); self.adc(v); 4 }
            0x61 => { let a = self.indx(bus); let v = bus.read(a); self.adc(v); 6 }
            0x71 => { let a = self.indy(bus); let v = bus.read(a); self.adc(v); 5 }
            // SBC
            0xE9 => { let v = self.fetch(bus); self.sbc(v); 2 }
            0xE5 => { let a = self.zp(bus); let v = bus.read(a); self.sbc(v); 3 }
            0xF5 => { let a = self.zpx(bus); let v = bus.read(a); self.sbc(v); 4 }
            0xED => { let a = self.abs(bus); let v = bus.read(a); self.sbc(v); 4 }
            0xFD => { let a = self.absx(bus); let v = bus.read(a); self.sbc(v); 4 }
            0xF9 => { let a = self.absy(bus); let v = bus.read(a); self.sbc(v); 4 }
            0xE1 => { let a = self.indx(bus); let v = bus.read(a); self.sbc(v); 6 }
            0xF1 => { let a = self.indy(bus); let v = bus.read(a); self.sbc(v); 5 }
            // CMP / CPX / CPY
            0xC9 => { let v = self.fetch(bus); self.cmp_op(self.a, v); 2 }
            0xC5 => { let a = self.zp(bus); let v = bus.read(a); self.cmp_op(self.a, v); 3 }
            0xD5 => { let a = self.zpx(bus); let v = bus.read(a); self.cmp_op(self.a, v); 4 }
            0xCD => { let a = self.abs(bus); let v = bus.read(a); self.cmp_op(self.a, v); 4 }
            0xDD => { let a = self.absx(bus); let v = bus.read(a); self.cmp_op(self.a, v); 4 }
            0xD9 => { let a = self.absy(bus); let v = bus.read(a); self.cmp_op(self.a, v); 4 }
            0xC1 => { let a = self.indx(bus); let v = bus.read(a); self.cmp_op(self.a, v); 6 }
            0xD1 => { let a = self.indy(bus); let v = bus.read(a); self.cmp_op(self.a, v); 5 }
            0xE0 => { let v = self.fetch(bus); self.cmp_op(self.x, v); 2 }
            0xE4 => { let a = self.zp(bus); let v = bus.read(a); self.cmp_op(self.x, v); 3 }
            0xEC => { let a = self.abs(bus); let v = bus.read(a); self.cmp_op(self.x, v); 4 }
            0xC0 => { let v = self.fetch(bus); self.cmp_op(self.y, v); 2 }
            0xC4 => { let a = self.zp(bus); let v = bus.read(a); self.cmp_op(self.y, v); 3 }
            0xCC => { let a = self.abs(bus); let v = bus.read(a); self.cmp_op(self.y, v); 4 }
            // INC / DEC
            0xE6 => { let a = self.zp(bus); let v = bus.read(a).wrapping_add(1); bus.write(a, v); self.set_zn(v); 5 }
            0xF6 => { let a = self.zpx(bus); let v = bus.read(a).wrapping_add(1); bus.write(a, v); self.set_zn(v); 6 }
            0xEE => { let a = self.abs(bus); let v = bus.read(a).wrapping_add(1); bus.write(a, v); self.set_zn(v); 6 }
            0xFE => { let a = self.absx(bus); let v = bus.read(a).wrapping_add(1); bus.write(a, v); self.set_zn(v); 7 }
            0xC6 => { let a = self.zp(bus); let v = bus.read(a).wrapping_sub(1); bus.write(a, v); self.set_zn(v); 5 }
            0xD6 => { let a = self.zpx(bus); let v = bus.read(a).wrapping_sub(1); bus.write(a, v); self.set_zn(v); 6 }
            0xCE => { let a = self.abs(bus); let v = bus.read(a).wrapping_sub(1); bus.write(a, v); self.set_zn(v); 6 }
            0xDE => { let a = self.absx(bus); let v = bus.read(a).wrapping_sub(1); bus.write(a, v); self.set_zn(v); 7 }
            0xE8 => { self.x = self.x.wrapping_add(1); self.set_zn(self.x); 2 }
            0xC8 => { self.y = self.y.wrapping_add(1); self.set_zn(self.y); 2 }
            0xCA => { self.x = self.x.wrapping_sub(1); self.set_zn(self.x); 2 }
            0x88 => { self.y = self.y.wrapping_sub(1); self.set_zn(self.y); 2 }
            // Shifts / rotates
            0x0A => { self.a = self.asl_v(self.a); 2 }
            0x06 => { rmw!(self.zp(bus), asl_v); 5 }
            0x16 => { rmw!(self.zpx(bus), asl_v); 6 }
            0x0E => { rmw!(self.abs(bus), asl_v); 6 }
            0x1E => { rmw!(self.absx(bus), asl_v); 7 }
            0x4A => { self.a = self.lsr_v(self.a); 2 }
            0x46 => { rmw!(self.zp(bus), lsr_v); 5 }
            0x56 => { rmw!(self.zpx(bus), lsr_v); 6 }
            0x4E => { rmw!(self.abs(bus), lsr_v); 6 }
            0x5E => { rmw!(self.absx(bus), lsr_v); 7 }
            0x2A => { self.a = self.rol_v(self.a); 2 }
            0x26 => { rmw!(self.zp(bus), rol_v); 5 }
            0x36 => { rmw!(self.zpx(bus), rol_v); 6 }
            0x2E => { rmw!(self.abs(bus), rol_v); 6 }
            0x3E => { rmw!(self.absx(bus), rol_v); 7 }
            0x6A => { self.a = self.ror_v(self.a); 2 }
            0x66 => { rmw!(self.zp(bus), ror_v); 5 }
            0x76 => { rmw!(self.zpx(bus), ror_v); 6 }
            0x6E => { rmw!(self.abs(bus), ror_v); 6 }
            0x7E => { rmw!(self.absx(bus), ror_v); 7 }
            // Jumps
            0x4C => { self.pc = self.fetch16(bus); 3 }
            0x6C => {
                // JMP (ind) with the 6502 page-wrap bug
                let ptr = self.fetch16(bus);
                let lo = bus.read(ptr) as u16;
                let hi_addr = (ptr & 0xFF00) | ((ptr.wrapping_add(1)) & 0x00FF);
                let hi = bus.read(hi_addr) as u16;
                self.pc = (hi << 8) | lo;
                5
            }
            0x20 => {
                let target = self.fetch16(bus);
                let ret = self.pc.wrapping_sub(1);
                self.push(bus, (ret >> 8) as u8);
                self.push(bus, ret as u8);
                self.pc = target;
                6
            }
            0x60 => {
                let lo = self.pop(bus) as u16;
                let hi = self.pop(bus) as u16;
                self.pc = ((hi << 8) | lo).wrapping_add(1);
                6
            }
            0x40 => {
                let p = self.pop(bus);
                self.p = (p | U) & !B;
                let lo = self.pop(bus) as u16;
                let hi = self.pop(bus) as u16;
                self.pc = (hi << 8) | lo;
                6
            }
            0x00 => {
                // BRK
                let ret = self.pc.wrapping_add(1);
                self.push(bus, (ret >> 8) as u8);
                self.push(bus, ret as u8);
                let p = self.p | B | U;
                self.push(bus, p);
                self.p |= I;
                let lo = bus.read(0xFFFE) as u16;
                let hi = bus.read(0xFFFF) as u16;
                self.pc = (hi << 8) | lo;
                7
            }
            // Branches
            0x90 => { let c = self.p & C == 0; self.branch(bus, c) }
            0xB0 => { let c = self.p & C != 0; self.branch(bus, c) }
            0xF0 => { let c = self.p & Z != 0; self.branch(bus, c) }
            0xD0 => { let c = self.p & Z == 0; self.branch(bus, c) }
            0x30 => { let c = self.p & N != 0; self.branch(bus, c) }
            0x10 => { let c = self.p & N == 0; self.branch(bus, c) }
            0x50 => { let c = self.p & V == 0; self.branch(bus, c) }
            0x70 => { let c = self.p & V != 0; self.branch(bus, c) }
            // Flags
            0x18 => { self.p &= !C; 2 }
            0x38 => { self.p |= C; 2 }
            0x58 => { self.p &= !I; 2 }
            0x78 => { self.p |= I; 2 }
            0xB8 => { self.p &= !V; 2 }
            0xD8 => { self.p &= !D; 2 }
            0xF8 => { self.p |= D; 2 }
            0xEA => 2, // NOP
            other => panic!(
                "unofficial/unsupported opcode {other:02X} at PC={:04X} — the ROM should only use official opcodes",
                self.pc.wrapping_sub(1)
            ),
        }
    }
}
