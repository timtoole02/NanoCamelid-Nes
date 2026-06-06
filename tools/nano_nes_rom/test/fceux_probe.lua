-- FCEUX probe for NanoCamelid NES v3 (MMC5 chat).
-- Types NANO_QUESTION on the ROM's on-screen keyboard via the controller,
-- presses Start, waits for the done flag, and dumps the generated word IDs
-- to /tmp/nano_fceux_result.txt for comparison against the Rust reference.

local out = io.open("/tmp/nano_fceux_result.txt", "w")
local QUESTION = os.getenv("NANO_QUESTION") or "ARE YOU REAL?"
local VOCAB = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'-!?"

local STATUS, COUNT, PID = 0x07F0, 0x07F1, 0x07F2

local function press(btn)
  for _ = 1, 2 do
    joypad.set(1, { [btn] = true })
    emu.frameadvance()
  end
  for _ = 1, 2 do
    joypad.set(1, {})
    emu.frameadvance()
  end
end

-- boot
for _ = 1, 60 do
  joypad.set(1, {})
  emu.frameadvance()
end

-- type the question: 4x11 grid, cell index == char ID
local row, col = 0, 0
for i = 1, #QUESTION do
  local ch = QUESTION:sub(i, i):upper()
  local id = VOCAB:find(ch, 1, true)
  if id then
    id = id - 1
    local tr, tc = math.floor(id / 11), id % 11
    while row < tr do press("down"); row = row + 1 end
    while row > tr do press("up"); row = row - 1 end
    while col < tc do press("right"); col = col + 1 end
    while col > tc do press("left"); col = col - 1 end
    press("A")
  end
end
press("start")

local frame = 0
while frame < 6000 and memory.readbyte(STATUS) ~= 2 do
  joypad.set(1, {})
  emu.frameadvance()
  frame = frame + 1
end

local n = memory.readbyte(COUNT)
local ids = {}
for i = 0, n - 1 do ids[#ids + 1] = memory.readbyte(0x0300 + i) end
out:write(string.format("FINAL status=%d prompt_id=%d count=%d question=%s\n",
  memory.readbyte(STATUS), memory.readbyte(PID), n, QUESTION))
out:write("word_ids=" .. table.concat(ids, ",") .. "\n")
out:close()
os.exit(0)
