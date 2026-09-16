$clangArgs = @(
  '--target=wasm32'
  '-Wl,--no-entry'
  '-Wl,--export=chip_init'
  '-Wl,--allow-undefined'
  '-nostdlib'
  '-O2'
  '-o'
  'chips/max30102.chip.wasm'
  'chips/max30102.chip.c'
)
& clang @clangArgs