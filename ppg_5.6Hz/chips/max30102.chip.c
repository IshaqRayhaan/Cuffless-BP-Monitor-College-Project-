#include "wokwi-api.h"

#define REG_INTSTAT1  0x00
#define REG_INTEN1    0x02
#define REG_FIFOWR    0x04
#define REG_FIFOOVF   0x05
#define REG_FIFORD    0x06
#define REG_FIFODATA  0x07
#define REG_FIFOCONFIG 0x08
#define REG_MODECFG   0x09
#define REG_PARTID    0xFF
#define REG_REVID     0xFE

const int ADDRESS = 0x57;
#define BASE_TICK_US 10000 // 10ms per raw FIFO sample. getIR()/getRed() each
                            // block a full tick -> 80ms/sample -> 12.5Hz,
                            // reproducing the double-poll bottleneck vs
                            // the intended 100Hz.

typedef struct {
  pin_t pin_int;
  timer_t sample_timer;
  uint8_t reg_ptr;
  bool got_reg_ptr;
  uint8_t fifo_wr, fifo_rd;
  uint32_t sample_idx;
  bool int_asserted;
  uint8_t fifo_config;
} chip_state_t;

static chip_state_t chip_instance; // static allocation instead of malloc

// crude sine approximation, freestanding-safe (no math.h)
static double fake_sin(double x) {
  const double PI = 3.14159265358979;
  while (x > PI) x -= 2*PI;
  while (x < -PI) x += 2*PI;
  double absx = (x < 0) ? -x : x;
  return 16 * x * (PI - absx) / (5*PI*PI - 4*x*(PI - absx));
}

static void set_int(chip_state_t *chip, bool asserted) {
  chip->int_asserted = asserted;
  pin_write(chip->pin_int, asserted ? LOW : HIGH);
}

static void on_sample_tick(void *user_data) {
  chip_state_t *chip = (chip_state_t*)user_data;
  chip->fifo_wr = (chip->fifo_wr + 1) % 32;
  chip->sample_idx++;
  set_int(chip, true);
}

static bool chip_i2c_connect(void *user_data, uint32_t address, bool read) {
  chip_state_t *chip = (chip_state_t*)user_data;
  (void)address;
  if (!read) {
    chip->got_reg_ptr = false;
  }
  return true;
}

static uint8_t chip_i2c_read(void *user_data) {
  chip_state_t *chip = (chip_state_t*)user_data;
  switch (chip->reg_ptr) {
    case REG_PARTID: return 0x15;
    case REG_REVID:  return 0x03;
    case REG_FIFOWR: return chip->fifo_wr;
    case REG_FIFORD: return chip->fifo_rd;
    case REG_FIFOOVF: return 0;
    case REG_FIFOCONFIG: return chip->fifo_config;
    case REG_INTSTAT1: {
      uint8_t v = chip->int_asserted ? 0x40 : 0x00;
      set_int(chip, false);
      return v;
    }
    case REG_FIFODATA: {
      double t = chip->sample_idx * 0.04;
      uint32_t red = 50000 + (uint32_t)(8000 * fake_sin(2 * 3.14159 * 1.2 * t));
      uint32_t ir  = 45000 + (uint32_t)(9000 * fake_sin(2 * 3.14159 * 1.2 * t + 0.3));
      static int byte_phase = 0;
      uint32_t val = (byte_phase < 3) ? red : ir;
      int shift = (2 - (byte_phase % 3)) * 8;
      uint8_t b = (val >> shift) & 0xFF;
      byte_phase = (byte_phase + 1) % 6;
      if (byte_phase == 0) chip->fifo_rd = (chip->fifo_rd + 1) % 32;
      return b;
    }
    default: return 0x00;
  }
}

static void apply_fifo_config(chip_state_t *chip, uint8_t data) {
  chip->fifo_config = data;
  uint8_t smp_ave_bits = (data >> 5) & 0x07;
  uint8_t smp_ave = 1 << smp_ave_bits;
  if (smp_ave > 32) smp_ave = 32;

  uint32_t new_period_us = BASE_TICK_US * smp_ave;
  timer_stop(chip->sample_timer);
  timer_start(chip->sample_timer, new_period_us, true);
}

static bool chip_i2c_write(void *user_data, uint8_t data) {
  chip_state_t *chip = (chip_state_t*)user_data;
  if (!chip->got_reg_ptr) {
    chip->reg_ptr = data;
    chip->got_reg_ptr = true;
    return true;
  }
    if (chip->reg_ptr == REG_FIFOCONFIG) {
    apply_fifo_config(chip, data);
  }
  chip->reg_ptr++;
  return true;
}

static void chip_i2c_disconnect(void *user_data) {
  (void)user_data;
}

void chip_init(void) {
  chip_state_t *chip = &chip_instance;
  chip->reg_ptr = 0;
  chip->got_reg_ptr = false;
  chip->fifo_wr = 0;
  chip->fifo_rd = 0;
  chip->sample_idx = 0;
  chip->int_asserted = false;
  chip->fifo_config = 0;

  chip->pin_int = pin_init("INT", OUTPUT_HIGH);

  const i2c_config_t i2c_config = {
    .user_data = chip,
    .address = 0x57,
    .scl = pin_init("SCL", INPUT),
    .sda = pin_init("SDA", INPUT),
    .connect = chip_i2c_connect,
    .read = chip_i2c_read,
    .write = chip_i2c_write,
    .disconnect = chip_i2c_disconnect,
  };
  i2c_init(&i2c_config);

  const timer_config_t timer_config = {
    .callback = on_sample_tick,
    .user_data = chip,
  };
  chip->sample_timer = timer_init(&timer_config);
  timer_start(chip->sample_timer, BASE_TICK_US, true);
}