#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DFRobot_MAX30102.h>

#define BUTTON_PIN 32
#define SESSION_DURATION 5000
#define SAMPLE_RATE 100
#define SAMPLE_INTERVAL (1000 / SAMPLE_RATE)
#define MAX_SAMPLES (SESSION_DURATION / SAMPLE_INTERVAL)

LiquidCrystal_I2C lcd(0x27, 16, 2);
DFRobot_MAX30102 particleSensor;

int32_t irBuffer[MAX_SAMPLES];
int32_t redBuffer[MAX_SAMPLES];
int sampleCount = 0;
bool sessionRunning = false;
unsigned long sessionStartTime = 0;
unsigned long lastSampleTime = 0;
int lastButtonState = HIGH;

int32_t SPO2 = 0;
int8_t SPO2Valid = 0;
int32_t heartRate = 0;
int8_t heartRateValid = 0;

unsigned long firstSampleMicros = 0;
unsigned long lastSampleMicros = 0;

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Wire.begin();
  Wire.setClock(400000);

  lcd.begin(16, 2);
  lcd.backlight();
  lcd.print("PPG BP Monitor");
  delay(2000);
  lcd.clear();
  lcd.print("Press to Start");

  while (!particleSensor.begin()) {
    Serial.println("MAX30102 not found");
    lcd.clear();
    lcd.print("Sensor Error");
    delay(1000);
    lcd.clear();
    lcd.print("Press to Start");
  }
  particleSensor.sensorConfiguration(
    60, SAMPLEAVG_8, MODE_MULTILED, SAMPLERATE_100, PULSEWIDTH_411, ADCRANGE_16384
  );
}

void loop() {
  int buttonState = digitalRead(BUTTON_PIN);
  if (buttonState != lastButtonState && buttonState == LOW && !sessionRunning) {
    sessionRunning = true;
    sessionStartTime = millis();
    lastSampleTime = millis();
    sampleCount = 0;
    firstSampleMicros = 0;

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Put Finger on");
    lcd.setCursor(0, 1);
    lcd.print("the sensor");
    delay(3000);

    lcd.clear();
    lcd.print("Session: Run");
    Serial.println("Session Start");
    delay(200);
  }
  lastButtonState = buttonState;

  if (sessionRunning && sampleCount < MAX_SAMPLES) {
    unsigned long currentMillis = millis();
    if (currentMillis - lastSampleTime >= SAMPLE_INTERVAL) {
      lastSampleTime = currentMillis;

      irBuffer[sampleCount] = particleSensor.getIR();
      redBuffer[sampleCount] = particleSensor.getRed();

      Serial.print("IR: ");
      Serial.println(irBuffer[sampleCount]);
      Serial.print("Red: ");
      Serial.println(redBuffer[sampleCount]);

      unsigned long nowMicros = micros();
      if (sampleCount == 0) firstSampleMicros = nowMicros;
      lastSampleMicros = nowMicros;

      sampleCount++;

      lcd.setCursor(0, 1);
      lcd.print("Samples: ");
      lcd.print(sampleCount);
      lcd.print("   ");
    }
  }

  if (sessionRunning && (millis() - sessionStartTime >= SESSION_DURATION || sampleCount >= MAX_SAMPLES)) {
    sessionRunning = false;

    particleSensor.heartrateAndOxygenSaturation(&SPO2, &SPO2Valid, &heartRate, &heartRateValid);

    lcd.clear();
    lcd.print(" Done");
    lcd.setCursor(0, 1);
    if (heartRateValid) {
      lcd.print("HR:");
      lcd.print(heartRate);
      lcd.print(" bpm");
    } else {
      lcd.print("HR: Invalid");
    }
    delay(3000);
    lcd.clear();
    lcd.print("SpO2: ");
    lcd.print(SPO2Valid ? SPO2 : -1);
    lcd.print(SPO2Valid ? "%" : "Invalid");
    delay(3000);
    lcd.clear();
    lcd.print("Press to Start");

    Serial.println("--- Session End ---");
    Serial.print("Heart Rate: ");
    Serial.print(heartRateValid ? heartRate : -1);
    Serial.println(heartRateValid ? " bpm" : " (Invalid)");
    Serial.print("SpO2: ");
    Serial.print(SPO2Valid ? SPO2 : -1);
    Serial.println(SPO2Valid ? "%" : " (Invalid)");
    Serial.print("Samples Collected: ");
    Serial.println(sampleCount);

    if (sampleCount > 1) {
  float elapsedSec = (lastSampleMicros - firstSampleMicros) / 1e6;
  float effectiveHz = (sampleCount - 1) / elapsedSec;

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Rate: ");
  lcd.print(effectiveHz, 1);
  lcd.print("Hz");
  lcd.setCursor(0, 1);
  lcd.print("Target: ");
  lcd.print(SAMPLE_RATE);
  lcd.print("Hz");
  delay(2000); // hold screen before next state, tune as needed
}
  }
}