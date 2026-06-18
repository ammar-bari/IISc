#include <Wire.h>
#include <DFRobotIRPosition.h>

// Instantiate the camera object using the exact class name (no underscore)
DFRobotIRPosition myCamera;

// The pins driving the 4 IR LED target modules
const int ledPins[4] = {3, 5, 6, 9};

void setup() {
  // 1. Initialize Serial Communication for debugging
  Serial.begin(9600);
  while (!Serial);

  // 2. Drive all 4 IR LEDs solidly HIGH to act as active targets
  for (int i = 0; i < 4; i++) {
    pinMode(ledPins[i], OUTPUT);
    digitalWrite(ledPins[i], HIGH);
  }

  // 3. Initialize the IR Positioning Camera
  Serial.println("Initializing SEN0158 Camera...");
  myCamera.begin();
  Serial.println("Camera Ready. Tracking points...");
}

void loop() {
  // Request a data refresh from the camera sensor over I2C
  myCamera.requestPosition();

  // Check if the sensor has finished preparing the new data
  if (myCamera.available()) {
    
    // Read the X and Y coordinates for all 4 tracked points (Index 0, 1, 2, and 3)
    // The camera outputs coordinates scaled from 0 to 1023.
    // If a point is not detected in the frame, it will default to 1023, 1023.
    int x0 = myCamera.readX(0);
    int y0 = myCamera.readY(0);

    int x1 = myCamera.readX(1);
    int y1 = myCamera.readY(1);

    int x2 = myCamera.readX(2);
    int y2 = myCamera.readY(2);

    int x3 = myCamera.readX(3);
    int y3 = myCamera.readY(3);

    // Print out the coordinates to the Serial Monitor
    Serial.print("P0: ["); Serial.print(x0); Serial.print(", "); Serial.print(y0); Serial.print("]  |  ");
    Serial.print("P1: ["); Serial.print(x1); Serial.print(", "); Serial.print(y1); Serial.print("]  |  ");
    Serial.print("P2: ["); Serial.print(x2); Serial.print(", "); Serial.print(y2); Serial.print("]  |  ");
    Serial.print("P3: ["); Serial.print(x3); Serial.print(", "); Serial.print(y3); Serial.print("]");
    Serial.println();
  }

  // Polling delay
  delay(20);
}