/* This driver reads quaternion data from the MPU6060 and sends
   Open Sound Control messages.

  GY-521  NodeMCU
  MPU6050 devkit 1.0
  board   Lolin         Description
  ======= ==========    ====================================================
  VCC     VU (5V USB)   Not available on all boards so use 3.3V if needed.
  GND     G             Ground
  SCL     D1 (GPIO05)   I2C clock
  SDA     D2 (GPIO04)   I2C data
  XDA     not connected
  XCL     not connected
  AD0     not connected
  INT     D8 (GPIO15)   Interrupt pin

*/

#if defined(ESP8266)
#include <ESP8266WiFi.h>
#else
#include <WiFi.h>
#endif
#include <ESP32Servo.h>
#include <WiFiClient.h>
#include <WiFiAP.h>

// I2Cdev and MPU6050 must be installed as libraries, or else the .cpp/.h files
// for both classes must be in the include path of your project
#include "I2Cdev.h"

#include "MPU6050_6Axis_MotionApps20.h"
//#include "MPU6050.h" // not necessary if using MotionApps include file

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

// class default I2C address is 0x68
// specific I2C addresses may be passed as a parameter here
// AD0 low = 0x68 (default for SparkFun breakout and InvenSense evaluation board)
// AD0 high = 0x69
MPU6050 mpu;
//MPU6050 mpu(0x69); // <-- use for AD0 high

/* =========================================================================
   NOTE: In addition to connection 5/3.3v, GND, SDA, and SCL, this sketch
   depends on the MPU-6050's INT pin being connected to the ESP8266 GPIO15
   pin.
 * ========================================================================= */

// MPU control/status vars
bool dmpReady = false;  // set true if DMP init was successful
uint8_t mpuIntStatus;   // holds actual interrupt status byte from MPU
uint8_t devStatus;      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[64]; // FIFO storage buffer

/////////////////////////////////////////ultrasonic vars
const double SOUND_VELOCITY=0.034826;

const int echoPin[]={32, 12};
const int trigPin[]={33, 14};


/////////////////////////////servo declaration and vars
Servo s1;
Servo s2;
Servo s3;
Servo s4;

int pos1 = 110;
int pos2 = 110;
int pos3 = 110;
int pos4 = 110;

int def =20;
int Max= 180;
int pitch=0;

/////////////Servo funtions/////////////////
void s1w(int pos){
  s1.write(180-pos);
}

void s2w(int pos){
  s2.write(pos);
}

void s3w(int pos){
  s3.write(180-pos);
}

void s4w(int pos){
  s4.write(pos);
}
//////////////////////////////////////////

/////////////////////////////////////////// Wi-Fi credentials
const char* ssid = "abcde";  // Replace with your desired SSID
const char* password = "12345678";  // Replace with your desired password
WiFiServer server(80);
WiFiClient client;


// orientation/motion vars
Quaternion q;           // [w, x, y, z]         quaternion container
VectorInt16 aa;         // [x, y, z]            accel sensor measurements
VectorInt16 aaReal;     // [x, y, z]            gravity-free accel sensor measurements
VectorInt16 aaWorld;    // [x, y, z]            world-frame accel sensor measurements
VectorFloat gravity;    // [x, y, z]            gravity vector

#define OUTPUT_READABLE_YAWPITCHROLL

#ifdef OUTPUT_READABLE_YAWPITCHROLL
float ypr[3];           // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector
#endif

#define INTERRUPT_PIN 3 // use pin 15 on ESP8266

const char DEVICE_NAME[] = "mpu6050";

// ================================================================
// ===               INTERRUPT DETECTION ROUTINE                ===
// ================================================================

volatile bool mpuInterrupt = false;     // indicates whether MPU interrupt pin has gone high
void ICACHE_RAM_ATTR dmpDataReady() {
    mpuInterrupt = true;
}
///////////////////////////////////////////////////////////////////////////////////
                  ////////////distance measurement///////////////
///////////////////////////////////////////////////////////////////////////////////
double dist(int c){
  float distance, duration;
  digitalWrite(trigPin[c], LOW);
  delayMicroseconds(2);
  
  digitalWrite(trigPin[c], HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin[c], LOW);
  
  duration = pulseIn(echoPin[c], HIGH);
  
  distance = duration * SOUND_VELOCITY/2;
  return distance;
}

///////////////////////////////////////////////////////////////////////////////////
                      //////////////reset function/////////////
///////////////////////////////////////////////////////////////////////////////////
void reset(){
  int pos=30;
  s1w(Max);
  s2w(def);
  s3w(def);
  s4w(def);
  if(dist(1)<50)
    return;
  delay(1000);
  
  s1w(def);
  s2w(def);
  s3w(Max-pos);
  s4w(def);
  if(dist(1)<50)
    return;
  delay(1000);

  s1w(def);
  s2w(def);
  s3w(def);
  s4w(Max-20);  //offset for leg balance, accounting for asymmetry
  if(dist(1)<50)
    return;
  delay(1000);
  
  s1w(def);
  s2w(Max-pos);
  s3w(def);
  s4w(def);
  if(dist(1)<50)
    return;
  delay(1000);
}
///////////////////////////////////////////////////////////////////////////////////
                      //////////////mpu setup/////////////
///////////////////////////////////////////////////////////////////////////////////
void mpu_setup()
{
  // join I2C bus (I2Cdev library doesn't do this automatically)
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
  Wire.begin();
  Wire.setClock(400000); // 400kHz I2C clock. Comment this line if having compilation difficulties
#elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
  Fastwire::setup(400, true);
#endif

  // initialize device
  // Serial.println(F("Initializing I2C devices..."));
  mpu.initialize();
  pinMode(INTERRUPT_PIN, INPUT);

  // verify connection
  // Serial.println(F("Testing device connections..."));
  // Serial.println(mpu.testConnection() ? F("MPU6050 connection successful") : F("MPU6050 connection failed"));

  // load and configure the DMP
  // Serial.println(F("Initializing DMP..."));
  devStatus = mpu.dmpInitialize();

  // supply your own gyro offsets here, scaled for min sensitivity
  mpu.setXGyroOffset(220);
  mpu.setYGyroOffset(76);
  mpu.setZGyroOffset(-85);
  mpu.setZAccelOffset(1788); // 1688 factory default for my test chip

  // make sure it worked (returns 0 if so)
  if (devStatus == 0) {
    // turn on the DMP, now that it's ready
    // Serial.println(F("Enabling DMP..."));
    mpu.setDMPEnabled(true);

    // enable Arduino interrupt detection
    // Serial.println(F("Enabling interrupt detection (Arduino external interrupt 0)..."));
    attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
    mpuIntStatus = mpu.getIntStatus();

    // set our DMP Ready flag so the main loop() function knows it's okay to use it
    // Serial.println(F("DMP ready! Waiting for first interrupt..."));
    dmpReady = true;

    // get expected DMP packet size for later comparison
    packetSize = mpu.dmpGetFIFOPacketSize();
  } else {
    // ERROR!
    // 1 = initial memory load failed
    // 2 = DMP configuration updates failed
    // (if it's going to break, usually the code will be 1)
    // Serial.print(F("DMP Initialization failed (code "));
    // Serial.print(devStatus);
    // Serial.println(F(")"));
    // client.println("failure");
  }
}
///////////////////////////////////////////////////////////////////////////////////
                        /////////////setup//////////////
///////////////////////////////////////////////////////////////////////////////////
void setup(void)
{
  Serial.begin(115200);

  // Set ESP32 in Access Point mode with the given SSID and password
  WiFi.softAP(ssid, password);
  delay(5000);
  
  IPAddress IP = WiFi.softAPIP();
  Serial.print("AP IP address: ");
  Serial.println(IP);

  server.begin();

  pinMode(trigPin[0], OUTPUT); // Sets the trigPin as an Output
  pinMode(trigPin[1], OUTPUT); // Sets the trigPin as an Output
  pinMode(echoPin[0], INPUT);
  pinMode(echoPin[1], INPUT);

  s1.attach(11);
  s2.attach(10);
  s3.attach(9);
  s4.attach(6); 
  
  s1.write(180-pos1);
  s2.write(pos2);
  s3.write(180-pos3);
  s4.write(pos4);
  delay(5000);
  mpu_setup();
}
///////////////////////////////////////////////////////////////////////////////////
                       /////////////mpu loop//////////////
///////////////////////////////////////////////////////////////////////////////////
String mpu_loop()
{ 
  mpuInterrupt = true;
  // if programming failed, don't try to do anything
  if (!dmpReady) return "";

  // wait for MPU interrupt or extra packet(s) available
  if (!mpuInterrupt && fifoCount < packetSize) return "";

  // reset interrupt flag and get INT_STATUS byte
  mpuInterrupt = false;
  mpuIntStatus = mpu.getIntStatus();

  // get current FIFO count
  fifoCount = mpu.getFIFOCount();

  // check for overflow (this should never happen unless our code is too inefficient)
  if ((mpuIntStatus & 0x10) || fifoCount == 1024) {
    // reset so we can continue cleanly
    mpu.resetFIFO();
    // Serial.println(F("FIFO overflow!"));

    // otherwise, check for DMP data ready interrupt (this should happen frequently)
  } else if (mpuIntStatus & 0x02) {
    // wait for correct available data length, should be a VERY short wait
    while (fifoCount < packetSize) fifoCount = mpu.getFIFOCount();

    // read a packet from FIFO
    mpu.getFIFOBytes(fifoBuffer, packetSize);

    // track FIFO count here in case there is > 1 packet available
    // (this lets us immediately read more without waiting for an interrupt)
    fifoCount -= packetSize;

#ifdef OUTPUT_READABLE_YAWPITCHROLL
    // display Euler angles in degrees
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
    // Serial.print(ypr[0] * 180/M_PI);
    // Serial.print("\t\t");
    // Serial.print(ypr[1] * 180/M_PI);
    // Serial.print("\t");
    // Serial.println(ypr[2] * 180/M_PI);

    //////////////////////////////////////Stuff necessaray for mpu ypr ends here///////////////////////////////////////////////////////////

    String data= String(ypr[0]* 180/M_PI)+" "+String(ypr[1]* 180/M_PI)+" "+String(ypr[2]* 180/M_PI)+" "+String(dist(0))+" "+String(dist(1));
    // client.println(data);
    return data;
#endif

// String response = client.readStringUntil('\n');
// // Serial.println(response);
// if(response=="RESET"){
//   Serial.println("reset");
//   delay(5000);
//   client.println("done");
//   s1w(0);
//   s2w(0);
//   s3w(0);
//   s4w(0);
// }
// else
//   Serial.println(response);
// delay(500);
  }
}
///////////////////////////////////////////////////////////////////////////////////
                            ///////////////////////////
///////////////////////////////////////////////////////////////////////////////////
void loop(void)
{
  // mpu_loop();
  client = server.available();
  while (client) {
    // Serial.println("New client connected");
    // client.println("Hello from ESP32 Server!");
    // String data= String(dist(0))+" "+String(dist(1));
    String data= mpu_loop();
    client.println(data);
  }
}
