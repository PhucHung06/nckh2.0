// hardware/arduino/traffic_light.ino
// Giao thuc: "SET:gns:yns:gew:yew\n"  -->  "ACK:OK\n"
//            "FORCE:0\n" (0=NS_GO, 1=Yellow, 2=EW_GO, 3=Yellow) -> Ép phase

const int NS_GREEN = 3, NS_YELLOW = 4, NS_RED = 5;
const int EW_GREEN = 9, EW_YELLOW = 10, EW_RED = 11;

enum Phase { NS_GO, NS_YELLOW_PH, EW_GO, EW_YELLOW_PH };
Phase currentPhase = NS_GO;
int greenNS = 30, yellowNS = 4, greenEW = 30, yellowEW = 4;
unsigned long phaseStart = 0;

void setPhase(Phase p) {
  digitalWrite(NS_GREEN, 0); digitalWrite(NS_YELLOW, 0); digitalWrite(NS_RED, 0);
  digitalWrite(EW_GREEN, 0); digitalWrite(EW_YELLOW, 0); digitalWrite(EW_RED, 0);
  
  switch (p) {
    case NS_GO:
      digitalWrite(NS_GREEN, 1); digitalWrite(EW_RED, 1); break;
    case NS_YELLOW_PH:
      digitalWrite(NS_YELLOW, 1); digitalWrite(EW_RED, 1); break;
    case EW_GO:
      digitalWrite(EW_GREEN, 1); digitalWrite(NS_RED, 1); break;
    case EW_YELLOW_PH:
      digitalWrite(EW_YELLOW, 1); digitalWrite(NS_RED, 1); break;
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(NS_GREEN, OUTPUT); pinMode(NS_YELLOW, OUTPUT); pinMode(NS_RED, OUTPUT);
  pinMode(EW_GREEN, OUTPUT); pinMode(EW_YELLOW, OUTPUT); pinMode(EW_RED, OUTPUT);
  setPhase(currentPhase);
  phaseStart = millis();
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    
    if (cmd.startsWith("FORCE:")) {
      int p = cmd.substring(6).toInt();
      if(p >= 0 && p <= 3) {
        currentPhase = (Phase)p;
        setPhase(currentPhase);
        phaseStart = millis(); // Reset timer đếm ngược
        Serial.println("ACK:FORCED");
      }
    }
    else if (cmd.startsWith("SET:")) {
      int vals[4];
      int idx = 0;
      String s = cmd.substring(4);
      while (s.length() > 0 && idx < 4) {
        int sep = s.indexOf(':');
        if (sep < 0) {
          vals[idx++] = s.toInt();
          break;
        } else {
          vals[idx++] = s.substring(0, sep).toInt();
          s = s.substring(sep + 1);
        }
      }
      
      if (idx == 4) {
        greenNS = vals[0]; yellowNS = vals[1]; 
        greenEW = vals[2]; yellowEW = vals[3];
        Serial.println("ACK:OK");
      } else {
        Serial.println("ERR:INVALID_PARAMS");
      }
    }
    else if (cmd == "STATUS") {
      Serial.print("STATUS:");
      Serial.print(currentPhase);
      Serial.print(":");
      Serial.println(millis() - phaseStart);
    }
  }

  // Tự động đếm ngược nội bộ (Failsafe)
  unsigned long elapsed = (millis() - phaseStart) / 1000UL;
  int durations[] = {greenNS, yellowNS, greenEW, yellowEW};
  if (elapsed >= (unsigned long)durations[currentPhase]) {
    currentPhase = (Phase)((currentPhase + 1) % 4);
    setPhase(currentPhase);
    phaseStart = millis();
  }
}
