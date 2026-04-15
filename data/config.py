'''
project/
│
├── main.py
├── config.py
├── roi_config.py
├── sumo_config/
│   └── your_simulation.sumocfg
├── video.mp4   (hoặc dùng camera)
'''

MODEL_PATH = r"C:\Users\hatun\Downloads\yolov26n_ver2.pt"   #  COPY PATH rồi PASTE lại vào trong r"   "
VIDEO_INPUT = r"C:\Users\hatun\Downloads\download.mp4"   # hoặc 0 nếu dùng webcam
SUMO_CFG =  r"C:\Users\hatun\Downloads\Test_NCKH_2\test2\sumo_config\run1.sumocfg"

CONF = 0.15

CLASS_NAMES = ['car']

# mapping ROI → edge vào trong SUMO
ROI_TO_EDGE = {
    "Zone_AL": "S2C",
    "Zone_AR": "S2C",
    "Zone_AM": "S2C",
    "Zone_BL": "W2C",
    "Zone_BR": "W2C",
    "Zone_BM": "W2C",
    "Zone_CL": "E2C",
    "Zone_CR": "E2C",
    "Zone_CM": "E2C",
    "Zone_DL": "N2C",
    "Zone_DR": "N2C",
    "Zone_DM": "N2C",
}

# Mỗi cạnh vào có 3 hướng rẽ:
# L = left, M = straight, R = right
INCOMING_TO_OUTGOING = {
    "S2C": {
        "L": "C2W",
        "M": "C2N",
        "R": "C2E",
    },
    "W2C": {
        "L": "C2N",
        "M": "C2E",
        "R": "C2S",
    },
    "E2C": {
        "L": "C2S",
        "M": "C2W",
        "R": "C2N",
    },
    "N2C": {
        "L": "C2E",
        "M": "C2S",
        "R": "C2W",
    },
}

# Theo network SUMO hiện tại:
# lane 0 = làn rẽ phải, lane 1 = làn đi thẳng, lane 2 = làn rẽ trái
TURN_TO_LANE = {
    "R": "0",
    "M": "1",
    "L": "2",
}
