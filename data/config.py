import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolov26n_ver2.pt")
VIDEO_INPUT = os.path.join(os.path.dirname(__file__), "ten_video.mp4") #Thêm video vào thư mục data rồi thay tên video vào đây
SUMO_CFG = os.path.join(os.path.dirname(__file__), "run1.sumocfg")

CONF = 0.15

CLASS_NAMES = ["car"]

# Map ROI to incoming SUMO edges.
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

# Each incoming edge has 3 turn directions.
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

# lane 0 = right, lane 1 = straight, lane 2 = left
TURN_TO_LANE = {
    "R": "0",
    "M": "1",
    "L": "2",
}
