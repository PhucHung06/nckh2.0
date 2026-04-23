import os

DATA_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(DATA_DIR, ".."))


def resolve_path(*parts):
    return os.path.abspath(os.path.join(PROJECT_DIR, *parts))


def resolve_video_path(path):
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_DIR, path))


MODEL_PATH = resolve_path("data", "model", "yolov26n_ver2.pt")  # Thay model khac neu can
VIDEO_INPUT = resolve_path("data", "video", "download.mp4")  # Cach cu: 1 video gom ca 4 huong
SUMO_CFG = resolve_path("data", "run1.sumocfg")

CONF = 0.3
DETECTION_INTERVAL = 5  # 1 = detect moi frame, 2 = detect moi 2 frame, 3 = detect moi 3 frame

CLASS_NAMES = ["car"]

# Neu ban tach thanh 4 video theo 4 huong, chi can dien duong dan vao day.
# A/B/C/D se tu dong map vao nhom ROI Zone_A*, Zone_B*, Zone_C*, Zone_D*.
# Co the dung duong dan tuyet doi hoac duong dan tuong doi tinh tu thu muc goc project.
MULTI_VIDEO_INPUTS = {
    "A": r"C:\Users\hatun\Downloads\nckh2.0\data\video\Road A.mp4",
    "B": r"C:\Users\hatun\Downloads\nckh2.0\data\video\Road B.mp4",
    "C": r"C:\Users\hatun\Downloads\nckh2.0\data\video\Road C.mp4",
    "D": r"C:\Users\hatun\Downloads\nckh2.0\data\video\Road D.mp4",
}

# Lech thoi gian bat dau cua tung video (giay). De 0.0 neu 4 video dong bo.
MULTI_VIDEO_OFFSETS = {
    "A": 0.0,
    "B": 0.0,
    "C": 0.0,
    "D": 0.0,
}


def build_video_sources():
    sources = []
    for group_name in ("A", "B", "C", "D"):
        video_path = resolve_video_path(MULTI_VIDEO_INPUTS.get(group_name))
        if not video_path:
            continue

        sources.append(
            {
                "source_id": f"cam_{group_name.lower()}",
                "label": f"Road {group_name}",
                "video_path": video_path,
                "roi_groups": [group_name],
                "time_offset": float(MULTI_VIDEO_OFFSETS.get(group_name, 0.0)),
            }
        )

    if sources:
        return sources

    return [
        {
            "source_id": "cam_all",
            "label": "All Roads",
            "video_path": VIDEO_INPUT,
            "roi_groups": ["A", "B", "C", "D"],
            "time_offset": 0.0,
        }
    ]


VIDEO_SOURCES = build_video_sources()

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
