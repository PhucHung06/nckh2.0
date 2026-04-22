import argparse
import os

import cv2
import numpy as np

from config.config import MULTI_VIDEO_INPUTS, VIDEO_INPUT, resolve_video_path
from config.roi_config import ROIS


DATA_DIR = os.path.dirname(__file__)
ROI_CONFIG_PATH = os.path.join(DATA_DIR, "config", "roi_config.py")
WINDOW_NAME = "Draw ROI"
VALID_GROUPS = ("A", "B", "C", "D")

LANE_SEQUENCE = [
    ("L", "Re trai"),
    ("M", "Di thang"),
    ("R", "Re phai"),
]

GROUP_COLORS = {
    "A": (0, 255, 255),
    "B": (0, 255, 0),
    "C": (255, 100, 0),
    "D": (0, 165, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Ve ROI cho tung video A/B/C/D")
    parser.add_argument(
        "group",
        nargs="?",
        default="A",
        help="Nhom ROI can ve: A, B, C, hoac D",
    )
    return parser.parse_args()


def normalize_group(group_name):
    group_name = (group_name or "").strip().upper()
    if group_name not in VALID_GROUPS:
        raise ValueError(f"Nhom ROI khong hop le: {group_name}. Hay chon mot trong {', '.join(VALID_GROUPS)}")
    return group_name


def extract_roi_group(roi_name):
    suffix = roi_name.split("_")[-1]
    return suffix[0]


def roi_sort_key(roi):
    lane_order = {"L": 0, "M": 1, "R": 2}
    suffix = roi["name"].split("_")[-1]
    return (extract_roi_group(roi["name"]), lane_order.get(suffix[-1], 99), roi["name"])


def get_group_video_path(group_name):
    group_video = resolve_video_path(MULTI_VIDEO_INPUTS.get(group_name))
    if group_video:
        return group_video
    return VIDEO_INPUT


def load_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Khong doc duoc frame dau tien tu video: {video_path}")

    return frame


def draw_polygon(image, points, color, label, closed=True, thickness=2):
    if not points:
        return

    pts = np.array(points, dtype=np.int32)
    if len(points) > 1:
        cv2.polylines(image, [pts], closed, color, thickness)

    for point in points:
        cv2.circle(image, tuple(point), 4, color, -1)

    cv2.putText(
        image,
        label,
        (points[0][0], max(20, points[0][1] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


class RoiEditorSession:
    def __init__(self, frame, group_name, existing_rois):
        self.frame = frame
        self.group_name = group_name
        self.current_points = []
        self.mouse_point = None
        self.saved_rois = [dict(roi) for roi in sorted(existing_rois, key=roi_sort_key)]

    def color(self):
        return GROUP_COLORS[self.group_name]

    def lane_index(self):
        return len(self.saved_rois)

    def lane_prompt(self):
        lane_idx = self.lane_index()
        if lane_idx >= len(LANE_SEQUENCE):
            return None
        return LANE_SEQUENCE[lane_idx]

    def prompt(self):
        lane_info = self.lane_prompt()
        if lane_info is None:
            return f"Da du 3 ROI cho nhom {self.group_name}"
        lane_code, lane_label = lane_info
        return f"Zone_{self.group_name}{lane_code} - {lane_label}"

    def add_point(self, x, y):
        if self.lane_prompt() is None:
            return False, "Da du 3 ROI. Nhan R neu muon ve lai."
        if len(self.current_points) >= 4:
            return False, "Moi ROI chi duoc 4 diem."
        self.current_points.append([int(x), int(y)])
        return True, ""

    def undo_point(self):
        if not self.current_points:
            return False, "Khong co diem de xoa."
        self.current_points.pop()
        return True, ""

    def clear_points(self):
        self.current_points = []

    def reset_group(self):
        self.saved_rois = []
        self.clear_points()

    def save_roi(self):
        lane_info = self.lane_prompt()
        if lane_info is None:
            return False, "Da du 3 ROI. Nhan R de xoa va ve lai."

        if len(self.current_points) != 4:
            return False, "Moi ROI can dung 4 diem."

        lane_code, _lane_label = lane_info
        saved_name = f"Zone_{self.group_name}{lane_code}"
        self.saved_rois.append(
            {
                "name": saved_name,
                "points": [point[:] for point in self.current_points],
                "color": self.color(),
            }
        )
        self.clear_points()
        return True, saved_name

    def can_finish(self):
        return len(self.saved_rois) == len(LANE_SEQUENCE) and not self.current_points


def render(session):
    canvas = session.frame.copy()

    for roi in session.saved_rois:
        draw_polygon(canvas, roi["points"], roi["color"], roi["name"], closed=True, thickness=3)

    if session.current_points:
        draw_polygon(canvas, session.current_points, session.color(), session.prompt(), closed=False, thickness=2)
        if session.mouse_point is not None and len(session.current_points) < 4:
            cv2.line(
                canvas,
                tuple(session.current_points[-1]),
                tuple(session.mouse_point),
                session.color(),
                1,
            )

    lines = [
        f"Video nhom {session.group_name}",
        f"Dang ve: {session.prompt()}",
        "Thu tu ROI: Re trai -> Di thang -> Re phai",
        "Chuot trai: them diem",
        "Enter hoac Space: luu ROI 4 diem",
        "Z hoac chuot phai: xoa diem cuoi",
        "C: xoa ROI dang ve",
        "R: xoa 3 ROI cua nhom nay va ve lai",
        "D: ghi vao roi_config.py khi da du 3 ROI",
        "ESC: thoat",
    ]

    height = 30 + 24 * len(lines)
    cv2.rectangle(canvas, (10, 10), (620, height), (30, 30, 30), -1)

    y = 34
    for line in lines:
        cv2.putText(canvas, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 24

    return canvas


def format_rois(rois):
    lines = ["ROIS = ["]
    for roi in rois:
        lines.append(
            f'    {{"name": "{roi["name"]}", "points": {roi["points"]}, "color": {roi["color"]}}},'
        )
    lines.append("")
    lines.append("]")
    return "\n".join(lines) + "\n"


def merge_rois(all_rois, group_name, replacement_rois):
    merged_rois = [dict(roi) for roi in all_rois if extract_roi_group(roi["name"]) != group_name]
    merged_rois.extend(dict(roi) for roi in replacement_rois)
    return sorted(merged_rois, key=roi_sort_key)


def write_roi_config(all_rois, group_name, replacement_rois):
    merged_rois = merge_rois(all_rois, group_name, replacement_rois)
    with open(ROI_CONFIG_PATH, "w", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write(format_rois(merged_rois))


def print_group_rois(group_name, rois):
    print()
    print(f"ROI nhom {group_name}:")
    for roi in sorted(rois, key=roi_sort_key):
        print(f'{roi["name"]}: {roi["points"]}')


def on_mouse(event, x, y, _flags, session):
    session.mouse_point = [int(x), int(y)]
    if event == cv2.EVENT_LBUTTONDOWN:
        ok, message = session.add_point(x, y)
        if not ok:
            print(message)
    elif event == cv2.EVENT_RBUTTONDOWN:
        ok, message = session.undo_point()
        if not ok:
            print(message)


def main():
    args = parse_args()
    group_name = normalize_group(args.group)
    video_path = get_group_video_path(group_name)
    existing_group_rois = [roi for roi in ROIS if extract_roi_group(roi["name"]) == group_name]
    frame = load_first_frame(video_path)
    session = RoiEditorSession(frame, group_name, existing_group_rois)

    print(f"Nhom dang ve: {group_name}")
    print(f"Video duoc dung: {video_path}")
    if existing_group_rois:
        print(f"Da tim thay {len(existing_group_rois)} ROI cu cho nhom {group_name}.")
        print("Nhan R neu muon xoa 3 ROI cu va ve lai tu dau.")
    else:
        print(f"Chua co ROI nao cho nhom {group_name}.")
    print("Moi ROI duoc xac dinh bang dung 4 diem.")
    print("Thu tu luu ROI: Re trai, Di thang, Re phai.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, session)

    while True:
        cv2.imshow(WINDOW_NAME, render(session))
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            print("Da thoat, khong ghi de roi_config.py")
            break

        if key in (13, 32):
            ok, message = session.save_roi()
            print(message)
            if not ok:
                continue

        if key in (ord("z"), ord("Z")):
            ok, message = session.undo_point()
            if not ok:
                print(message)

        if key in (ord("c"), ord("C")):
            session.clear_points()
            print("Da xoa ROI dang ve.")

        if key in (ord("r"), ord("R")):
            session.reset_group()
            print(f"Da xoa ROI nhom {group_name}. Hay ve lai 3 ROI moi.")

        if key in (ord("d"), ord("D")):
            if not session.can_finish():
                print("Can du 3 ROI va khong con ROI dang ve moi co the luu.")
                continue

            write_roi_config(ROIS, group_name, session.saved_rois)
            print_group_rois(group_name, session.saved_rois)
            print()
            print(f"Da ghi de ROI nhom {group_name} vao: {ROI_CONFIG_PATH}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
