import os
import string

import cv2
import numpy as np


DATA_DIR = os.path.dirname(__file__)
VIDEO_PATH = os.path.join(DATA_DIR, "video", "download.mp4")    #Thay đường dẫn video khác nếu cần
ROI_CONFIG_PATH = os.path.join(DATA_DIR, "roi_config.py")
WINDOW_NAME = "Draw ROI"

LANE_SEQUENCE = [
    ("L", "Re trai"),
    ("M", "Di thang"),
    ("R", "Re phai"),
]

GROUP_COLORS = [
    (0, 255, 255),
    (0, 255, 0),
    (255, 100, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 255, 0),
    (255, 128, 0),
]


class RoiSession:
    def __init__(self, frame):
        self.frame = frame
        self.current_points = []
        self.mouse_point = None
        self.group_index = 0
        self.stage = "large"
        self.large_rois = []
        self.small_rois = []

    def group_name(self):
        if self.group_index < len(string.ascii_uppercase):
            return string.ascii_uppercase[self.group_index]
        return f"G{self.group_index + 1}"

    def color(self):
        return GROUP_COLORS[self.group_index % len(GROUP_COLORS)]

    def lane_index(self):
        current_group = self.group_name()
        return len([roi for roi in self.small_rois if roi["group"] == current_group])

    def prompt(self):
        group = self.group_name()
        if self.stage == "large":
            return f"ROI lon cho huong {group}"
        lane_code, lane_label = LANE_SEQUENCE[self.lane_index()]
        return f"Zone_{group}{lane_code} - {lane_label}"

    def add_point(self, x, y):
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

    def save_roi(self):
        if len(self.current_points) != 4:
            return False, "Moi ROI can dung 4 diem."

        group = self.group_name()
        points = [point[:] for point in self.current_points]
        color = self.color()

        if self.stage == "large":
            self.large_rois.append(
                {
                    "name": f"Road_{group}",
                    "group": group,
                    "points": points,
                    "color": color,
                }
            )
            saved_name = f"Road_{group}"
            self.stage = "small"
        else:
            lane_code, _lane_label = LANE_SEQUENCE[self.lane_index()]
            saved_name = f"Zone_{group}{lane_code}"
            self.small_rois.append(
                {
                    "name": saved_name,
                    "group": group,
                    "points": points,
                    "color": color,
                }
            )
            if self.lane_index() == 3:
                self.group_index += 1
                self.stage = "large"

        self.clear_points()
        return True, saved_name

    def can_finish(self):
        return self.stage == "large" and not self.current_points


def load_first_frame():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {VIDEO_PATH}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Khong doc duoc frame dau tien tu video: {VIDEO_PATH}")

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


def render(session):
    canvas = session.frame.copy()

    for roi in session.large_rois:
        draw_polygon(canvas, roi["points"], roi["color"], roi["name"], closed=True, thickness=2)

    for roi in session.small_rois:
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
        f"Dang ve: {session.prompt()}",
        "Chuoi thao tac: ROI lon -> Re trai -> Di thang -> Re phai",
        "Chuot trai: them diem",
        "Enter hoac Space: luu ROI 4 diem",
        "Z hoac chuot phai: xoa diem cuoi",
        "C: xoa ROI dang ve",
        "D: ket thuc sau khi xong mot cum",
        "ESC: thoat",
    ]

    height = 30 + 24 * len(lines)
    cv2.rectangle(canvas, (10, 10), (560, height), (30, 30, 30), -1)

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


def write_roi_config(rois):
    with open(ROI_CONFIG_PATH, "w", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write(format_rois(rois))


def print_rois(rois, large_rois):
    print()
    print("ROI lon:")
    for roi in large_rois:
        print(f'{roi["name"]}: {roi["points"]}')

    print()
    print("ROI lane:")
    for roi in rois:
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
    frame = load_first_frame()
    session = RoiSession(frame)

    print(f"Lay frame dau tien tu: {VIDEO_PATH}")
    print("Moi ROI duoc xac dinh bang dung 4 diem.")
    print("Moi cum gom: 1 ROI lon, sau do 3 ROI nho theo thu tu Re trai, Di thang, Re phai.")

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

        if key in (ord("d"), ord("D")):
            if not session.can_finish():
                print("Hay ve xong cum hien tai roi moi nhan D.")
                continue

            write_roi_config(session.small_rois)
            print_rois(session.small_rois, session.large_rois)
            print()
            print(f"Da ghi de ROIS vao: {ROI_CONFIG_PATH}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
