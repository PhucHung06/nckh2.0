import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import supervision as sv
import traci
from ultralytics import YOLO

from config.config import CONF, DETECTION_INTERVAL, INCOMING_TO_OUTGOING, MODEL_PATH, ROI_TO_EDGE, SUMO_CFG, TURN_TO_LANE
from config.config import VIDEO_SOURCES
from config.roi_config import ROIS

DATA_DIR = os.path.dirname(__file__)
TRACKED_JSON = os.path.join(DATA_DIR, "..", "data", "tracked_vehicles.json")
ROUTES_XML = os.path.join(DATA_DIR, "..", "data", "xml", "yolo_routes.rou.xml")
VIDEO_WINDOW_NAME = "Tracking Multi-View"
DISPLAY_CELL_SIZE = (640, 360)
COUNTED_CLASSES = ("bus", "car", "motor", "truck")
TRACK_BOX_COLOR = (70, 120, 255)
TRACK_TEXT_BG_COLOR = (70, 120, 255)
PANEL_BG_COLOR = (35, 35, 35)


def build_route_specs():
    route_specs = {}
    for roi_name, incoming_edge in ROI_TO_EDGE.items():
        turn = roi_name.split("_")[-1][-1]
        outgoing_edge = INCOMING_TO_OUTGOING[incoming_edge][turn]
        route_specs[roi_name] = {
            "route_id": f"route_{roi_name}",
            "incoming_edge": incoming_edge,
            "outgoing_edge": outgoing_edge,
            "turn": turn,
            "depart_lane": TURN_TO_LANE[turn],
        }
    return route_specs


def extract_roi_group(roi_name):
    suffix = roi_name.split("_")[-1]
    return suffix[0]


def build_track_spawn_key(source_id, roi_name, track_id):
    return f"{source_id}_{roi_name}_{int(track_id)}"


def get_model_class_name(model, class_id):
    names = getattr(model, "names", {})
    if class_id is None:
        return "vehicle"
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))
    if isinstance(names, (list, tuple)) and 0 <= int(class_id) < len(names):
        return str(names[int(class_id)])
    return str(class_id)


def normalize_vehicle_class(class_name):
    normalized = str(class_name).strip().lower().replace("-", " ").replace("_", " ")
    alias_map = {
        "bicycle": "motor",
        "bike": "motor",
        "bus": "bus",
        "car": "car",
        "motorbike": "motor",
        "motorcycle": "motor",
        "motor": "motor",
        "truck": "truck",
        "van": "truck",
    }
    if normalized in alias_map:
        return alias_map[normalized]
    if "motor" in normalized or "bike" in normalized:
        return "motor"
    if "truck" in normalized or "van" in normalized:
        return "truck"
    if "bus" in normalized:
        return "bus"
    if "car" in normalized:
        return "car"
    return normalized


def build_vehicle_mask(detections, model):
    if len(detections) == 0:
        return np.zeros(0, dtype=bool)

    class_ids = detections.class_id
    mask = []
    for index in range(len(detections)):
        class_id = None if class_ids is None else class_ids[index]
        class_name = get_model_class_name(model, class_id)
        mask.append(normalize_vehicle_class(class_name) in COUNTED_CLASSES)
    return np.array(mask, dtype=bool)


def build_display_label(entry):
    class_name = entry["class_name"]
    vehicle_ids = entry.get("vehicle_ids") or []
    if vehicle_ids:
        return f"{class_name} {vehicle_ids[0]}"

    track_id = entry.get("track_id")
    if track_id is None:
        return class_name
    return f"{class_name} #{int(track_id)}"


def draw_tracked_detections(frame, entries):
    for entry in entries:
        x1, y1, x2, y2 = [int(value) for value in entry["bbox"]]
        label = build_display_label(entry)

        cv2.rectangle(frame, (x1, y1), (x2, y2), TRACK_BOX_COLOR, 2)

        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = max(4, x1)
        text_y = max(26, y1 - 8)
        bg_top = max(0, text_y - text_size[1] - 8)
        bg_bottom = min(frame.shape[0], text_y + baseline - 2)
        bg_right = min(frame.shape[1], text_x + text_size[0] + 10)

        cv2.rectangle(frame, (text_x - 4, bg_top), (bg_right, bg_bottom), TRACK_TEXT_BG_COLOR, -1)
        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def compute_roi_stats(entries, rois):
    stats_by_roi = {
        roi["name"]: {
            "roi": roi,
            "counts": {class_name: 0 for class_name in COUNTED_CLASSES},
            "total": 0,
        }
        for roi in rois
    }

    for entry in entries:
        class_name = entry["class_name"]
        for roi_name in entry.get("roi_names", []):
            roi_stats = stats_by_roi.get(roi_name)
            if roi_stats is None:
                continue
            if class_name in roi_stats["counts"]:
                roi_stats["counts"][class_name] += 1
                roi_stats["total"] += 1

    return [stats_by_roi[roi["name"]] for roi in rois]


def draw_roi_stats_panel(frame, roi_stats):
    if not roi_stats:
        return

    line_height = 28
    panel_width = 190
    panel_height = 18 + len(roi_stats) * (line_height * (len(COUNTED_CLASSES) + 2) + 10)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), PANEL_BG_COLOR, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 35
    for roi_stat in roi_stats:
        roi = roi_stat["roi"]
        roi_name = roi["name"]
        roi_color = tuple(int(channel) for channel in roi["color"])

        cv2.putText(frame, f"[ {roi_name} ]", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, roi_color, 2)
        y += line_height

        for class_name in COUNTED_CLASSES:
            count = roi_stat["counts"][class_name]
            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (28, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (235, 235, 235),
                2,
            )
            y += line_height

        cv2.putText(
            frame,
            f"Total: {roi_stat['total']}",
            (28, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            roi_color,
            2,
        )
        y += line_height + 8


def infer_tracked_detections(model, tracker, frame):
    results = model(frame, conf=CONF, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    if len(detections) == 0:
        return detections

    vehicle_mask = build_vehicle_mask(detections, model)
    detections = detections[vehicle_mask]
    if len(detections) == 0:
        return detections

    return tracker.update_with_detections(detections)


def build_frame_track_entries(source, detections, zones, model):
    roi_names_by_track = {}

    for roi, zone in zip(source["rois"], zones):
        mask = zone.trigger(detections=detections)
        zone_detections = detections[mask]
        for index in range(len(zone_detections)):
            tracker_ids = zone_detections.tracker_id
            track_id = None if tracker_ids is None else tracker_ids[index]
            if track_id is None:
                continue
            track_id = int(track_id)
            roi_names_by_track.setdefault(track_id, set()).add(roi["name"])

    entries = []
    for index in range(len(detections)):
        tracker_ids = detections.tracker_id
        class_ids = detections.class_id
        track_id = None if tracker_ids is None else tracker_ids[index]
        if track_id is None:
            continue

        track_id = int(track_id)
        roi_names = sorted(roi_names_by_track.get(track_id, set()))
        if not roi_names:
            continue

        class_id = None if class_ids is None else class_ids[index]
        class_name = normalize_vehicle_class(get_model_class_name(model, class_id))
        entries.append(
            {
                "track_id": track_id,
                "class_name": class_name,
                "bbox": [int(value) for value in detections.xyxy[index].astype(int).tolist()],
                "roi_names": roi_names,
                "spawn_keys": [
                    build_track_spawn_key(source["source_id"], roi_name, track_id)
                    for roi_name in roi_names
                ],
                "vehicle_ids": [],
            }
        )

    return entries


def build_source_contexts():
    rois_by_group = {}
    for roi in ROIS:
        rois_by_group.setdefault(extract_roi_group(roi["name"]), []).append(roi)

    contexts = []
    for source in VIDEO_SOURCES:
        roi_groups = source.get("roi_groups") or []
        source_rois = []
        for group_name in roi_groups:
            group_rois = rois_by_group.get(group_name)
            if not group_rois:
                raise ValueError(f"Khong tim thay ROI group {group_name} cho source {source['source_id']}")
            source_rois.extend(sorted(group_rois, key=lambda item: item["name"]))

        video_path = source["video_path"]
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Khong tim thay video: {video_path}")

        contexts.append(
            {
                "source_id": source["source_id"],
                "label": source.get("label", source["source_id"]),
                "video_path": video_path,
                "time_offset": float(source.get("time_offset", 0.0)),
                "roi_groups": roi_groups,
                "rois": source_rois,
            }
        )

    if not contexts:
        raise ValueError("Khong co video nao duoc cau hinh trong VIDEO_SOURCES")

    return contexts


def export_tracked_vehicles(records, sources):
    payload = {
        "sumo_cfg": SUMO_CFG,
        "vehicle_count": len(records),
        "sources": [
            {
                "source_id": source["source_id"],
                "label": source["label"],
                "video_path": source["video_path"],
                "roi_groups": source["roi_groups"],
                "time_offset": source["time_offset"],
            }
            for source in sources
        ],
        "vehicles": records,
    }
    if len(sources) == 1:
        payload["source_video"] = sources[0]["video_path"]

    with open(TRACKED_JSON, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def export_route_file(records, route_specs):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        '    <vType id="yolo_car" accel="2.6" decel="4.5" sigma="0.5" '
        'length="5.0" minGap="2.5" maxSpeed="13"/>',
    ]

    for spec in route_specs.values():
        lines.append(
            f'    <route id="{spec["route_id"]}" '
            f'edges="{spec["incoming_edge"]} {spec["outgoing_edge"]}"/>'
        )

    for record in sorted(records, key=lambda item: (item["depart"], item["vehicle_id"])):
        lines.append(
            f'    <vehicle id="{record["vehicle_id"]}" type="yolo_car" '
            f'depart="{record["depart"]:.2f}" route="{record["route_id"]}" '
            f'departLane="{record["depart_lane"]}"/>'
        )

    lines.append("</routes>")

    with open(ROUTES_XML, "w", encoding="utf-8") as file_obj:
        file_obj.write("\n".join(lines) + "\n")


def ensure_route_file(route_specs):
    export_route_file([], route_specs)


def update_sumo_config(max_depart):
    tree = ET.parse(SUMO_CFG)
    root = tree.getroot()

    input_node = root.find("./input")
    if input_node is None:
        input_node = ET.SubElement(root, "input")

    route_node = input_node.find("./route-files")
    if route_node is None:
        route_node = ET.SubElement(input_node, "route-files")
    route_rel_path = os.path.relpath(ROUTES_XML, start=os.path.dirname(SUMO_CFG))
    route_node.set("value", route_rel_path.replace(os.path.sep, "/"))

    time_node = root.find("./time")
    if time_node is None:
        time_node = ET.SubElement(root, "time")

    begin_node = time_node.find("./begin")
    if begin_node is None:
        begin_node = ET.SubElement(time_node, "begin")
    begin_node.set("value", "0")

    end_node = time_node.find("./end")
    if end_node is None:
        end_node = ET.SubElement(time_node, "end")
    simulation_end = max(3600, int(max_depart) + 300)
    end_node.set("value", str(simulation_end))

    tree.write(SUMO_CFG, encoding="UTF-8", xml_declaration=True)


def analyze_source(model, route_specs, source):
    tracker = sv.ByteTrack()
    cap = cv2.VideoCapture(source["video_path"])
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video input: {source['video_path']}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    detection_interval = max(1, int(DETECTION_INTERVAL))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else 0.0
    zones = [sv.PolygonZone(polygon=np.array(roi["points"])) for roi in source["rois"]]

    records = []
    spawned_ids = set()
    frame_index = 0
    frame_tracks = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % detection_interval != 0:
                frame_index += 1
                continue

            detections = infer_tracked_detections(model, tracker, frame)
            frame_entries = build_frame_track_entries(source, detections, zones, model)
            frame_tracks[frame_index] = frame_entries

            for entry in frame_entries:
                track_id = entry["track_id"]
                for roi_name in entry["roi_names"]:
                    unique_id = build_track_spawn_key(source["source_id"], roi_name, track_id)
                    if unique_id in spawned_ids:
                        continue

                    spawned_ids.add(unique_id)
                    spec = route_specs[roi_name]
                    depart = round(source["time_offset"] + (frame_index / fps), 2)

                    records.append(
                        {
                            "source_id": source["source_id"],
                            "source_label": source["label"],
                            "track_id": track_id,
                            "class_name": entry["class_name"],
                            "roi_name": roi_name,
                            "incoming_edge": spec["incoming_edge"],
                            "outgoing_edge": spec["outgoing_edge"],
                            "turn": spec["turn"],
                            "route_id": spec["route_id"],
                            "depart_lane": spec["depart_lane"],
                            "depart": depart,
                        }
                    )

            frame_index += 1
    finally:
        cap.release()

    source_summary = {
        "source_id": source["source_id"],
        "label": source["label"],
        "video_path": source["video_path"],
        "roi_groups": source["roi_groups"],
        "time_offset": source["time_offset"],
        "fps": fps,
        "detection_interval": detection_interval,
        "frame_count": frame_count,
        "duration": duration,
        "rois": source["rois"],
        "frame_tracks": frame_tracks,
    }
    return records, source_summary


def merge_tracked_records(records):
    merged_records = []
    for vehicle_index, record in enumerate(
        sorted(records, key=lambda item: (item["depart"], item["source_id"], item["track_id"], item["roi_name"]))
    ):
        merged_record = dict(record)
        merged_record["vehicle_id"] = f"veh_{vehicle_index}"
        merged_records.append(merged_record)
    return merged_records


def attach_vehicle_ids_to_source_summaries(source_summaries, tracked_records):
    vehicle_id_by_spawn_key = {
        build_track_spawn_key(record["source_id"], record["roi_name"], record["track_id"]): record["vehicle_id"]
        for record in tracked_records
    }

    for summary in source_summaries:
        for frame_entries in summary.get("frame_tracks", {}).values():
            for entry in frame_entries:
                vehicle_ids = []
                for spawn_key in entry.get("spawn_keys", []):
                    vehicle_id = vehicle_id_by_spawn_key.get(spawn_key)
                    if vehicle_id and vehicle_id not in vehicle_ids:
                        vehicle_ids.append(vehicle_id)
                entry["vehicle_ids"] = vehicle_ids


def draw_source_overlays(frame, source_summary):
    rendered = frame.copy()

    for roi in source_summary["rois"]:
        pts = np.array(roi["points"], np.int32)
        cv2.polylines(rendered, [pts], True, roi["color"], 2)
        cv2.putText(
            rendered,
            roi["name"],
            (roi["points"][0][0], roi["points"][0][1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            roi["color"],
            2,
        )

    header = f"{source_summary['label']} | offset={source_summary['time_offset']:.2f}s"
    cv2.rectangle(rendered, (10, 10), (420, 52), (30, 30, 30), -1)
    cv2.putText(rendered, header, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return rendered


def annotate_tracking_frame(frame, stream, frame_entries):
    annotated = draw_source_overlays(frame, stream["summary"])
    roi_stats = compute_roi_stats(frame_entries, stream["summary"]["rois"])
    draw_tracked_detections(annotated, frame_entries)
    draw_roi_stats_panel(annotated, roi_stats)
    return annotated


def build_placeholder_frame(source_summary, status_text):
    width, height = DISPLAY_CELL_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (width, height), (45, 45, 45), 2)
    cv2.putText(frame, source_summary["label"], (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, status_text, (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
    return frame


def resize_for_wall(frame):
    width, height = DISPLAY_CELL_SIZE
    return cv2.resize(frame, (width, height))


def compose_video_wall(frames):
    padded_frames = list(frames)
    if not padded_frames:
        return np.zeros((DISPLAY_CELL_SIZE[1], DISPLAY_CELL_SIZE[0], 3), dtype=np.uint8)

    while len(padded_frames) % 2 != 0:
        padded_frames.append(np.zeros_like(padded_frames[0]))

    rows = []
    for index in range(0, len(padded_frames), 2):
        rows.append(np.hstack(padded_frames[index:index + 2]))

    return rows[0] if len(rows) == 1 else np.vstack(rows)


def open_playback_streams(source_summaries):
    streams = []
    for summary in source_summaries:
        cap = cv2.VideoCapture(summary["video_path"])
        if not cap.isOpened():
            raise RuntimeError(f"Khong mo duoc video playback: {summary['video_path']}")

        streams.append(
            {
                "summary": summary,
                "cap": cap,
                "fps": summary["fps"] if summary["fps"] > 0 else 30.0,
                "next_frame_time": summary["time_offset"],
                "frame_index": 0,
                "frame_tracks": summary.get("frame_tracks", {}),
                "active_entries": [],
                "last_frame": None,
                "finished": False,
            }
        )

    return streams


def close_playback_streams(streams):
    for stream in streams:
        stream["cap"].release()


def update_stream_frame(stream, current_time):
    epsilon = 1e-6
    if current_time + epsilon < stream["summary"]["time_offset"]:
        return

    while not stream["finished"] and stream["next_frame_time"] <= current_time + epsilon:
        ret, frame = stream["cap"].read()
        if not ret:
            stream["finished"] = True
            break

        if stream["frame_index"] in stream["frame_tracks"]:
            stream["active_entries"] = stream["frame_tracks"][stream["frame_index"]]

        stream["last_frame"] = annotate_tracking_frame(frame, stream, stream["active_entries"])
        stream["next_frame_time"] += 1.0 / stream["fps"]
        stream["frame_index"] += 1


def render_stream_frame(stream, current_time):
    if current_time < stream["summary"]["time_offset"]:
        return build_placeholder_frame(stream["summary"], "Dang cho den offset")

    if stream["last_frame"] is None:
        status_text = "Dang cho frame dau tien"
        if stream["finished"]:
            status_text = "Video da ket thuc"
        return build_placeholder_frame(stream["summary"], status_text)

    frame = resize_for_wall(stream["last_frame"])
    status_text = f"t={current_time:.2f}s"
    if stream["finished"]:
        status_text += " | done"
    width, height = DISPLAY_CELL_SIZE
    cv2.rectangle(frame, (10, height - 42), (245, height - 8), (30, 30, 30), -1)
    cv2.putText(
        frame,
        status_text,
        (20, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return frame


def add_vehicle_to_sumo(record):
    traci.vehicle.add(
        vehID=record["vehicle_id"],
        routeID=record["route_id"],
        typeID="DEFAULT_VEHTYPE",
        departLane=record["depart_lane"],
    )
    traci.vehicle.setMaxSpeed(record["vehicle_id"], 13)


def run_sumo_playback(records, source_summaries, route_specs):
    ensure_route_file(route_specs)
    traci.start(["sumo-gui", "-c", SUMO_CFG])
    for spec in route_specs.values():
        if spec["route_id"] not in traci.route.getIDList():
            traci.route.add(spec["route_id"], [spec["incoming_edge"], spec["outgoing_edge"]])

    streams = open_playback_streams(source_summaries)
    current_time = 0.0
    record_index = 0
    last_depart = max((record["depart"] for record in records), default=0.0)
    last_video_time = max(
        (summary["time_offset"] + summary["duration"] for summary in source_summaries),
        default=0.0,
    )
    end_time = max(last_depart, last_video_time)
    max_fps = max((summary["fps"] for summary in source_summaries), default=30.0)
    playback_step = 1.0 / max_fps if max_fps > 0 else (1.0 / 30.0)

    try:
        while current_time <= end_time + playback_step or record_index < len(records):
            while record_index < len(records) and records[record_index]["depart"] <= current_time + 1e-6:
                add_vehicle_to_sumo(records[record_index])
                record_index += 1

            traci.simulationStep(current_time)

            frames = []
            for stream in streams:
                update_stream_frame(stream, current_time)
                frames.append(render_stream_frame(stream, current_time))

            if frames:
                cv2.imshow(VIDEO_WINDOW_NAME, compose_video_wall(frames))

            if cv2.waitKey(max(1, int(playback_step * 1000))) == 27:
                break

            current_time = round(current_time + playback_step, 6)
    finally:
        close_playback_streams(streams)
        traci.close()
        cv2.destroyAllWindows()


def main():
    route_specs = build_route_specs()
    source_contexts = build_source_contexts()
    model = YOLO(MODEL_PATH)

    all_records = []
    source_summaries = []

    for source in source_contexts:
        print(f"Dang xu ly {source['label']}: {source['video_path']}")
        source_records, source_summary = analyze_source(model, route_specs, source)
        print(f"  -> Phat hien {len(source_records)} xe tu {source['label']}")
        all_records.extend(source_records)
        source_summaries.append(source_summary)

    tracked_records = merge_tracked_records(all_records)
    attach_vehicle_ids_to_source_summaries(source_summaries, tracked_records)

    try:
        run_sumo_playback(tracked_records, source_summaries, route_specs)
    finally:
        export_tracked_vehicles(tracked_records, source_summaries)
        export_route_file(tracked_records, route_specs)
        max_depart = max((record["depart"] for record in tracked_records), default=0.0)
        update_sumo_config(max_depart)

    print(f"Da luu {len(tracked_records)} xe vao {TRACKED_JSON}")
    print(f"Da sinh route SUMO tai {ROUTES_XML}")
    print(f"Da cap nhat SUMO config tai {SUMO_CFG}")


if __name__ == "__main__":
    main()
