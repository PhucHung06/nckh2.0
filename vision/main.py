import json
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import supervision as sv
import traci
from ultralytics import YOLO

from config import CONF, MODEL_PATH, ROI_TO_EDGE, SUMO_CFG, TURN_TO_LANE, VIDEO_INPUT
from config import INCOMING_TO_OUTGOING
from roi_config import ROIS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRACKED_JSON = os.path.join(DATA_DIR, "tracked_vehicles.json")
ROUTES_XML = os.path.join(DATA_DIR, "yolo_routes.rou.xml")


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


def export_tracked_vehicles(records):
    payload = {
        "source_video": VIDEO_INPUT,
        "sumo_cfg": SUMO_CFG,
        "vehicle_count": len(records),
        "vehicles": records,
    }
    with open(TRACKED_JSON, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def export_route_file(records, route_specs):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        '    <vType id="yolo_car" accel="2.6" decel="4.5" sigma="0.5" '
        'length="5.0" minGap="2.5" maxSpeed="13.9"/>',
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


def ensure_route_file_exists(route_specs):
    if not os.path.exists(ROUTES_XML):
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
    route_node.set("value", os.path.basename(ROUTES_XML))

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


def main():
    model = YOLO(MODEL_PATH)
    tracker = sv.ByteTrack()
    cap = cv2.VideoCapture(VIDEO_INPUT)

    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video input: {VIDEO_INPUT}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    route_specs = build_route_specs()
    ensure_route_file_exists(route_specs)

    traci.start(["sumo-gui", "-c", SUMO_CFG])
    for spec in route_specs.values():
        if spec["route_id"] not in traci.route.getIDList():
            traci.route.add(spec["route_id"], [spec["incoming_edge"], spec["outgoing_edge"]])

    zones = [sv.PolygonZone(polygon=np.array(roi["points"])) for roi in ROIS]

    spawned_ids = set()
    tracked_records = []
    vehicle_count = 0
    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            traci.simulationStep()

            results = model(frame, conf=CONF, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            for roi, zone in zip(ROIS, zones):
                in_zone = zone.trigger(detections=detections)
                det_zone = detections[in_zone]
                roi_name = roi["name"]
                spec = route_specs[roi_name]

                for i in range(len(det_zone)):
                    track_id = det_zone.tracker_id[i]
                    if track_id is None:
                        continue

                    track_id = int(track_id)
                    unique_id = f"{roi_name}_{track_id}"
                    if unique_id in spawned_ids:
                        continue

                    spawned_ids.add(unique_id)

                    veh_id = f"veh_{vehicle_count}"
                    depart = round(frame_index / fps, 2)

                    traci.vehicle.add(
                        vehID=veh_id,
                        routeID=spec["route_id"],
                        typeID="DEFAULT_VEHTYPE",
                        departLane=spec["depart_lane"],
                    )

                    tracked_records.append(
                        {
                            "vehicle_id": veh_id,
                            "track_id": track_id,
                            "roi_name": roi_name,
                            "incoming_edge": spec["incoming_edge"],
                            "outgoing_edge": spec["outgoing_edge"],
                            "turn": spec["turn"],
                            "route_id": spec["route_id"],
                            "depart_lane": spec["depart_lane"],
                            "depart": depart,
                        }
                    )

                    vehicle_count += 1

            for roi in ROIS:
                pts = np.array(roi["points"], np.int32)
                cv2.polylines(frame, [pts], True, roi["color"], 2)
                cv2.putText(
                    frame,
                    roi["name"],
                    (roi["points"][0][0], roi["points"][0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    roi["color"],
                    2,
                )

            cv2.imshow("Tracking", frame)
            frame_index += 1

            if cv2.waitKey(1) == 27:
                break
    finally:
        cap.release()
        traci.close()
        cv2.destroyAllWindows()

    export_tracked_vehicles(tracked_records)
    export_route_file(tracked_records, route_specs)
    max_depart = max((record["depart"] for record in tracked_records), default=0)
    update_sumo_config(max_depart)

    print(f"Da luu {len(tracked_records)} xe vao {TRACKED_JSON}")
    print(f"Da sinh route SUMO tai {ROUTES_XML}")
    print(f"Da cap nhat SUMO config tai {SUMO_CFG}")


if __name__ == "__main__":
    main()
