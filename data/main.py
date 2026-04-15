import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import traci

from config import *
from roi_config import ROIS

# ===== INIT =====
model   = YOLO(MODEL_PATH)
tracker = sv.ByteTrack()

cap = cv2.VideoCapture(VIDEO_INPUT)

# ===== SUMO =====
traci.start(["sumo-gui", "-c", SUMO_CFG])

# tạo route cho từng ROI theo hướng rẽ và làn tương ứng
for roi_name, incoming_edge in ROI_TO_EDGE.items():
    turn = roi_name.split("_")[-1][-1]
    outgoing_edge = INCOMING_TO_OUTGOING[incoming_edge][turn]
    route_id = f"route_{roi_name}"

    if route_id not in traci.route.getIDList():
        traci.route.add(route_id, [incoming_edge, outgoing_edge])

# ===== ROI =====
zones = []
for roi in ROIS:
    polygon = np.array(roi['points'])
    zones.append(sv.PolygonZone(polygon=polygon))

# ===== CONTROL =====
spawned_ids = set()
vehicle_count = 0

# ===== LOOP =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    traci.simulationStep()

    # YOLO detect
    results = model(frame, conf=CONF, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # ===== PROCESS ROI =====
    for roi, zone in zip(ROIS, zones):
        in_zone = zone.trigger(detections=detections)
        det_zone = detections[in_zone]

        roi_name = roi['name']
        route_id = f"route_{roi_name}"
        turn = roi_name.split("_")[-1][-1]
        depart_lane = TURN_TO_LANE[turn]

        for i in range(len(det_zone)):
            track_id = int(det_zone.tracker_id[i])
            unique_id = f"{roi_name}_{track_id}"

            if unique_id in spawned_ids:
                continue

            spawned_ids.add(unique_id)

            veh_id = f"veh_{vehicle_count}"

            traci.vehicle.add(
                vehID=veh_id,
                routeID=route_id,
                typeID="DEFAULT_VEHTYPE",
                departLane=depart_lane
            )

            vehicle_count += 1

    # ===== DRAW ROI =====
    for roi in ROIS:
        pts = np.array(roi['points'], np.int32)
        cv2.polylines(frame, [pts], True, roi['color'], 2)
        cv2.putText(frame, roi['name'],
                    (roi['points'][0][0], roi['points'][0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, roi['color'], 2)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

# ===== CLEAN =====
cap.release()
traci.close()
cv2.destroyAllWindows()
