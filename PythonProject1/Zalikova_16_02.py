import cv2
import yt_dlp
import numpy as np
from ultralytics import YOLO

SOURCE_MODE = "youtube"
YOUTUBE_URL = "https://www.youtube.com/watch?v=Lxqcg1qt0XU"
VIDEO_FILE = "videos/test.mp4"

MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.4
TRACKER = "bytetrack.yaml"

zone = [(100,100), (300,100), (300,300), (100,300)]

selected_vertex = 0
STEP = 40

def get_youtube_stream(url):
    ydl_opts = {'format': 'best[ext=mp4]'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

def point_in_polygon(point, polygon):
    polygon_np = np.array(polygon, np.int32)
    return cv2.pointPolygonTest(polygon_np, point, False) >= 0

if SOURCE_MODE == "webcam":
    cap = cv2.VideoCapture(0)
elif SOURCE_MODE == "youtube":
    source = get_youtube_stream(YOUTUBE_URL)
    cap = cv2.VideoCapture(source)
else:
    cap = cv2.VideoCapture(VIDEO_FILE)

model = YOLO(MODEL_PATH)

inside_ids = set()
ever_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model.track(frame, conf=CONF_THRESH, tracker=TRACKER, persist=True, verbose=False)
    r = result[0]

    current_inside = set()

    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        track_id = boxes.id.cpu().numpy() if boxes.id is not None else None

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            class_id = int(cls[i])
            class_name = model.names[class_id]

            if class_name.lower() != "person":
                continue

            tid = int(track_id[i]) if track_id is not None else -1
            score = conf[i]

            h = y2 - y1
            cx = int((x1 + x2) / 2)
            cy = int(y2 - h * 0.1)

            inside = point_in_polygon((cx, cy), zone)

            if inside and tid != -1:
                current_inside.add(tid)
                ever_ids.add(tid)

            color = (0,0,255) if inside else (0,255,0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 4, color, -1)

            label = f'person ID {tid} {score:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    inside_ids = current_inside

    pts = np.array(zone, np.int32)
    cv2.polylines(frame, [pts], True, (0,255,0), 2)

    for i, (vx, vy) in enumerate(zone):
        c = (255,0,255) if i == selected_vertex else (0,255,0)
        cv2.circle(frame, (vx, vy), 8, c, -1)

    cv2.putText(frame, f'Trespassers detected: {len(inside_ids)}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, f'Trespassers ever: {len(ever_ids)}',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('l'):
        selected_vertex = (selected_vertex + 1) % len(zone)

    x, y = zone[selected_vertex]

    if key == ord('a'):
        x -= STEP
    if key == ord('d'):
        x += STEP
    if key == ord('w'):
        y -= STEP
    if key == ord('s'):
        y += STEP

    zone[selected_vertex] = (x, y)

cap.release()
cv2.destroyAllWindows()
