import os
import time
import urllib.request
import warnings
from datetime import datetime
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Принудительно используем надежный TCP-транспорт для профессиональных RTSP-камер (Hikvision, Dahua, Axis)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

from app.config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, INCIDENTS_DIR
from app.cv_module.biometrics import BiometricsEngine
from app.database.db import log_incident

warnings.filterwarnings("ignore", category=FutureWarning)


class UniversalVideoStream:
    """
    Универсальный захват видеопотока промышленного и локального уровня:
    - RTSP / RTSPS (IP-камеры видеонаблюдения Hikvision, Dahua, Uniview, ONVIF) — TCP без накопления буфера.
    - HTTP / HTTPS (MJPEG-потоки мобильных и сетевых камер) — прямой разбор кадров без прокси.
    - Индекс USB/веб-камеры (0, 1) или архивный видеофайл (.mp4, .avi, .mkv).
    """

    def __init__(self, source):
        if isinstance(source, str) and source.strip().isdigit():
            source = int(source.strip())

        self.source = source
        self.is_http = isinstance(source, str) and source.startswith(("http://", "https://"))
        self.is_rtsp = isinstance(source, str) and source.startswith(("rtsp://", "rtsps://"))
        self.cap = None
        self.stream = None
        self.bytes_buffer = b""

        if self.is_http:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                self.stream = opener.open(self.source, timeout=5)
            except Exception as e:
                print(f"[Ошибка HTTP-потока] {e}")
                self.stream = None
        elif self.is_rtsp:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if self.cap.isOpened():
                # Отключаем буферизацию старых кадров для работы в реальном времени
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            self.cap = cv2.VideoCapture(self.source)

    def isOpened(self) -> bool:
        if self.is_http:
            return self.stream is not None
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        if not self.is_http:
            return self.cap.read()

        try:
            while True:
                chunk = self.stream.read(8192)
                if not chunk:
                    return False, None
                self.bytes_buffer += chunk

                end_idx = self.bytes_buffer.find(b"\xff\xd9")
                if end_idx != -1:
                    start_idx = self.bytes_buffer.rfind(b"\xff\xd8", 0, end_idx)
                    if start_idx != -1:
                        jpg_bytes = self.bytes_buffer[start_idx : end_idx + 2]
                        self.bytes_buffer = self.bytes_buffer[end_idx + 2 :]
                        frame = cv2.imdecode(
                            np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if frame is not None:
                            return True, frame
                    else:
                        self.bytes_buffer = self.bytes_buffer[end_idx + 2 :]
        except Exception as e:
            print(f"[Обрыв потока] {e}")
            return False, None

    def release(self):
        if self.cap is not None:
            self.cap.release()
        if self.stream is not None:
            self.stream.close()


class PerimeterDetector:
    """Модуль видеоаналитики: детекция людей, трекинг и контроль пересечения границы."""

    def __init__(self, camera_id: int = 1, line_orientation: str = "vertical", line_position: float = 0.5):
        self.camera_id = camera_id
        self.line_orientation = line_orientation
        self.line_position = line_position

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Детектор] Загрузка YOLO11s на устройство: {self.device.upper()}...")
        self.model = YOLO(str(YOLO_MODEL_PATH))

        self.biometrics = BiometricsEngine()

        self.track_history = {}
        self.cooldowns = {}
        self.cooldown_seconds = 5.0

        self.last_alert_info = None
        self.last_alert_time = 0.0

    def _has_crossed_line(self, prev_pt: tuple, curr_pt: tuple, frame_w: int, frame_h: int) -> bool:
        px, py = prev_pt
        cx, cy = curr_pt

        if self.line_orientation == "vertical":
            line_x = int(frame_w * self.line_position)
            return (px < line_x <= cx) or (px > line_x >= cx)
        else:
            line_y = int(frame_h * self.line_position)
            return (py < line_y <= cy) or (py > line_y >= cy)

    def process_frame(self, frame: np.ndarray):
        annotated = frame.copy()
        fh, fw = frame.shape[:2]
        incident_event = None

        if self.line_orientation == "vertical":
            lx = int(fw * self.line_position)
            line_pt1, line_pt2 = (lx, 0), (lx, fh)
        else:
            ly = int(fh * self.line_position)
            line_pt1, line_pt2 = (0, ly), (fw, ly)

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE_THRESHOLD,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = [int(v) for v in box]
                if (x2 - x1) * (y2 - y1) < 2500:
                    continue

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                crossed = False
                if track_id in self.track_history:
                    prev_pt = self.track_history[track_id]
                    if self._has_crossed_line(prev_pt, (cx, cy), fw, fh):
                        now = time.time()
                        if now - self.cooldowns.get(track_id, 0.0) > self.cooldown_seconds:
                            crossed = True
                            self.cooldowns[track_id] = now

                self.track_history[track_id] = (cx, cy)

                box_color = (0, 0, 255) if crossed else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(annotated, (cx, cy), 5, box_color, -1)
                cv2.putText(
                    annotated,
                    f"ID: {track_id}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                )

                if crossed:
                    bio_res = self.biometrics.analyze_person(frame, (x1, y1, x2, y2))
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                    full_filename = f"incident_{self.camera_id}_{ts_str}_full.jpg"
                    face_filename = f"incident_{self.camera_id}_{ts_str}_face.jpg"
                    full_path = INCIDENTS_DIR / full_filename
                    face_path = INCIDENTS_DIR / face_filename

                    cv2.line(annotated, line_pt1, line_pt2, (0, 0, 255), 4)
                    cv2.imwrite(str(full_path), annotated)

                    rel_face_path = None
                    if bio_res["face_crop"] is not None and bio_res["face_crop"].size > 0:
                        cv2.imwrite(str(face_path), bio_res["face_crop"])
                        rel_face_path = f"data/incidents/{face_filename}"

                    rel_full_path = f"data/incidents/{full_filename}"

                    db_inc = log_incident(
                        camera_id=self.camera_id,
                        person_status=bio_res["person_status"],
                        has_mask=bio_res["has_mask"],
                        alert_priority=bio_res["alert_priority"],
                        full_frame_path=rel_full_path,
                        face_crop_path=rel_face_path,
                        employee_id=bio_res["employee_id"],
                        similarity_score=bio_res["similarity"],
                    )

                    self.last_alert_info = {
                        "id": db_inc.id,
                        "camera_id": self.camera_id,
                        "timestamp": db_inc.timestamp.strftime("%H:%M:%S (%d.%m.%Y)"),
                        "status": bio_res["person_status"],
                        "name": bio_res["employee_name"],
                        "has_mask": bio_res["has_mask"],
                        "priority": bio_res["alert_priority"],
                        "similarity": bio_res["similarity"],
                        "full_frame_path": rel_full_path,
                        "face_crop_path": rel_face_path,
                    }
                    self.last_alert_time = time.time()
                    incident_event = self.last_alert_info
                    print(
                        f"[ТРЕВОГА] Камера #{self.camera_id} | Статус: {bio_res['person_status']} "
                        f"({bio_res['employee_name']}), Маска: {bio_res['has_mask']}, "
                        f"Приоритет: {bio_res['alert_priority']}"
                    )

        is_alarm_active = (time.time() - self.last_alert_time) < 3.0
        line_color = (0, 0, 255) if is_alarm_active else (0, 255, 255)
        cv2.line(annotated, line_pt1, line_pt2, line_color, 3)

        if is_alarm_active and self.last_alert_info:
            info = self.last_alert_info
            banner_color = (0, 0, 200) if info["priority"] == "CRITICAL" else (0, 140, 255)
            if info["status"] == "EMPLOYEE" and not info["has_mask"]:
                banner_color = (0, 160, 0)

            cv2.rectangle(annotated, (10, 10), (fw - 10, 60), banner_color, -1)
            mask_txt = "MASK: YES" if info["has_mask"] else "MASK: NO"
            alert_text = (
                f"ALERT [{info['priority']}] | {info['status']} | "
                f"{mask_txt} | Sim: {info['similarity']}"
            )
            cv2.putText(
                annotated,
                alert_text,
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return annotated, incident_event