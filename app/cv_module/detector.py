import time
from datetime import datetime
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, INCIDENTS_DIR
from app.cv_module.biometrics import BiometricsEngine
from app.database.db import log_incident


class PerimeterDetector:
    """Модуль видеоаналитики: детекция людей, трекинг и контроль пересечения границы."""

    def __init__(self, camera_id: int = 1, line_orientation: str = "vertical", line_position: float = 0.5):
        """
        :param camera_id: ID камеры в базе данных
        :param line_orientation: 'vertical' (удобно для защиты с вебкамеры/телефона) или 'horizontal'
        :param line_position: Положение линии от 0.1 до 0.9 (0.5 — ровно посередине кадра)
        """
        self.camera_id = camera_id
        self.line_orientation = line_orientation
        self.line_position = line_position

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Детектор] Загрузка YOLO11s на устройство: {self.device.upper()}...")
        self.model = YOLO(str(YOLO_MODEL_PATH))

        self.biometrics = BiometricsEngine()

        # История предыдущих положений объектов: {track_id: (cx, cy)}
        self.track_history = {}
        # Время последнего срабатывания для конкретного track_id (защита от дублей)
        self.cooldowns = {}
        self.cooldown_seconds = 6.0

        # Последнее зафиксированное событие для отображения плашки на экране
        self.last_alert_info = None
        self.last_alert_time = 0.0

    def _has_crossed_line(self, prev_pt: tuple, curr_pt: tuple, frame_w: int, frame_h: int) -> bool:
        """Проверяет, пересек ли центр объекта виртуальную границу между двумя кадрами."""
        px, py = prev_pt
        cx, cy = curr_pt

        if self.line_orientation == "vertical":
            line_x = int(frame_w * self.line_position)
            return (px < line_x <= cx) or (px > line_x >= cx)
        else:
            line_y = int(frame_h * self.line_position)
            return (py < line_y <= cy) or (py > line_y >= cy)

    def process_frame(self, frame: np.ndarray):
        """
        Обрабатывает один кадр видеопотока.
        Возвращает: (аннотированный_кадр, данные_инцидента_если_был_прорыв_иначе_None)
        """
        annotated = frame.copy()
        fh, fw = frame.shape[:2]
        incident_event = None

        # 1. Координаты виртуальной границы
        if self.line_orientation == "vertical":
            lx = int(fw * self.line_position)
            line_pt1, line_pt2 = (lx, 0), (lx, fh)
        else:
            ly = int(fh * self.line_position)
            line_pt1, line_pt2 = (0, ly), (fw, ly)

        # 2. Детекция и трекинг людей (class 0 = person) через ByteTrack
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
                # Отсеиваем слишком мелкие случайные объекты на заднем плане
                if (x2 - x1) * (y2 - y1) < 2500:
                    continue

                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Проверяем пересечение границы
                crossed = False
                if track_id in self.track_history:
                    prev_pt = self.track_history[track_id]
                    if self._has_crossed_line(prev_pt, (cx, cy), fw, fh):
                        now = time.time()
                        if now - self.cooldowns.get(track_id, 0.0) > self.cooldown_seconds:
                            crossed = True
                            self.cooldowns[track_id] = now

                self.track_history[track_id] = (cx, cy)

                # Отрисовка рамки человека и его центра
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

                # 3. Если зафиксировано пересечение границы — запускаем биометрию и запись в БД
                if crossed:
                    bio_res = self.biometrics.analyze_person(frame, (x1, y1, x2, y2))
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                    full_filename = f"incident_{self.camera_id}_{ts_str}_full.jpg"
                    face_filename = f"incident_{self.camera_id}_{ts_str}_face.jpg"
                    full_path = INCIDENTS_DIR / full_filename
                    face_path = INCIDENTS_DIR / face_filename

                    # Подсвечиваем линию красным на сохраняемом кадре
                    cv2.line(annotated, line_pt1, line_pt2, (0, 0, 255), 4)
                    cv2.imwrite(str(full_path), annotated)

                    rel_face_path = None
                    if bio_res["face_crop"] is not None and bio_res["face_crop"].size > 0:
                        cv2.imwrite(str(face_path), bio_res["face_crop"])
                        rel_face_path = f"data/incidents/{face_filename}"

                    rel_full_path = f"data/incidents/{full_filename}"

                    # Записываем событие в БД SQLite
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
                        "status": bio_res["person_status"],
                        "name": bio_res["employee_name"],
                        "has_mask": bio_res["has_mask"],
                        "priority": bio_res["alert_priority"],
                        "similarity": bio_res["similarity"],
                    }
                    self.last_alert_time = time.time()
                    incident_event = self.last_alert_info
                    print(
                        f"[ТРЕВОГА] Пересечение границы! Статус: {bio_res['person_status']} "
                        f"({bio_res['employee_name']}), Маска: {bio_res['has_mask']}, "
                        f"Приоритет: {bio_res['alert_priority']}"
                    )

        # 4. Отрисовка виртуальной линии периметра (красная при недавней тревоге, иначе желтая)
        is_alarm_active = (time.time() - self.last_alert_time) < 3.0
        line_color = (0, 0, 255) if is_alarm_active else (0, 255, 255)
        cv2.line(annotated, line_pt1, line_pt2, line_color, 3)

        # 5. Отрисовка информационной плашки последнего срабатывания
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