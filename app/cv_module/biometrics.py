import os
import cv2
import numpy as np
import torch

# Проброс CUDA DLL из PyTorch для корректной работы ONNX Runtime на Windows
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(torch_lib_path)

import onnxruntime as ort
from insightface.app import FaceAnalysis
from app.config import MODELS_DIR, FACE_SIMILARITY_THRESHOLD
from app.database.db import get_all_employees


class BiometricsEngine:
    """Модуль распознавания лиц («Свой — Чужой») и детекции масок/балаклав."""

    def __init__(self):
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        print(f"[Биометрия] Инициализация InsightFace (провайдеры: {providers})...")
        # Модели InsightFace автоматически скачаются в data/models/insightface при первом запуске
        self.face_app = FaceAnalysis(
            name="buffalo_l",
            root=str(MODELS_DIR / "insightface"),
            providers=providers,
        )
        self.face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        self.employees_cache = []
        self.reload_employees()

    def reload_employees(self):
        """Обновляет кэш эмбеддингов сотрудников из БД."""
        self.employees_cache = get_all_employees()
        print(f"[Биометрия] Загружено сотрудников в кэш: {len(self.employees_cache)}")

    def extract_embedding(self, image_bgr: np.ndarray):
        """Извлекает лицо и 512-мерный вектор из изображения (для регистрации сотрудника)."""
        faces = self.face_app.get(image_bgr)
        if not faces:
            return None, None
        # Берем самое крупное лицо в кадре
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return best_face.normed_embedding, best_face.bbox.astype(int)

    @staticmethod
    def _check_mask_on_face(face_crop_bgr: np.ndarray) -> bool:
        """
        Проверяет наличие медицинской маски/балаклавы на вырезанном лице.
        Сравнивает долю открытой кожи в верхней части лица (лоб/глаза) и нижней (нос/рот) в пространстве YCrCb.
        """
        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return True

        h, w = face_crop_bgr.shape[:2]
        if h < 30 or w < 30:
            return False

        # Перевод в цветовую модель YCrCb для выделения тона человеческой кожи
        ycrcb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2YCrCb)
        # Стандартный диапазон хроматических компонент кожи человека
        skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 175, 129]))

        # Верхняя зона лица (область глаз и переносицы: от 15% до 50% высоты)
        upper_zone = skin_mask[int(h * 0.15):int(h * 0.50), int(w * 0.15):int(w * 0.85)]
        # Нижняя зона лица (область носа, рта и подбородка: от 52% до 92% высоты)
        lower_zone = skin_mask[int(h * 0.52):int(h * 0.92), int(w * 0.15):int(w * 0.85)]

        upper_skin_ratio = np.count_nonzero(upper_zone) / max(upper_zone.size, 1)
        lower_skin_ratio = np.count_nonzero(lower_zone) / max(lower_zone.size, 1)

        # Если нижняя часть лица содержит очень мало пикселей кожи (< 22%)
        # или значительно перекрыта относительно верхней части — на лице маска/повязка
        if lower_skin_ratio < 0.22:
            return True
        if upper_skin_ratio > 0.25 and (lower_skin_ratio / upper_skin_ratio) < 0.45:
            return True

        return False

    def analyze_person(self, frame_bgr: np.ndarray, person_bbox: tuple) -> dict:
        """
        Анализирует силуэт человека, пересекшего периметр:
        1. Ищет лицо в области тела/головы.
        2. Проверяет наличие маски или полного сокрытия лица.
        3. Сверяет биометрический вектор с базой сотрудников («Свой — Чужой»).
        """
        fh, fw = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)

        person_crop = frame_bgr[y1:y2, x1:x2]
        ph, pw = person_crop.shape[:2]

        # Ищем лицо на кропе человека
        faces = self.face_app.get(person_crop)

        # Случай 1: Лицо не обнаружено (прикрыто балаклавой, капюшоном или человек отвернулся)
        if not faces:
            # Вырезаем верхние 35% силуэта (область головы) для карточки инцидента
            head_h = max(int(ph * 0.35), 40)
            head_crop = person_crop[0:min(head_h, ph), :]
            return {
                "person_status": "UNKNOWN",
                "employee_id": None,
                "employee_name": "Не определен (лицо скрыто)",
                "similarity": 0.0,
                "has_mask": True,
                "alert_priority": "CRITICAL",
                "face_crop": head_crop,
            }

        # Берем основное лицо
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]

        # Делаем небольшой отступ вокруг лица для красивого сохранения кропа
        pad_x = int((fx2 - fx1) * 0.15)
        pad_y = int((fy2 - fy1) * 0.15)
        cx1, cy1 = max(0, fx1 - pad_x), max(0, fy1 - pad_y)
        cx2, cy2 = min(pw, fx2 + pad_x), min(ph, fy2 + pad_y)
        face_crop = person_crop[cy1:cy2, cx1:cx2]

        # Проверяем, надета ли маска на область лица
        face_tight_crop = person_crop[max(0, fy1):min(ph, fy2), max(0, fx1):min(pw, fx2)]
        has_mask = self._check_mask_on_face(face_tight_crop)

        # Сверяем эмбеддинг с базой данных сотрудников (косинусное сходство)
        embedding = face.normed_embedding
        best_match = None
        best_score = -1.0

        for emp in self.employees_cache:
            score = float(np.dot(embedding, emp["embedding"]))
            if score > best_score:
                best_score = score
                best_match = emp

        # Определяем статус «Свой — Чужой» и приоритет тревоги
        if best_match and best_score >= FACE_SIMILARITY_THRESHOLD and best_match["has_access"]:
            person_status = "EMPLOYEE"
            employee_id = best_match["id"]
            employee_name = best_match["full_name"]
            # Даже если сотрудник в маске, приоритет средний, иначе низкий (проход разрешен)
            alert_priority = "HIGH" if has_mask else "LOW"
        else:
            person_status = "INTRUDER"
            employee_id = None
            employee_name = "Посторонний нарушитель"
            # Если нарушитель скрывает лицо маской — максимальный приоритет тревоги по ТЗ
            alert_priority = "CRITICAL" if has_mask else "HIGH"

        return {
            "person_status": person_status,
            "employee_id": employee_id,
            "employee_name": employee_name,
            "similarity": round(max(best_score, 0.0), 3),
            "has_mask": has_mask,
            "alert_priority": alert_priority,
            "face_crop": face_crop,
        }