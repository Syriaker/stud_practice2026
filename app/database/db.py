import json
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, joinedload
from app.config import DATABASE_URL, DATA_DIR
from app.database.models import Base, Camera, Employee, Incident

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

PERIMETER_ZONE_FILE = DATA_DIR / "perimeter_zone.json"

DEFAULT_PERIMETER_POINTS = [
    [47.2048, 38.9432],
    [47.2048, 38.9456],
    [47.2030, 38.9456],
    [47.2030, 38.9432],
]


def _ensure_sqlite_columns():
    """Автоматически добавляет новые колонки в существующую БД SQLite без потери данных."""
    migrations = {
        "cameras": {
            "protocol_type": "VARCHAR(50) DEFAULT 'HTTP/MJPEG'",
        },
        "incidents": {
            "is_acknowledged": "BOOLEAN DEFAULT 0",
        },
    }
    with engine.begin() as conn:
        for table_name, columns in migrations.items():
            existing_cols = {
                row[1] for row in conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
            }
            for col_name, col_def in columns.items():
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}"))


def init_db():
    """Создает таблицы в БД, применяет миграции и добавляет стартовые камеры."""
    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_columns()

    if not PERIMETER_ZONE_FILE.exists():
        save_perimeter_polygon(DEFAULT_PERIMETER_POINTS)

    with SessionLocal() as session:
        if session.query(Camera).count() == 0:
            default_cameras = [
                Camera(
                    name="Рубеж №1 — Северный КПП",
                    sector_description="Центральная входная группа и турникетный узел",
                    latitude=47.2045,
                    longitude=38.9438,
                    protocol_type="HTTP/MJPEG",
                    video_source="http://192.168.0.160:8080/video",
                ),
                Camera(
                    name="Рубеж №2 — Восточный сектор",
                    sector_description="Внешнее ограждение технологической площадки",
                    latitude=47.2039,
                    longitude=38.9451,
                    protocol_type="RTSP",
                    video_source="rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101",
                ),
                Camera(
                    name="Рубеж №3 — Южные ворота",
                    sector_description="Транспортный КПП и зона разгрузки",
                    latitude=47.2033,
                    longitude=38.9440,
                    protocol_type="USB/Local",
                    video_source="0",
                ),
            ]
            session.add_all(default_cameras)
            session.commit()


def load_perimeter_polygon() -> list:
    """Загружает список координат вершин многоугольника периметра."""
    if PERIMETER_ZONE_FILE.exists():
        try:
            with open(PERIMETER_ZONE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return DEFAULT_PERIMETER_POINTS


def save_perimeter_polygon(points: list):
    """Сохраняет вершины многоугольника периметра (включая пустой список при создании нового)."""
    with open(PERIMETER_ZONE_FILE, "w", encoding="utf-8") as f:
        json.dump(points, f, ensure_ascii=False, indent=2)


def add_camera(name: str, sector_description: str, latitude: float, longitude: float, protocol_type: str, video_source: str):
    with SessionLocal() as session:
        cam = Camera(
            name=name,
            sector_description=sector_description,
            latitude=latitude,
            longitude=longitude,
            protocol_type=protocol_type,
            video_source=video_source,
        )
        session.add(cam)
        session.commit()
        session.refresh(cam)
        return cam


def delete_camera(camera_id: int):
    with SessionLocal() as session:
        cam = session.query(Camera).filter(Camera.id == camera_id).first()
        if cam:
            session.delete(cam)
            session.commit()


def update_camera_settings(camera_id: int, video_source: str, protocol_type: str = None):
    with SessionLocal() as session:
        cam = session.query(Camera).filter(Camera.id == camera_id).first()
        if cam:
            cam.video_source = video_source
            if protocol_type:
                cam.protocol_type = protocol_type
            session.commit()


def add_employee(full_name: str, position: str, photo_path: str, embedding: np.ndarray, has_access: bool = True):
    embedding_bytes = embedding.astype(np.float32).tobytes()
    with SessionLocal() as session:
        emp = Employee(
            full_name=full_name,
            position=position,
            photo_path=photo_path,
            face_embedding=embedding_bytes,
            has_access=has_access,
        )
        session.add(emp)
        session.commit()
        session.refresh(emp)
        return emp


def get_all_employees():
    with SessionLocal() as session:
        employees = session.query(Employee).all()
        result = []
        for emp in employees:
            vec = np.frombuffer(emp.face_embedding, dtype=np.float32)
            result.append({
                "id": emp.id,
                "full_name": emp.full_name,
                "position": emp.position,
                "photo_path": emp.photo_path,
                "embedding": vec,
                "has_access": emp.has_access,
            })
        return result


def delete_employee(employee_id: int):
    with SessionLocal() as session:
        emp = session.query(Employee).filter(Employee.id == employee_id).first()
        if emp:
            session.delete(emp)
            session.commit()


def log_incident(
    camera_id: int,
    person_status: str,
    has_mask: bool,
    alert_priority: str,
    full_frame_path: str,
    face_crop_path: str = None,
    employee_id: int = None,
    similarity_score: float = 0.0,
):
    with SessionLocal() as session:
        incident = Incident(
            camera_id=camera_id,
            employee_id=employee_id,
            person_status=person_status,
            similarity_score=similarity_score,
            has_mask=has_mask,
            alert_priority=alert_priority,
            full_frame_path=full_frame_path,
            face_crop_path=face_crop_path,
            is_acknowledged=False,
        )
        session.add(incident)
        session.commit()
        session.refresh(incident)
        return incident


def set_incident_acknowledged(incident_id: int, acknowledged: bool = True):
    """Устанавливает статус ревью инцидента (подтвержден / возвращен в работу)."""
    with SessionLocal() as session:
        inc = session.query(Incident).filter(Incident.id == incident_id).first()
        if inc:
            inc.is_acknowledged = acknowledged
            session.commit()


def acknowledge_all_incidents():
    with SessionLocal() as session:
        session.query(Incident).update({Incident.is_acknowledged: True})
        session.commit()


def delete_incident(incident_id: int):
    """Удаляет конкретную запись инцидента из БД."""
    with SessionLocal() as session:
        inc = session.query(Incident).filter(Incident.id == incident_id).first()
        if inc:
            session.delete(inc)
            session.commit()


def get_cameras():
    with SessionLocal() as session:
        return session.query(Camera).all()


def get_recent_incidents(limit: int = 200):
    with SessionLocal() as session:
        return (
            session.query(Incident)
            .options(joinedload(Incident.camera), joinedload(Incident.employee))
            .order_by(Incident.timestamp.desc())
            .limit(limit)
            .all()
        )


def clear_all_incidents():
    with SessionLocal() as session:
        session.query(Incident).delete()
        session.commit()