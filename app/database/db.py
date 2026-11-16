import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, joinedload
from app.config import DATABASE_URL
from app.database.models import Base, Camera, Employee, Incident

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Создает таблицы в БД и добавляет тестовые камеры периметра, если база пуста."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        if session.query(Camera).count() == 0:
            default_cameras = [
                Camera(
                    name="Камера №1 (Северный КПП)",
                    sector_description="Входная группа со стороны ул. Чехова",
                    latitude=47.2045,
                    longitude=38.9438,
                    video_source="data/videos/test_perimeter.mp4",
                    line_coords_json="[[150, 450], [1100, 450]]",
                ),
                Camera(
                    name="Камера №2 (Восточный периметр)",
                    sector_description="Ограждение внутреннего двора (Корпус Г)",
                    latitude=47.2039,
                    longitude=38.9451,
                    video_source="0",
                    line_coords_json="[[100, 350], [900, 350]]",
                ),
                Camera(
                    name="Камера №3 (Южные ворота)",
                    sector_description="Хозяйственный въезд и парковка",
                    latitude=47.2033,
                    longitude=38.9440,
                    video_source="data/videos/test_perimeter.mp4",
                    line_coords_json="[[200, 500], [1000, 500]]",
                ),
            ]
            session.add_all(default_cameras)
            session.commit()
            print("[БД] Таблицы созданы, добавлены 3 камеры охраняемого периметра.")
        else:
            print("[БД] База данных успешно подключена.")


def add_employee(full_name: str, position: str, photo_path: str, embedding: np.ndarray, has_access: bool = True):
    """Сохраняет сотрудника и его 512-мерный биометрический вектор в БД."""
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
    """Возвращает список сотрудников с распакованными numpy-векторами лиц."""
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
    """Фиксирует инцидент нарушения периметра в БД."""
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
        )
        session.add(incident)
        session.commit()
        session.refresh(incident)
        return incident


def get_cameras():
    """Возвращает список всех камер."""
    with SessionLocal() as session:
        return session.query(Camera).all()


def get_recent_incidents(limit: int = 50):
    """Возвращает последние инциденты вместе с данными о камере и сотруднике."""
    with SessionLocal() as session:
        return (
            session.query(Incident)
            .options(joinedload(Incident.camera), joinedload(Incident.employee))
            .order_by(Incident.timestamp.desc())
            .limit(limit)
            .all()
        )
def delete_employee(employee_id: int):
    """Удаляет сотрудника из БД."""
    with SessionLocal() as session:
        emp = session.query(Employee).filter(Employee.id == employee_id).first()
        if emp:
            session.delete(emp)
            session.commit()


def clear_all_incidents():
    """Очищает журнал инцидентов в БД (удобно перед демонстрацией на защите)."""
    with SessionLocal() as session:
        session.query(Incident).delete()
        session.commit()


def update_camera_settings(camera_id: int, video_source: str):
    """Обновляет адрес видеопотока для выбранной камеры."""
    with SessionLocal() as session:
        cam = session.query(Camera).filter(Camera.id == camera_id).first()
        if cam:
            cam.video_source = video_source
            session.commit()
def add_camera(name: str, sector_description: str, latitude: float, longitude: float, video_source: str, login: str = "", password: str = ""):
    """Добавляет новую камеру в систему."""
    with SessionLocal() as session:
        cam = Camera(
            name=name,
            sector_description=sector_description,
            latitude=latitude,
            longitude=longitude,
            video_source=video_source,
            stream_login=login,
            stream_password=password,
        )
        session.add(cam)
        session.commit()
        session.refresh(cam)
        return cam

def delete_camera(camera_id: int):
    """Удаляет камеру из системы."""
    with SessionLocal() as session:
        cam = session.query(Camera).filter(Camera.id == camera_id).first()
        if cam:
            session.delete(cam)
            session.commit()

if __name__ == "__main__":
    init_db()
    cams = get_cameras()
    print(f"Загружено камер из БД: {len(cams)}")
    for c in cams:
        print(f" - [{c.id}] {c.name} ({c.latitude}, {c.longitude})")