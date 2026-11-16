from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, LargeBinary, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Camera(Base):
    """Модель камеры наблюдения с поддержкой RTSP/IP и кастомных зон."""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    sector_description = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)

    # Промышленный поток (поддержка rtsp://, http://, или индексов 0, 1)
    video_source = Column(String(255), default="0")
    stream_login = Column(String(100), nullable=True)  # Для реальных IP-камер
    stream_password = Column(String(100), nullable=True)

    # Координаты виртуальной линии или полигона зоны охраны (JSON)
    line_coords_json = Column(String(255), default="[[200, 400], [1000, 400]]")
    custom_polygon_json = Column(Text, nullable=True)  # Произвольный полигон периметра на ГИС-карте

    is_active = Column(Boolean, default=True)

    incidents = relationship("Incident", back_populates="camera", cascade="all, delete-orphan")


class Employee(Base):
    """Модель сотрудника объекта для биометрической идентификации («Свой — Чужой»)."""
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(150), nullable=False)
    position = Column(String(100), nullable=False)
    photo_path = Column(String(255), nullable=False)
    face_embedding = Column(LargeBinary, nullable=False)
    has_access = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)

    incidents = relationship("Incident", back_populates="employee")


class Incident(Base):
    """Журнал инцидентов нарушения периметра."""
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)

    person_status = Column(String(50), default="INTRUDER")
    similarity_score = Column(Float, default=0.0)
    has_mask = Column(Boolean, default=False)
    alert_priority = Column(String(50), default="HIGH")

    full_frame_path = Column(String(255), nullable=False)
    face_crop_path = Column(String(255), nullable=True)

    camera = relationship("Camera", back_populates="incidents")
    employee = relationship("Incident", back_populates="employee")