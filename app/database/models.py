from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, LargeBinary
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Camera(Base):
    """Модель рубежа видеонаблюдения с привязкой к ГИС-координатам."""
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    sector_description = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    protocol_type = Column(String(50), default="HTTP/MJPEG")  # RTSP, HTTP/MJPEG, USB/Local
    video_source = Column(String(255), default="0")
    line_coords_json = Column(String(255), default="[[200, 400], [1000, 400]]")
    is_active = Column(Boolean, default=True)

    incidents = relationship("Incident", back_populates="camera", cascade="all, delete-orphan")


class Employee(Base):
    """Модель сотрудника объекта для биометрической идентификации «Свой — Чужой»."""
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

    person_status = Column(String(50), default="INTRUDER")  # EMPLOYEE, INTRUDER, UNKNOWN
    similarity_score = Column(Float, default=0.0)
    has_mask = Column(Boolean, default=False)
    alert_priority = Column(String(50), default="HIGH")     # LOW, HIGH, CRITICAL
    is_acknowledged = Column(Boolean, default=False)        # Отработан ли инцидент оператором

    full_frame_path = Column(String(255), nullable=False)
    face_crop_path = Column(String(255), nullable=True)

    camera = relationship("Camera", back_populates="incidents")
    employee = relationship("Employee", back_populates="incidents")