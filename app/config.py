from pathlib import Path

# Корневая директория проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Пути к директориям данных
DATA_DIR = BASE_DIR / "data"
EMPLOYEES_DIR = DATA_DIR / "employees"
INCIDENTS_DIR = DATA_DIR / "incidents"
MODELS_DIR = DATA_DIR / "models"
VIDEOS_DIR = DATA_DIR / "videos"

# Гарантируем существование папок
for directory in [EMPLOYEES_DIR, INCIDENTS_DIR, MODELS_DIR, VIDEOS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Путь к базе данных SQLite
DB_PATH = DATA_DIR / "perimeter_security.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Настройки моделей и порогов
YOLO_MODEL_PATH = MODELS_DIR / "yolo11s.pt"
CONFIDENCE_THRESHOLD = 0.5       # Порог уверенности детекции человека
FACE_SIMILARITY_THRESHOLD = 0.45 # Порог косинусного сходства для ArcFace (Свой/Чужой)

# Координаты центра карты по умолчанию (г. Таганрог, кампус ИТА ЮФУ)
DEFAULT_MAP_CENTER = [47.2041, 38.9442]
DEFAULT_MAP_ZOOM = 17