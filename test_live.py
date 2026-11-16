import cv2
from app.config import EMPLOYEES_DIR
from app.database.db import init_db, add_employee
from app.cv_module.detector import PerimeterDetector, UniversalVideoStream


def main():
    init_db()
    video_source = "http://192.168.0.160:8080/video"

    detector = PerimeterDetector(camera_id=1, line_orientation="vertical", line_position=0.5)
    cap = UniversalVideoStream(video_source)

    if not cap.isOpened():
        print(f"Не удалось открыть источник видео: {video_source}")
        return

    window_name = "Perimeter Security - Live Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 576)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if w > 1280:
            frame = cv2.resize(frame, (1280, int(h * (1280 / w))))

        annotated_frame, _ = detector.process_frame(frame)
        cv2.imshow(window_name, annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("v"):
            detector.line_orientation = (
                "horizontal" if detector.line_orientation == "vertical" else "vertical"
            )
        elif key == ord("r"):
            emb, _ = detector.biometrics.extract_embedding(frame)
            if emb is not None:
                photo_path = EMPLOYEES_DIR / "chernetsov_m_e.jpg"
                cv2.imwrite(str(photo_path), frame)
                add_employee(
                    full_name="Чернецов Максим Евгеньевич",
                    position="Администратор безопасности",
                    photo_path="data/employees/chernetsov_m_e.jpg",
                    embedding=emb,
                    has_access=True,
                )
                detector.biometrics.reload_employees()
                print("[УСПЕХ] Вы добавлены в БД сотрудников!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()