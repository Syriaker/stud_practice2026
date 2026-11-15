import cv2
from app.config import EMPLOYEES_DIR
from app.database.db import init_db, add_employee
from app.cv_module.detector import PerimeterDetector


def main():
    init_db()

    video_source = "[http://192.168.0.160:8080/video](http://192.168.0.160:8080/video)"

    detector = PerimeterDetector(camera_id=1, line_orientation="vertical", line_position=0.5)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print(f"Не удалось открыть источник видео: {video_source}")
        print("Проверьте подключение веб-камеры или укажите путь к .mp4 файлу.")
        return

    print("\n=== УПРАВЛЕНИЕ В ОКНЕ КАМЕРЫ ===")
    print(" [R] — Сфотографировать ваше лицо и добавить в БД как 'Своего сотрудника'")
    print(" [V] — Переключить линию периметра (Вертикальная / Горизонтальная)")
    print(" [Q] — Выход из режима тестирования\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Зеркалим кадр для удобства при тесте с веб-камерой
        frame = cv2.flip(frame, 1)

        annotated_frame, _ = detector.process_frame(frame)
        cv2.imshow("Perimeter Security - Live Test", annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("v"):
            detector.line_orientation = (
                "horizontal" if detector.line_orientation == "vertical" else "vertical"
            )
            print(f"[Настройки] Ориентация границы изменена на: {detector.line_orientation}")
        elif key == ord("r"):
            # Регистрация текущего лица в кадре как сотрудника
            emb, bbox = detector.biometrics.extract_embedding(frame)
            if emb is not None:
                photo_path = EMPLOYEES_DIR / "chernetsov_m_e.jpg"
                cv2.imwrite(str(photo_path), frame)
                emp = add_employee(
                    full_name="Чернецов Максим Евгеньевич",
                    position="Администратор безопасности",
                    photo_path="data/employees/chernetsov_m_e.jpg",
                    embedding=emb,
                    has_access=True,
                )
                detector.biometrics.reload_employees()
                print(f"[УСПЕХ] Вы добавлены в БД сотрудников (ID: {emp.id})! Теперь пересеките линию.")
            else:
                print("[ОШИБКА] Лицо в кадре не найдено. Посмотрите в камеру без маски и нажмите R.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()