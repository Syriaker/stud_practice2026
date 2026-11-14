import torch
import onnxruntime as ort
from ultralytics import YOLO

def main():
    print("=== 1. Проверка PyTorch и GPU ===")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA доступна: {cuda_available}")
    if cuda_available:
        print(f"Видеокарта: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Объем VRAM: {vram_gb:.2f} ГБ")

    print("\n=== 2. Проверка ONNX Runtime (для InsightFace) ===")
    providers = ort.get_available_providers()
    print(f"Доступные провайдеры: {providers}")
    if "CUDAExecutionProvider" in providers:
        print("Отлично! ONNX Runtime видит GPU.")
    else:
        print("Внимание: CUDAExecutionProvider не найден, работает CPU.")

    print("\n=== 3. Тестовая загрузка YOLO11s ===")
    model = YOLO("data/models/yolo11s.pt")
    print("Модель YOLO11s успешно загружена и готова к работе!")

if __name__ == "__main__":
    main()