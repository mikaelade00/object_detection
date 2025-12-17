from ultralytics import YOLO
import torch

def main():
    # Cek GPU
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Load model pretrained YOLOv11
    model = YOLO("yolo11n.pt")  # bisa ganti: yolo11s.pt, yolo11m.pt

    # Training
    results = model.train(
        data="dataset/data.yaml",  # path ke data.yaml
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,                  # GPU 0
        workers=8,
        project="runs/detect",
        name="yolov11_custom",
        pretrained=True,
        verbose=True
    )

    print("Training selesai!")

if __name__ == "__main__":
    main()
