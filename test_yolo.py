from ultralytics import YOLO
import cv2

# Load model
model = YOLO("runs/detect/yolov11_custom/weights/best.pt")

# Inferensi folder test
results = model.predict(
    source="dataset/test/images",
    conf=0.25,
    device=0
)

# Tampilkan satu per satu
for r in results:
    img = r.plot()  # gambar + bbox
    cv2.imshow("YOLOv11 Inference", img)

    key = cv2.waitKey(0)  # tekan tombol untuk lanjut
    if key == 27:  # ESC untuk keluar
        break

cv2.destroyAllWindows()
