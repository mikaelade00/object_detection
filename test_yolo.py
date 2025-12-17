from ultralytics import YOLO
import cv2
import os
import numpy as np

# =========================
# Helper: YOLO label -> pixel bbox
# =========================
def yolo_to_xyxy(label, img_w, img_h):
    cls, x, y, w, h = label
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2, int(cls)

# =========================
# Load model
# =========================
model = YOLO("runs/detect/yolov11_custom/weights/best.pt")

image_dir = "dataset/test/images"
label_dir = "dataset/test/labels"

results = model.predict(
    source=image_dir,
    conf=0.25,
    device=0,
)

image_files = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
])

idx = 0
total = len(image_files)
while True:
    img_path = image_files[idx]
    img = cv2.imread(img_path)
    h, w, _ = img.shape

    img_pred = img.copy()
    img_gt = img.copy()

    # =========================
    # Inference ONE image
    # =========================
    result = model.predict(
        source=img_path,
        conf=0.25,
        device=0,
        verbose=False
    )[0]
# =========================
# Loop inference
# =========================
        # ========= DRAW PREDICTION (LEFT - RED) =========
    if result.boxes is not None:
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_pred, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img_pred, f"P:{int(cls)} {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1
            )

    # ========= DRAW GROUND TRUTH (RIGHT - GREEN) =========
    label_path = os.path.join(
        label_dir,
        os.path.basename(result.path).replace(".jpg", ".txt")
    )

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.split())
                x1, y1, x2, y2, cls = yolo_to_xyxy(
                    (cls, x, y, bw, bh), w, h
                )
                cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    img_gt, f"GT:{cls}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1
                )

    # ========= CONCAT LEFT & RIGHT =========
    combined = np.hstack((img_pred, img_gt))

    # Add titles
    cv2.putText(combined, "Prediction", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(combined, "Ground Truth", (w + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, f"{idx+1}/{total}",
                (w - 120, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)

    # ========= SHOW =========
    cv2.imshow("YOLOv11 | Left: Prediction | Right: Ground Truth", combined)

    key = cv2.waitKey(0) & 0xFF

    if key == 27:  # ESC
        break
    elif key == 100:  # →
        idx = min(idx + 1, total - 1)
    elif key == 97:  # ←
        idx = max(idx - 1, 0)

cv2.destroyAllWindows()
