import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt",
                               "data/MobileNet/mobilenet.caffemodel")

classes = []
with open("data/MobileNet/synset.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

files = [
    "images/MobileNet/cat.jpg",
    "images/MobileNet/tree.jpg",
    "images/MobileNet/bunny.jpg",
    "images/MobileNet/squirrel.jpg",
    "images/MobileNet/bread.jpg"
]

stats = {}

for file in files:
    img = cv2.imread(file)
    img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))
    if img is None:
        continue

    blob = cv2.dnn.blobFromImage(cv2.resize(img, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))
    net.setInput(blob)
    preds = net.forward()
    idx = preds[0].argmax()
    conf = float(preds[0][idx]) * 100
    label = classes[idx] if idx < len(classes) else "unknown"
    stats[label] = stats.get(label, 0) + 1

    print(f"{file} → {label} ({conf:.2f}%)")

    txt = f"{label}: {conf:.1f}%"
    cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 100), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\nТаблиця частот:")
for k, v in stats.items():
    print(f"{k}: {v}")
