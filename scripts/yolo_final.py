from ultralytics import YOLO as yolo
from pyzbar.pyzbar import decode, ZBarSymbol
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

os.system("rm -rf runs/detect/predict*")

image_path = "sample/all_barcode/2.jpg"
og_im = cv.imread(image_path)
copy_im1 = np.copy(og_im)
copy_im2 = np.copy(og_im)
copy_im3 = np.copy(og_im)

og_height, og_width, _ = og_im.shape

cv.imshow("input", og_im)

model = yolo("yolov8n.pt")
results = model.predict(og_im, save=True, show_labels=False, show_conf=False, conf=0.2)

results_im = cv.imread("runs/detect/predict/image0.jpg")
cv.imwrite("results.jpg", results_im)

coords = []
for result in results:
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        r = box.xyxy[0].astype(int)
        coords.append(r)

        cv.rectangle(copy_im1, r[:2], r[2:], (0, 0, 0), 4)

cv.imwrite("all_objects.jpg", copy_im1)

roi = []
for i in range(len(coords)):
    roi_temp = copy_im1[coords[i][1] : coords[i][3], coords[i][0] : coords[i][2]]
    roi.append(roi_temp)

acceptable = ["CODE128", "CODE39", "EAN13", "EAN8", "UPCA", "UPCE", "QRCODE"]
range_im = list(range(0, len(coords)))
count = []
bars = []
for i in range(len(coords)):
    gray = cv.cvtColor(roi[i], cv.COLOR_BGR2GRAY)
    decoded_im = decode(gray)

    for barcode in decoded_im:
        _type = barcode.type
        # data = barcode.data
        if _type in acceptable:
            count.append(i)

    if len(decoded_im) > 0:
        for obj in decoded_im:
            data = obj.data.decode("utf-8")
            x, y, w, h = obj.rect

            if len(obj.polygon) >= 4:
                cv.rectangle(
                    copy_im2,
                    (coords[i][0] + x, coords[i][1] + y),
                    (coords[i][0] + x + w, coords[i][1] + y + h),
                    (255, 0, 0),
                    4,
                )
            else:
                cv.rectangle(
                    copy_im2,
                    (coords[i][0] + x, coords[i][1] + y),
                    (coords[i][0] + x + w, coords[i][1] + y + h),
                    (0, 255, 255),
                    4,
                )
            bars.append(data)

            print("Decoded data:", data)

cv.imwrite("codes.jpg", copy_im2)

count_fail = [item for item in range_im if item not in count]

print(count, range_im, count_fail)
red = (0, 0, 255)
for i in count_fail:
    x1, y1, x2, y2 = coords[i]
    cv.rectangle(copy_im3, (x1, y1), (x2, y2), red, 2)

cv.imwrite("failed.jpg",copy_im3)
