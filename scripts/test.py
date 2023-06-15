import barcode
import qrcode
from PIL import Image, ImageDraw
import cv2 as cv
from pyzbar.pyzbar import decode, ZBarSymbol
from kraken import binarization

"""
    Binarization needed

    Resize while preserving aspect ratio
https://www.youtube.com/watch?v=AV4x8bgPgk8&pp=ygULZHluYW1pYyByb2k%3D
https://stackoverflow.com/questions/61442775/preprocessing-images-for-qr-detection-in-python
   X  kraken
   X otsu
    QReader
"""

image_path = "sample/all_barcode/IMG_20220303_173846.jpg"
im = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
half = cv.resize(im, (0, 0), fx=0.1, fy=0.1)
blur = cv.GaussianBlur(half, (5, 5), 0)
ret, bw_im = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
decoded = decode(bw_im, symbols=[ZBarSymbol.QRCODE])

print(decoded)

img = Image.open(image_path)

draw = ImageDraw.Draw(img)

for barcode in decode(img):
    rect = barcode.rect
    draw.rectangle(
        ((rect.left, rect.top), (rect.left + rect.width, rect.top + rect.height)),
        outline="#0080ff",
        width=5,
    )
    # draw.polygon(barcode.polygon, outline="#e945ff", width=5)

cv.imwrite("bw.jpg", bw_im)
img.save("rectangle.png")
# qr = qrcode.QRCode(
#     version=1,
#     error_correction=qrcode.constants.ERROR_CORRECT_H,
#     box_size=10,
#     border=4,
# )
# qr.add_data("https://github.com/KSaiAkshit")
# qr.make(fit=True)

# img = qr.make_image(fill_color="black", back_color="white")
# img.save("qrcode.png")

# img = Image.open("qrcode.png")
# image = Image.open("sample/all_barcode/IMG_20220303_173611.jpg").convert("RGB")
# draw = ImageDraw.Draw(image)
# for barcode in decode(image):
# rect = barcode.rect
# draw.rectangle(
#     ((rect.left, rect.top), (rect.left + rect.width, rect.top + rect.height)),
#     outline="#0080ff",
#     width=5,
# )
# draw.polygon(barcode.polygon, outline="#e945ff", width=5)


# image.save("rectangle.png")
# img = cv.imread("sample/all_barcode/IMG_20220303_173611.jpg")
# bd = cv.barcode.BarcodeDetector()

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# retval, decoded_info, decoded_type, points = bd.detectAndDecode(img)

# print(bd.detectAndDecode(gray))
# img = cv.polylines(img, points.astype(int), True, (0, 255, 0), 3)

# for s, p in zip(decoded_info, points):
#     img = cv.putText(
#         img,
#         s,
#         p[1].astype(int),
#         cv.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0, 0, 255),
#         1,
#         cv.LINE_AA,
#     )

# cv.imwrite("opencv.jpg")
