# QRcode/Barcode detection

## Needed outputs

-The output should contain the value of every barcode, and how many times every barcode appears in the image ( there could be repetitions of the barcode). This output could be printed.

- Display input image (5 points).
- Create a Bounding box of blue color on the input image, around all the barcodes which are correctly read. (15 points)
- Create a Bounding box of yellow color on the input image around the barcodes which are partially visible and couldn’t be completely read (20 points)
- Create a Bounding box of red color on the input image around items that did not have any barcodes. (20 points)
- Create bounding boxes of black color around each ITEM in the image ( not the barcode) - 20 points

## Dependencies

- `ultralytics` (Yolov8)
- `opencv-python`
- `pyzbar`

## methodology

- The COCO128 dataset is used to train the yolov8n model from ultralytics.
- After an image is input, it is passed to the nn model for object recognition. The `model.predict` returns a list with `boxes`, `masks` and `probs`, out of which `boxes` is the one that is used extensively. `boxes.xyxy` returns two points for the bounding box, the Top Left and Bottom Right points.
- ROI (Region of interest) is extracted from the above points, aiding in barcode/qrcode detection. Code detection, provided by `pyzbar`, often struggles with large file sizes, so running the detection in smaller ROI's works out for the better.
- After running `pyzbar.decode()` on the ROI's, the result is iterated to extract the detected _Code type_ (`EAN13`, `QRCODE`), decoded `data`, and position of the the qrcode/barcode in the form of `rect` or `poly`. Draw **Blue** or **Yellow** bounding boxes around the code.
- Check if the detected `type` is of the acceptable type, and keep track of the ones that fail the test. After running through all the `decoded_data`, Draw **RED** box arounf the ROI's that failed the test.

### Shortcomings

- The dataset being quite small means that not all items were recognized.
- I couldn't find a way to identify partial codes.

### Possible fixes

- Bigger dataset
- Run multiple yolo models, one for items and another for QRcodes/barcodes
- Better QRcode localization, possibly _Viola-Jones method_ (need to read up on it).
