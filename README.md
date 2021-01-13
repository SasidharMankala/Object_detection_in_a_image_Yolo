# Yolo_img

In this project the function is to detect an object in a given image.

The directory follows the pattern

```
## images
   -images needed to be examine
## yolo-coco
   -yolov3-320.cfg
   -yolov3.weights
   -coco.names
## main.py

```
# This code can do both detect and crop of an object that is being determined 

## crop funtion
``` crop = image[y:y + h, x:x + w] ```

## Showing the output as desired, commenting the cropped code line will show an object detection and commenting the image code line will show the cropped image of detected object


----------------------------


``` cv2.imshow("CroppedImage", crop) ```

```cv2.imshow("Image", image) ```

------------
