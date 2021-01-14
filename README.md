# Yolo_img

In this project the function is to detect an object in a given image.

The directory follows the pattern

```
## images
   ###Cropped_images(here the cropped images will be saved)
   -images needed to be examine
## yolo-coco
   -yolov3-320.cfg
   -yolov3.weights
   -coco.names
## main.py

```
# This code can do both detect and crop of an object that is being determined 

## crop funtion
----------------------------


``` crop = image[y-5: y+h+5 , x-5: x+w+5] ```


----------------------------

## This two lines make the last part of the code possible to save the cropped image and showing the image given as input with some border around the objects


----------------------------


``` cv2.imwrite(image_path,crop) ```

```cv2.imshow("Image", image) ```

----------------------------
 To make this code run you should open terminal or powershell open in the same folder where the ```main.py``` file located and enter the command as
 
 ---------------------------
 
 ``` python main.py --image images/person.jpg --yolo yolo-coco ```
 
 ---------------------------
 
 
