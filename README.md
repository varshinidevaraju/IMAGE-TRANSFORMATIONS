# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
<br>

### Step2:
<br>

### Step3:
<br>

### Step4:
<br>

### Step5:
<br>

## Program:
```python
Developed By:
Register Number:

import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('chennai.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
plt.title("Original Image")  
plt.axis('off')

</br>
i)Image Translation

tx, ty = 100, 50 
M_translation = np.float32([[1, 0, tx], [0, 1, ty]])  
translated_image = cv2.warpAffine(image, M_translation, (image.shape[1], image.shape[0]))

plt.imshow(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB))  
plt.title("Translated Image")  
plt.axis('off')

ii) Image Scaling

fx, fy = 5.0, 2.0 
scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

plt.imshow(cv2.cvtColor(scaled_image, cv2.COLOR_BGR2RGB)) 
plt.title("Scaled Image") 
plt.axis('off')

iii)Image shearing

shear_matrix = np.float32([[1, 0.5, 0], [0.5, 1, 0]])
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

plt.imshow(cv2.cvtColor(sheared_image, cv2.COLOR_BGR2RGB)) 
plt.title("Sheared Image")  
plt.axis('off')

iv)Image Reflection

reflected_image = cv2.flip(image, 2)

plt.imshow(cv2.cvtColor(reflected_image, cv2.COLOR_BGR2RGB)) 
plt.title("Reflected Image")  
plt.axis('off')

v)Image Rotation

(height, width) = image.shape[:2]  
angle = 45  
center = (width // 2, height // 2)  
M_rotation = cv2.getRotationMatrix2D(center, angle, 1)  
rotated_image = cv2.warpAffine(image, M_rotation, (width, height))

plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)) 
plt.title("Rotated Image")  
plt.axis('off')


vi)Image Cropping

x, y, w, h = 100, 100, 200, 150  
cropped_image = image[y:y+h, x:x+w]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)) 
plt.title("Cropped Image")  
plt.axis('off')
```
## Output:
### i)Image Translation
![translate](https://github.com/user-attachments/assets/7b0589ae-e8e7-4e74-a834-bb4331774713)


### ii) Image Scaling
![scale](https://github.com/user-attachments/assets/a816a7f5-548e-4b08-8167-a1ebf10be5a3)


### iii)Image shearing
![shear](https://github.com/user-attachments/assets/7cd86398-0a61-4753-acb8-d0e479236cd2)



### iv)Image Reflection
![reflected](https://github.com/user-attachments/assets/8d6b525d-631e-4f5f-87d5-a984440cf221)



### v)Image Rotation
![rotate](https://github.com/user-attachments/assets/77f303e8-68ef-405b-a60e-464459234e9d)



### vi)Image Cropping
![croped](https://github.com/user-attachments/assets/cd58547e-4d51-483b-8b44-ed59c8c24b44)




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
