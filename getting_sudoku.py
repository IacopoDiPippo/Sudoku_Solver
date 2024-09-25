from skimage.io import imread
from skimage import measure
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import matplotlib.patches as patches
from skimage.transform import resize
from number_model import guess_number
import numpy as np
import joblib
from skimage.filters import threshold_otsu

sudoku=[]

position=[]

model=joblib.load('Sudoku_Solver\models\svc\svm_mnist_model.pkl')

image=imread('Sudoku_Solver/Sudoku_photos/Example1.png', as_gray=True)

height, width = image.shape[:2] 

fig, (x,y,z) = plt.subplots(1,3)

x.imshow(image, cmap='gray')

labeled_image = measure.label(image)

regionprop = regionprops(labeled_image)

sorted_prop = sorted(regionprop, key=lambda x: (int(x.centroid[0]), x.centroid[1]))

threshold_value = 0.95
i=0
for region in sorted_prop:
    if region.area <100:
        continue
    
    minRow, minCol, maxRow, maxCol = region.bbox

    if np.mean(image[minRow:maxRow, minCol:maxCol])> threshold_value:
        continue
    Border = patches.Rectangle((minCol,minRow), maxCol-minCol,maxRow-minRow,edgecolor='red', linewidth=2, fill=False)
    x.add_patch(Border)

    number = image[minRow:maxRow, minCol:maxCol]
    
    height_sqaure = height//9.0
    position_x= region.centroid[0] //height_sqaure

    position_y = region.centroid[1] // height_sqaure
        # Ridimensiona l'immagine
    resized_image = resize(number, (28,28), anti_aliasing=True)

    binary_image = resized_image < threshold_otsu(resized_image)    
    i=i+1
    if i==2:
        y.imshow(binary_image, cmap='gray')

    if i==4:
        z.imshow(resized_image, cmap='gray')

    
    
    binary_image = binary_image.reshape(1,-1)
    number = model.predict(binary_image)
    

    sudoku.append(number[0])
    position.append((position_x,position_y))
plt.show()
print(sudoku)

