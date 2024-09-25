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

model=joblib.load('Sudoku_Solver\models\svc\svc.pkl')

image=imread('Sudoku_Solver/Sudoku_photos/Example1.png', as_gray=True)

fig, (x,y,z) = plt.subplots(1,3)

x.imshow(image, cmap='gray')

labeled_image = measure.label(image)

regionprop = regionprops(labeled_image)

sorted_prop = sorted(regionprop, key=lambda x: (x.centroid[0], x.centroid[1]))

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
    

        # Ridimensiona l'immagine
    resized_image = resize(number, (20,20), anti_aliasing=True)


    i=i+1
    if i==1:
        y.imshow(resized_image, cmap='gray')

    if i==2:
        z.imshow(resized_image, cmap='gray')

    binary_image = resized_image < threshold_otsu(resized_image)
    
    binary_image = binary_image.reshape(1,-1)
    number = model.predict(binary_image)
    sudoku.append(number)

plt.show()
print(sudoku)


