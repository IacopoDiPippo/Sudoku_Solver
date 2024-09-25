import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage.transform import resize
from sklearn.neural_network import MLPClassifier

model=joblib.load('Sudoku_Solver/models/svc/digits_cls.pkl')

image = imread('Sudoku_Solver/Sudoku_photos/Screenshot 2024-09-25 152516.png', as_gray=True)

resized_image = resize(image, (20,20), anti_aliasing=True)

binary_image = resized_image < threshold_otsu(resized_image)
binary_image = binary_image.reshape(1,-1)
number = model.predict(binary_image)
print(number)
