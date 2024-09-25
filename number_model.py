from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from skimage.io import imread
import os
import joblib
from skimage.filters import threshold_otsu


def read_training_data(training_directory):

    image=[]
    target=[]

    for i in range(0,10):

        for j in range(0,10):

            directory = training_directory + str(i) +'/' + str(i) +'_' +str(j) +'.jpg'

            photo = imread(directory, as_gray=True)

            binary_image = photo < threshold_otsu(photo)

            flat_image = binary_image.reshape(-1)
            
            image.append(flat_image)

            target.append(i)

    return image,target


def cross_validation(model, X ,y):

    accuracy_result=cross_val_score(model, X, y, cv=5)
    
    print("Cross Validation Result for ", str(5), " -fold")

    print(accuracy_result * 100)


dir='Sudoku_Solver/Numbers/'

X, y= read_training_data(dir)

model = SVC(kernel='linear', probability=True)

cross_validation(model, X , y)

model.fit(X, y)

current_dir='C:/Users/iacop/Desktop/Programmazione/Github/Sudoku_Solver'

save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(model, save_directory+'/svc.pkl')

def guess_number(image):
    binary_image = image < threshold_otsu(image)
    binary_image = binary_image.reshape(1,-1)
    number = model.predict(binary_image)
    return number



