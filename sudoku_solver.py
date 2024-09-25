import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread


help= np.ones((9,9,9))*-1
missing=81-27
output=np.ones((9,9))*-1
output[0][1]=2
output[0][2]=7
output[1][2]=3
output[2][0]=4
output[2][1]=6
output[1][3]=6
output[2][6]=1
output[2][7]=3
output[1][7]=9
output[3][1]=9
output[3][2]=1
output[5][0]=7
output[5][2]=8
output[4][4]=3
output[3][5]=6
output[4][6]=5
output[3][8]=2
output[7][1]=1
output[8][1]=5
output[6][4]=2
output[6][5]=9
output[6][6]=4
output[6][7]=1
output[6][8]=3
output[7][5]=4
output[7][7]=8
output[8][6]=7
print(output)


while missing>0:
    indices= np.where(output!=-1)
    for (i,j) in zip(indices[0],indices[1]):
        help[:,i,j]=0
        help[int(output[i,j])-1,i,:]=0
        help[int(output[i,j])-1,:,j]=0
        help[int(output[i,j])-1,(i//3)*3:(i//3)*3+3,(j//3)*3:(j//3)*3+3]=0


    for i in range(0,9):
        variable1=np.sum(help[:,:,i],axis=1)
        variable2=np.sum(help[:,i,:],axis=1)

        #print(f"Variable1 for index {i}:", variable1)
        #print(f"Variable2 for index {i}:", variable2)


        if len(np.argwhere(variable1==-1))>0:  
            x=np.argwhere(variable1==-1)[0,0]
            y=np.argwhere(help[x,:,i]==-1)[0,0]
            output[y,i]=x+1
            missing-=1
            break

        if len(np.argwhere(variable2==-1))>0:
            x=np.argwhere(variable2==-1)[0,0]
            y=np.argwhere(help[x,i,:]==-1)[0,0]
            output[i,y]=x+1
            missing-=1
            break



    
fig, axes=plt.subplots(9,9, figsize=(10, 10)) 
for i in range(0,9):
    for j in range(0,9):
        var=str(int(output[i,j]))
        img_path = f'Sudoku_Solver/Numbers/{var}/{var}_{var}.jpg'
         # Prova a leggere l'immagine
        try:
            img = imread(img_path)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')  # Nascondi gli assi
        except FileNotFoundError:
            print(f"File non trovato: {img_path}")
            axes[i, j].axis('off')  # Nascondi gli assi anche se non c'è immagine
plt.tight_layout()  # Migliora l'aspetto della disposizione delle sottotrame
plt.show()  # Mostra la figura

print(output)






