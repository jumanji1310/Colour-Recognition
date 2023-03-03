import cv2
import numpy as np
import os
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

#Loading training data               
with open('training.data','r') as file:
    train_pixels = []
    train_labels = []
    for line in file:
        data = line.strip().split(',')
        train_pixels.append((int(data[0]),int(data[1]),int(data[2])))
        train_labels.append(data[3])
        
# train the k-NN model
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(train_pixels, train_labels)


#Coordinates of cube center pieces

offset = 33
coords = [[220 + offset, 140 + offset], [220 + 3*offset, 140 + offset], [220 + 5*offset, 140 + offset],
          [220 + offset, 140 + 3*offset], [220 + 3*offset,
                                           140 + 3*offset], [220 + 5*offset, 140 + 3*offset],
          [220 + offset, 140 + 5*offset], [220 + 3*offset, 140 + 5*offset], [220 + 5*offset, 140 + 5*offset]]
patch_size = 40
patch_offset = patch_size//2

# Predict colour for each file
for filename in os.listdir("./test_img"):
    img = cv2.imread(os.path.join('./test_img', filename))
    patch_colour_list = ""
    
    # Predicting colour of each patch
    for coord in coords:
        patch = img[coord[1]-patch_offset:coord[1]+patch_offset,
                    coord[0]-patch_offset:coord[0]+patch_offset]
        patch_pixels = patch.reshape((-1, 3))
        predicted_labels = model.predict(patch_pixels)
        patch_color, _ = stats.mode(predicted_labels)
        patch_colour_list += f'{patch_color[0]} '

    # add text
    img = cv2.putText(img, patch_colour_list, (20,400),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),  2 )
    cv2.imshow(filename, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()