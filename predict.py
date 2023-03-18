import cv2
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#Loading training data               
with open('./detector/training.data','r') as file:
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

#convert colour string to Singmaster notation
def stringToSingmaster(string):
    turn_string = ''
    string_list = string.strip().split()
    for colour in string_list:
        if colour == 'white':
            turn_string += 'F'
        elif colour == 'orange':
            turn_string += 'U'
        elif colour == 'blue':
            turn_string += 'R'
        elif colour == 'green':
            turn_string += 'L'
        elif colour == 'red':
            turn_string += 'D'
        elif colour == 'yellow':
            turn_string += 'B'
    print(turn_string)
    return turn_string

# Predict colour for each file
def predict_Colour(folder_name):
    final_turn_string = ''

    for filename in os.listdir(folder_name):
        img = cv2.imread(os.path.join(folder_name, filename))
        patch_colour_list = ""
        
        # Predicting colour of each patch
        for coord in coords:
            patch = img[coord[1]-patch_offset:coord[1]+patch_offset,
                        coord[0]-patch_offset:coord[0]+patch_offset]
            patch_pixels = patch.reshape((-1, 3))
            predicted_labels = model.predict(patch_pixels)
            patch_color = pd.DataFrame.mode(pd.DataFrame(predicted_labels))
            patch_colour_list += f'{patch_color[0][0]} '

        #convert to turn string and add to final turn string
        final_turn_string += stringToSingmaster(patch_colour_list)

        # add text
        img = cv2.putText(img, patch_colour_list, (20,400),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,0),  2 )
        cv2.imshow(filename, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_turn_string
# predict_Colour('test_img')