import cv2
import numpy as np
import os

def train_Images():
    #Parse all training data and extract average BGR value of each
    folder_path = './detector/training_data'
    with open('./detector/training.data', 'w') as file:
        for foldername in os.listdir(folder_path):
            for filename in os.listdir(os.path.join(folder_path, foldername)):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    # load the image
                    image = cv2.imread(os.path.join(folder_path, foldername, filename))
                    
                    # calculate the average pixel value
                    average_pixel_value = np.round(np.mean(image, axis=(0,1)))
                    
                    # print the result
                    print(f"{filename}: {average_pixel_value}")

                    # write to file
                    file.write(f'{int(average_pixel_value[0])},{int(average_pixel_value[1])},{int(average_pixel_value[2])},{foldername}\n')

