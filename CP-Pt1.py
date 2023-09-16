# Part 1: Data Pre-Processing

# Imports
import cv2
import mediapipe as mp
import os
import pandas as pd
import matplotlib.pyplot as plt
# To serialize and save data to be used in the future
import pickle


# Define Main Directory for Dataset
ASL_dir = "C:/Users/gcort/Desktop/Bootcamp/Bootcamp Course/Capstone Project/Datasets/ASL_Alphabet"


# # Goes through directory and renames files
# for item in os.listdir(ASL_dir):
#     if os.path.isfile(os.path.join(ASL_dir, item)):
#         continue
#     print(f'\nDir: {item}: ')
#     for sub in os.listdir(ASL_dir+'/'+item):
#         print(f'\nFolder: {sub}')
#         i=0
#         for file in os.listdir(ASL_dir+'/'+item+'/'+sub):
#             new_name = sub+'_'+str(i).zfill(3)
#             os.rename((ASL_dir+'/'+item+'/'+sub+'/'+file), (ASL_dir+'/'+item+'/'+sub+'/'+new_name+'.png'))
#             i = i+1
#             print(f'File Name: {file}' )


# Go through directories and create lists for:
# class id, labels, file_path, file_name, dataset
class_id = []
labels = []
file_path = []
file_name = []
dataset = []

for item in os.listdir(ASL_dir):
    i=0
    if os.path.isfile(os.path.join(ASL_dir, item)):
        continue
    for sub in os.listdir(ASL_dir+'/'+item):
        j=0
        for file in os.listdir(ASL_dir+'/'+item+'/'+sub):
            if sub=='zBlank':
                continue
            # Keeping samples down to 10 images per class for train AND test data
            if j < 150:
                # print(f'DIR: {item}: \nSub: {sub}: \nFile: {file}: {j}\n')
                dataset.append(item)
                class_id.append(i)
                file_path.append(item+'/'+sub+'/'+file)
                file_name.append(file)
                name, ext = os.path.splitext(file)
                labels.append(name[:-4])
                j+=1
        i+=1
    

# Take all cols and make df/export to .csv file
df = pd.DataFrame(list(zip(class_id, labels, file_path, file_name, dataset)), columns =['Class_id', 'Labels', 'File_path', 'File_name', 'Dataset'])
df

df.to_csv('Image_info_df.csv', index=False)


# Read df back in to not continuously run code above
df = pd.read_csv('Image_info_df.csv')


# Hand Landmarks/connections... initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
label = []

# Goes through df to access images for pre-processing
for row in df.index:
    path = ASL_dir+'/'+df.iloc[row]['File_path']
    # print(path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data_xy = []

    # Here we want to specify; If landmarks detected on image, we want to store
    # the data in an array that will correspond to a certain image label
    results = hands.process(img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_xy.append(x)
                data_xy.append(y)
            # print(len(data_xy))
        data.append(data_xy)
        label.append(path[-9])
    # # This next block plots the landmarks and hand connections to be shown on images
    #     mp_drawing.draw_landmarks(
    #     img,  # image to draw
    #     hand_landmarks,  # model output
    #     mp_hands.HAND_CONNECTIONS,  # hand connections
    #     mp_drawing_styles.get_default_hand_landmarks_style(),
    #     mp_drawing_styles.get_default_hand_connections_style())
    #     plt.imshow(img)
    #     plt.title(path)
    #     plt.show()
    
    
# Labels and data length may be less than df entries because some
# Images didnt regitser well enough to process hand landmarks
len(label)
len(data)

data_df = pd.DataFrame(data)
data_df
data_df.info()
data_df = data_df.iloc[:,:42]
data_df.info()

data_df = data_df.values.tolist()

f = open('dl.pickle', 'wb')
pickle.dump({'data': data_df, 'labels': label}, f)
f.close()