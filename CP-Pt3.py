# Implementation

# Imports
import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


model_dict1 = pickle.load(open('model1.pickle', 'rb'))
model1 = model_dict1['model1']

model_dict2 = pickle.load(open('model2.pickle', 'rb'))
model2 = model_dict2['model2']

model_dict3 = pickle.load(open('model3.pickle', 'rb'))
model3 = model_dict3['model3']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# This part captures feed from webcam for live recognition
#*************************************************************************************************************
capture = cv2.VideoCapture(0)
capture.isOpened()

while capture.isOpened():
    success, img = capture.read()     
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data_xy = []
    results = hands.process(img)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS, 
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())     
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_xy.append(x)
                data_xy.append(y)
                
        # # MODEL #1: **
        # prediction1 = model1.predict([np.asarray(data_xy)])
        # print(prediction1)
        
        # # MODEL #2: 
        # prediction2 = model2.predict([np.asarray(data_xy)])
        # print(prediction2)
        
        # MODEL #3: **
        prediction3 = model3.predict([np.asarray(data_xy)])
        print(prediction3)
    cv2.imshow('Window', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()
#*************************************************************************************************************




# Try fitting saved images into models:
#*************************************************************************************************************

# This part uses different data to test the models that was not included in the original training/testing sets
# I only included the first 150 images from each class for each original set

# testing_dir1 uses images from the original training dataset that was not part of the model training
# testing_dir2 uses random images from google (not processed/cropped/resized)
# testing_dir3 uses images from my own webcam 

testing_dir1 = 'C:/Users/gcort/Desktop/Bootcamp/Bootcamp Course/Capstone Project/Datasets/ASL_Alphabet/Train_Alphabet'
testing_dir2 = 'C:/Users/gcort/Desktop/Bootcamp/Bootcamp Course/Capstone Project/Testing Images/google'
testing_dir3 = 'C:/Users/gcort/Desktop/Bootcamp/Bootcamp Course/Capstone Project/Testing Images/me'


data = []
label = []

# ******************* TESTING_DIR1 ************************
for sub in os.listdir(testing_dir1):
    if sub=='zBlank':
        continue
    # print(sub)
    i = 0
    for img in os.listdir(testing_dir1+'/'+sub):
        if i == 200:
            path = testing_dir1+'/'+sub+'/'+img
            img = cv2.imread(path)
            # img = cv2.resize(img, (150,150))
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
                # This next block plots the landmarks and hand connections to be shown on images
                mp_drawing.draw_landmarks(
                img,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
                plt.imshow(img)
                plt.title(path)
                plt.show()
                break
        i+=1


label
for i in data:
    print(len(i))


# MODEL #1:
prediction1 = []
for i in range(len(data)):
    prediction1.append(model1.predict([np.asarray(data[i])]))
prediction1

results1 = accuracy_score(prediction1, label)
results1

# MODEL #2: 
prediction2 = []
for i in range(len(data)):
    prediction2.append(model2.predict([np.asarray(data[i])]))
prediction2

results2 = accuracy_score(prediction2, label)
results2

# MODEL #3: 
prediction3 = []
for i in range(len(data)):
    prediction3.append(model3.predict([np.asarray(data[i])]))
prediction3

results3 = accuracy_score(prediction3, label)
results3



# ******************* TESTING_DIR2 | TESTING_DIR3 ************************
data = []
label = []

# for img in os.listdir(testing_dir2):
    #************
for img in os.listdir(testing_dir3):
    # path = testing_dir2+'/'+img
    #************
    path = testing_dir3+'/'+img
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
        #******************************
        # label.append(path[-8])
        label.append(path[-7])
        # This next block plots the landmarks and hand connections to be shown on images
        mp_drawing.draw_landmarks(
        img,  # image to draw
        hand_landmarks,  # model output
        mp_hands.HAND_CONNECTIONS,  # hand connections
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style())
    plt.imshow(img)
    plt.title(path)
    plt.show()
            

label
for i in data:
    print(len(i))


# MODEL #1:
prediction1 = []
for i in range(len(data)):
    prediction1.append(model1.predict([np.asarray(data[i])]))
prediction1

results1 = accuracy_score(prediction1, label)
results1

# MODEL #2: 
prediction2 = []
for i in range(len(data)):
    prediction2.append(model2.predict([np.asarray(data[i])]))
prediction2

results2 = accuracy_score(prediction2, label)
results2

# MODEL #3: 
prediction3 = []
for i in range(len(data)):
    prediction3.append(model3.predict([np.asarray(data[i])]))
prediction3

results3 = accuracy_score(prediction3, label)
results3



 

# Models/ Image Deteection/Capture can be better with:
'''

- More data/ better segmented data
- More diverse data (Predicting classes from new image landmarks doesn't work as well,
but it does well enough when using the live capture detection)

- Adding a third dimension (z-axis) for the data-point collection for each 
hand detection node
- 


'''