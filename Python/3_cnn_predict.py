# -*- coding: utf-8 -*-

# SetUp:
import os
os.chdir('C:\\Users\\Michael Nelz\\Documents\\Schulungen_Konferenzen\\Schulung AI')

# classify single picture:
import numpy as np
import pandas as pd
from keras.preprocessing import image as image_utils
from keras.models import load_model
###############################################################################
# open file from gui:
import easygui
#easygui.egdemo()
pic = easygui.fileopenbox()
###############################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import math
import cv2

figure(num=None, figsize=(10, 10), dpi=80, edgecolor='k')
#img=mpimg.imread(pic)
img=cv2.imread(pic)
#plt.imshow(img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r,g,b = cv2.split(img_rgb)

titles = ['Original', 'Rot', 'Grün', 'Blau']
images = [cv2.merge((r,g,b)), r, g, b]
cmaps = [None, 'Reds', 'Greens', 'Blues']

figure(num=None, figsize=(15, 15), dpi=80, edgecolor='k')
for i in range(len(images)):
    plt.subplot(math.sqrt(len(images)),math.sqrt(len(images)),i+1)
    plt.imshow(images[i], cmap = cmaps[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
plt.show()



loaded_model = load_model("checkpoints/model.20-0.785-0.69-0.67.hdf5")
loaded_model.summary()

###############################################s################################
from keras import backend as K
sess = K.get_session()

test_image = image_utils.load_img(pic, target_size = (150,150))
test_image = np.expand_dims(test_image, axis = 0)
test_image = K.reshape(test_image, [1,150,150,3])

result = loaded_model.predict_on_batch(test_image.eval(session = sess))

#training_set.class_indices
#classes = training_set.class_indices
classes = {'Bademantel': 0, 'Businessanzug': 1, 'Judoanzug': 2}

#Vorhersage:

result_list = result[0].tolist()
class_index = result_list.index(max(result_list))
prediction = list(classes)[class_index]
probabs = pd.DataFrame(columns = classes, index = range(1))
probabs.loc[0] = result[0].tolist()

print(prediction)
print(probabs)

# export Model to onnx
