#!/usr/bin/env python
# coding: utf-8

# In[9]:


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

import numpy as np
from keras import backend as k
from keras.models import Sequential 
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# valid_generator = ImageDataGenerator().flow_from_directory(
#     directory=r"Data/Validate",
#     target_size=(640, 360),
#     color_mode="rgb",
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=True,
#     seed=42
# )

test_generator = ImageDataGenerator().flow_from_directory(
    directory=r"/Test",
    target_size=(640, 360),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(valid_generator, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=loaded_model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
labels = {'Cheating': 0, 'Not_Cheating': 1}
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[ ]:




