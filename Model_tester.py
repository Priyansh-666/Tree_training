import cv2
import numpy as np
from PIL import Image
from keras import models
import tensorflow as tf
import os
cwd = os.getcwd()
model = models.load_model(cwd + r"\treechecker.model")
video = cv2.VideoCapture(0)
class_names = ['Not_Tree_Images', 'Tree_Images']
while True:
        _, frame = video.read()

        im = Image.fromarray(frame, 'RGB')

        im = im.resize((256,256))
        img_array = np.array(im)

        img_array = np.expand_dims(img_array, axis=0)


        prediction = int(model.predict(img_array)[0][0])
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        if prediction == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(f'its {class_names[np.argmax(score)]} with {100*np.max(score)} accuracy')

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()