from flask import Flask, request, jsonify 
from flask_restful import Resource, Api
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils
import base64

bg = None
app = Flask(__name__)
api = Api(app)


def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # initialize the background
    bg = image.copy().astype("float")
    # compute weighted average, accumulate it and update the background

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2] + prediction[0][3] + prediction[0][4] + prediction[0][5]))



# Model defined
tf.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,6,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)

# Load Saved Model
model.load("TrainedModel/GestureRecogModel.tfl")


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}

    def post(self):
    	return {'hello': 'post'}

class BgImage(Resource):
	def post(self):
                # initialize weight for running average
                aWeight = 0.5
                # region of interest (ROI) coordinates
                top, right, bottom, left = 10, 350, 225, 590
    
                file = request.files['image'].read()
                npimg = np.frombuffer(file, np.uint8)
                img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
                ######### Do preprocessing here ################
                # img[img > 150] = 0
                ## any random stuff do here
                ################################################
                # resize the frame
                frame = imutils.resize(img, width = 700)

                # flip the frame so that it is not the mirror view
                frame = cv2.flip(frame, 1)

                # clone the frame
                clone = frame.copy()

                # get the height and width of the frame
                (height, width) = frame.shape[:2]

                # get the ROI
                roi = frame[top:bottom, right:left]

                # convert the roi to grayscale and blur it
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                run_avg(gray,aWeight)
                
		#json_data = request.get_json()
		#print(type(json_data))
		#segmented_text = segment(json_data['text'])
		#print(segmented_text)
		#prediction = sentiment(segmented_text)
		#return jsonify({'segmented_text':segmented_text, 'sentiment': prediction.tolist()})

class Predict(Resource):
	def post(self):
                # initialize weight for running average
                aWeight = 0.5
                # region of interest (ROI) coordinates
                top, right, bottom, left = 10, 350, 225, 590
    
                file = request.files['image'].read()
                npimg = np.frombuffer(file, np.uint8)
                img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
                
                ######### Do preprocessing here ################
                # img[img > 150] = 0
                ## any random stuff do here
                ################################################
                # resize the frame
                frame = imutils.resize(img, width = 700)

                # flip the frame so that it is not the mirror view
                frame = cv2.flip(frame, 1)

                # clone the frame
                clone = frame.copy()

                # get the height and width of the frame
                (height, width) = frame.shape[:2]

                # get the ROI
                roi = frame[top:bottom, right:left]

                # convert the roi to grayscale and blur it
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)

                hand = segment(gray)

                # check whether hand region is segmented
                if hand is not None:
                        # if yes, unpack the thresholded image and
                        # segmented region
                        (thresholded, segmented) = hand

                        # draw the segmented region and display the frame
                        cv2.imwrite('Temp.png', thresholded)
                        resizeImage('Temp.png')
                        predictedClass, confidence = getPredictedClass()
                        return jsonify({'predictedClass':str(predictedClass), 'confidence': str(confidence)})
                

api.add_resource(HelloWorld,'/')
api.add_resource(BgImage,'/bgImage')
api.add_resource(Predict, '/predict')
if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
    #Use option host='0.0.0.0' in run function to allow other computers to connect
