import os
import cv2
import numpy as np
import tensorflow as tf

from keras.models import load_model
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        # load classifier

        cwd = os.path.dirname(os.path.realpath(__file__))

        # load the keras Lenet model from the tl_classifier.h5 file
        self.class_model = load_model(cwd+'/tl_classifier.h5')
        self.class_graph = tf.get_default_graph()

        # detection graph for detection rectangular shaped traffic lights in images
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            grahDef = tf.GraphDef()
            with open(cwd+"/frozen_inference_graph.pb", 'rb') as file:
                grahDef.ParseFromString(file.read())
                tf.import_graph_def(grahDef, name="" )

            self.session = tf.Session(graph=self.detection_graph)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes =  self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections    = self.detection_graph.get_tensor_by_name('num_detections:0')

        self.tl_classes = [ TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN ]

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Implement light color prediction

        light_classification = TrafficLight.UNKNOWN
        box = None

        with self.detection_graph.as_default():
            # Convert the recieved image from BGR to RGB.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            tf_image_input = np.expand_dims(image,axis=0)
            # Detect the box for trafic light rectangle
            (detection_boxes, detection_scores, detection_classes, num_detections) = self.session.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: tf_image_input})

            detection_boxes = np.squeeze(detection_boxes)
            detection_classes = np.squeeze(detection_classes)
            detection_scores = np.squeeze(detection_scores)

            # Find first detection of signal. It's labeled with number 10
            # Check if it fits into rectangular box so we can be sure it's a traffic light
            for i, detection_class in enumerate(detection_classes.tolist()):
                if detection_class == 10 and detection_scores[i] > 0.3:
                    dim = image.shape[0:2]
                    height, width = dim[0], dim[1]
                    box = np.array([int(detection_boxes[i][0]*height), int(detection_boxes[i][1]*width),
                           int(detection_boxes[i][2]*height), int(detection_boxes[i][3]*width)])
                    box_h = box[2] - box[0]
                    box_w = box[3] - box[1]
                    # too small to be a traffic light
                    if box_h < 20 or box_w < 20:
                        box = None

        if box is None:
            return light_classification

        # cut the image into ROI (region of size) and resize the image to 32,32
        # as the keras model has been trained on 32, 32 input shape
        class_image = cv2.resize(image[box[0]:box[2], box[1]:box[3]], (32,32))

        img_resize = np.expand_dims(class_image, axis=0).astype('float32')
        with self.class_graph.as_default():
            predict = self.class_model.predict(img_resize)
            light_classification  = self.tl_classes[np.argmax(predict)]

        return light_classification
