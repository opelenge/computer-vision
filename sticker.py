from tokenize import cookie_re
from cv2 import cvtColor
from matplotlib.style import available
import torch
import numpy as np
import cv2
from time import time

x_centre, y_centre = (), ()
matrix = []
transformed_frame = ()
pts = [[0,0], [0,0], [0,0], [0,0]]
labl = ['q', 'j', 'p', 'f']
lablu = ['TL-stickers', 'BL-stickers', 'TR-stickers', 'BR-stickers']
labeled = ()
class StickerDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom',
            path = model_name, force_reload =True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5', pretrained = True)    
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord
        

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                x_centre, y_centre = int((x1 + x2)/2), int((y1 + y2)/2)
                cv2.circle(frame, (x_centre, y_centre), 5, bgr, -1 )
                
                
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
              
        return frame

    def whiteboard_mode(self, frame, pts):
            
        pts1 = np.float32(pts)
        pts2 = np.float32([[0,0] , [0,480] , [640,0] , [640,480]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)      
        transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))
        cv2.imshow('whiteboard_mode', transformed_frame)  
  

    def coordinates(self, results, frame, pts):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]  
          
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                global x_centre, y_centre
                x_centre, y_centre = int((x1 + x2)/2), int((y1 + y2)/2)
                global labeled
                labeled = self.class_to_label(labels[i])
                

        global labl
        global lablu
        if labeled == 'TL-stickers':
            pts[0] = [10, y_centre]
            labl = [('TL-stickers' if ('q' in str)else str) for str in labl]
        if labeled == 'BL-stickers':
            pts[1] = [10 , y_centre] 
            labl = [('BL-stickers' if ('j' in str)else str) for str in labl]
        if labeled == 'TR-stickers':
            pts[2] = [630, y_centre]
            labl = [('TR-stickers' if ('p' in str)else str) for str in labl]    
        if labeled == 'BR-stickers':
            pts[3] = [630 , y_centre]
            labl = [('BR-stickers' if ('f' in str)else str) for str in labl]
        print(labl)
        print(pts)  

       
        
        if labl == lablu:
      
            self.whiteboard_mode(frame, pts)


        return pts     
        


    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
        
      
        while True:
          
            start_time = time()
            
            ret, frame = cap.read()
            assert ret
            global pts
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            self.coordinates(results, frame, pts)

            end_time = time()
            fps = 1/np.round(end_time - start_time, 2)
            
            #print(f"Frames Per Second : {fps}")
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            

            cv2.imshow('YOLOv5 Detection', frame)
            
            Key = cv2.waitKey(5)
            if (Key == ord('q')):
                break 
      
        cap.release()
        
        
# Create a new object and execute.
detector = StickerDetection(capture_index= 0, model_name = 'best1.pt')
detector()