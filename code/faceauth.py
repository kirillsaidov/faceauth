import time
import cv2
import numpy as np
import yolov5model

class FaceAuth():
    def __init__(self, face_model = None, person_model = None, face_model_img_size = (640, 640), person_model_img_size = (64, 64)):
        assert face_model, f'ERROR: provide the face model weights: {face_model}'
        assert person_model, f'ERROR: provide the face model weights: {person_model}'
        
        self.face_model_img_size = face_model_img_size
        self.person_model_img_size = person_model_img_size
        self.face_model = yolov5model.YOLOv5Model(face_model, force_reload = False)
        # self.person_model = None

    def launch(self):
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), 'ERROR: failed to open the Video Capture Device!'
        
        while True:
            # get a new frame
            retval, frame = cap.read()
            
            # resize image to model's input size
            frame = cv2.resize(frame, self.face_model_img_size)
            
            # detect and plot a face 
            t_start = time.time()
            face = self.face_model.detect(frame)
            if face:
                frame = self.face_model.plotBoxes(face, frame, thickness = 2)
            t_stop = time.time()

            # calculate and draw window fps
            fps = int(1/(t_stop - t_start))
            cv2.putText(frame, f'FPS: {fps}', (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # show img in a window
            cv2.imshow('FaceAuth', frame)

            # process window events
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()










