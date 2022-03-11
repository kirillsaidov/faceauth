import time
import cv2
import torch
import torchvision
import numpy as np
import yolov5model
import cnnclassifier as cnn

import smtplib
from email.message import EmailMessage

device = torch.device(yolov5model.getDevice())

class FaceAuth():
    def __init__(self, face_model = None, person_model = None, face_model_img_size = (640, 640), person_model_img_size = (64, 64), person_img_mean = [0.5, 0.5, 0.5], person_img_std = [0.5, 0.5, 0.5], person_classes = None, conf = 0.5):
        assert face_model, f'ERROR: provide the face model weights: {face_model}'
        assert person_model, f'ERROR: provide the face model weights: {person_model}'
        assert person_classes, f'ERROR: provide person classes path: {person_classes}'

        # save variables
        self.conf = conf
        self.face_model_img_size = face_model_img_size
        self.person_model_img_size = person_model_img_size

        # load face model
        self.face_model = yolov5model.YOLOv5Model(face_model, force_reload = False)

        # create img transformer
        self.transformer = torchvision.transforms.Compose([
            torchvision.transforms.Resize(person_model_img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean = person_img_mean,
                std = person_img_std
            )
        ])

        # read classes from a file
        self.classes = []
        with open(person_classes, "r") as f:
            lines = [line.rstrip() for line in f]
            self.classes = lines[0].split(";")

        # load person model
        self.person_model = cnn.loadModel(person_model).to(device)

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
            modelOutput = self.face_model.detect(frame)
            crop_data = self.face_model.getBoxData(modelOutput, frame)
            if crop_data:
                crop_data = crop_data[0]

                # crop face from frame
                x1, y1, x2, y2, label = crop_data
                crop_face = frame[y1:y2, x1:x2]

                # predict person's name
                class_id, alpha = cnn.predictFromFrame(self.person_model, crop_face, self.transformer, self.classes)

                if alpha < self.conf:
                    class_id = 'UNKNOWN'

                print(alpha)

                # draw boxes around faces
                frame = self.face_model.plotBox(crop_data, frame, thickness = 2, class_id = class_id)
            t_stop = time.time()

            # calculate and draw window fps
            fps = int(1/(t_stop - t_start))
            cv2.putText(frame, f'FPS: {fps}', (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # show img in a window
            cv2.imshow('FaceAuth', frame)

            # process window events
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()

    def notify(self,
            email_from = None,
            email_to = None,
            email_password = None,
            email_subject = 'FaceAuth notification test.',
            email_body = 'This is a test. Some dummy text. Ignore it.'
        ):
        assert email_from, 'ERROR: sender email unknown!'
        assert email_to, 'ERROR: recipient unknown!'
        assert email_password, f'ERROR: password not provided for <{email_from}>!'

        # prepare email message
        msg = EmailMessage()
        msg['subject'] = email_subject
        msg['from'] = email_from
        msg['to'] = email_to
        msg.set_content(email_body)

        # send message
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_from, email_password)
            smtp.send_message(msg)

    def authorize(self):
        pass
