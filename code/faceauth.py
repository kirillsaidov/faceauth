import time
import math
import datetime
import cv2
import torch
import torchvision
import numpy as np
import yolov5model
import cnnclassifier as cnn
from threading import Thread

import smtplib
from email.message import EmailMessage

device = torch.device(yolov5model.getDevice())

class FaceAuth():
    def __init__(self, face_model = None, person_model = None, face_model_img_size = (640, 640), person_model_img_size = (64, 64), person_img_mean = [0.5, 0.5, 0.5], person_img_std = [0.5, 0.5, 0.5], conf = 0.5, detect_dist_threshold = 0.3, timeDiff = 10, email_from = None, email_to = None, email_password = None):
        assert face_model, f'ERROR: provide the face model weights: {face_model}'
        assert person_model, f'ERROR: provide the face model weights: {person_model}'

        # save variables
        self.conf = conf
        self.face_model_img_size = face_model_img_size
        self.person_model_img_size = person_model_img_size
        self.detect_dist_threshold = detect_dist_threshold
        self.timeDiff = timeDiff
        self.email_from = email_from
        self.email_to = email_to
        self.email_password = email_password

        # load face and person models
        self.face_model = yolov5model.YOLOv5Model(face_model, force_reload = False)
        self.person_model = yolov5model.YOLOv5Model(person_model, force_reload = False)

        # registered
        self.registedPeople = dict()

    def launch(self):
        cap = cv2.VideoCapture(0)
        assert cap.isOpened(), 'ERROR: failed to open the Video Capture Device!'

        while True:
            # threads list
            authList = list()

            # get a new frame
            retval, frame = cap.read()

            # resize image to model's input size
            frame = cv2.resize(frame, self.face_model_img_size)

            # detect and plot a face
            t_start = time.time()
            modelOutputFaces = self.face_model.detect(frame)
            crop_face = self.face_model.getBoxData(modelOutputFaces, frame)

            # check if face is close enough to the camera, if true, recognize the face
            if crop_face:
                for cf in crop_face:
                    x1, y1, x2, y2 = cf[0:4]
                    crop_img_face = frame[y1:y2, x1:x2]
                    if (x2 - x1)/self.face_model_img_size[0] > self.detect_dist_threshold or (y2 - y1)/self.face_model_img_size[1] > self.detect_dist_threshold:
                        # face recognition and draw bounding box
                        modelOutputPerson = self.person_model.detect(frame)

                        # if face was recognized
                        crop_person = self.person_model.getBoxData(modelOutputPerson, frame)
                        if crop_person:
                            crop_person = crop_person[0]
                            class_id = crop_person[4] if crop_person[5] >= self.conf else 'Unknown'

                            # register person
                            updateTime = datetime.datetime.now().replace(microsecond = 0)
                            if class_id in self.registedPeople:
                                if getTimeDiff(self.registedPeople[class_id]) > self.timeDiff:
                                    # append person
                                    authList.append((class_id, crop_img_face))

                                    # update time
                                    self.registedPeople.update({class_id: updateTime})

                                    # draw athentication text
                                    frame = cv2.putText(frame, "Authentication...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 200), 3)
                            else:
                                # update time
                                self.registedPeople.update({class_id: updateTime})

                                # append person
                                authList.append((class_id, crop_img_face))

                                # draw athentication text
                                frame = cv2.putText(frame, "Authentication...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 200), 3)

                            # print time diff
                            # print(getTimeDiff(self.registedPeople[class_id]))

                            # draw box
                            frame = self.person_model.plotBox(crop_person[0:4], frame, thickness = 2, class_id = class_id)

            """
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
            """
            t_stop = time.time()

            # calculate and draw window fps
            fps = int(1/(t_stop - t_start))
            cv2.putText(frame, f'FPS: {fps}', (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # show img in a window
            cv2.imshow('FaceAuth', frame)

            # process window events
            if cv2.waitKey(5) & 0xFF == 27:
                break

            # notify
            for person, face in authList:
                self.notify(
                    email_from = self.email_from,
                    email_to = self.email_to,
                    email_password = self.email_password,
                    email_subject = "FaceAuth notification.",
                    email_body = f"{person} has been registed at {self.registedPeople[person]}.",
                    email_attach_img_frame = face
                )

        cap.release()

    def notify(self,
            email_from = None,
            email_to = None,
            email_password = None,
            email_subject = 'FaceAuth notification.',
            email_body = 'This is a test. Some dummy text. Ignore it.',
            email_attach_img_frame = None
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

        if email_attach_img_frame is not None:
            img = cv2.imencode('.jpg', email_attach_img_frame)[1].tostring()
            msg.add_attachment(img, maintype = 'image', subtype = 'jpeg')

        # send message
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(email_from, email_password)
            smtp.send_message(msg)

    def authorize(self):
        pass

def getTimeDiff(dt, fmt = 'm'):
    now = datetime.datetime.now()

    # calculate difference
    diff = (now - dt).total_seconds()

    if fmt[0] == 'm':
        diff = diff / 60
    elif fmt[0] == 'h':
        diff = diff / 360

    return round(diff, 4)
