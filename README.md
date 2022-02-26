# FaceAuth
Face Authenticator

## How does it work?
The Face Authenticator system consists of two subsystems:
1. Face detection
2. Face identification

The first subsystem detects and tracks all faces in the frame. However, only faces that take up 70% of the frame size are further sent to the identification subsystem. It is programmed to do so to lower the system load and to avoid unneeded detection.

## How was it created?
The face detection subsystem was trained using YOLOv5 model to detect faces.

The face identification subsystem utilizes the Convolutional Neural Network as well, but the CNN itself was built from ground up using the PyTorch framework. It was then trained to identify people.  

## LICENSE
All code is licensed by MIT license.
