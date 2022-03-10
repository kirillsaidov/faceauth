import faceauth

print('|-------------- STARTING --------------|')

auth = faceauth.FaceAuth('../models/face_model98n.pt', '../models/dummy.pt')
auth.launch()

print('|---------------- DONE ----------------|')








