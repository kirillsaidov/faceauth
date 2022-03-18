import faceauth

print('|-------------- STARTING --------------|')

auth = faceauth.FaceAuth(
	face_model = '../models/face_model98n.pt',
	person_model = '../models/auth_model47n.pt',
	person_img_mean = [0.5, 0.5, 0.5],
	person_img_std = [0.5, 0.5, 0.5],
	conf = 0.5,
    detect_dist_threshold = 0.4,
	timeDiff = 1, # 1 minute
	email_from = 'from@gmail.com',
	email_to = 'to@gmail.com',
	email_password = '***'
)
auth.launch()

print('|---------------- DONE ----------------|')
