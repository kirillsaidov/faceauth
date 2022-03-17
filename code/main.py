import faceauth

print('|-------------- STARTING --------------|')

auth = faceauth.FaceAuth(
	face_model = '../models/face_model98n.pt',
	person_model = '../models/auth_model47n.pt',
	person_img_mean = [0.65625078, 0.48664141, 0.40608295],
	person_img_std = [0.20471508, 0.17793475, 0.16603905],
	conf = 0.5,
    detect_dist_threshold = 0.45
)
auth.launch()

"""
msg = 'Authorazied: {}, {}, {}\n'.format('Kirill Saidov', '16:55', '10/03/2022');
auth.notify(
	email_from = 'from@gmail.com',
	email_to = 'to@gmail.com',
	email_password = '***',
	email_subject = 'FaceAuth notification',
	email_body = msg
)
"""
print('|---------------- DONE ----------------|')
