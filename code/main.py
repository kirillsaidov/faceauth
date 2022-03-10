import faceauth

print('|-------------- STARTING --------------|')

auth = faceauth.FaceAuth('../models/face_model98n.pt', '../models/dummy.pt')
auth.launch()

msg = 'Authorazied: {}, {}, {}\n'.format('Kirill Saidov', '16:55', '10/03/2022');

auth.notify(
	email_from = 'from@gmail.com',
	email_to = 'to@gmail.com',
	email_password = '***',
	email_subject = 'FaceAuth notification',
	email_body = msg
)

print('|---------------- DONE ----------------|')








