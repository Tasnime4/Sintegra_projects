# This code is for person recognition 
import cv2
import sys
import json
import locale
from watson_developer_cloud import VisualRecognitionV3

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(1)
i=0
j=0
personne="Tasnime"
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
       # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )
	
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
	if x is None:
		print('ok')
	imgCrop = frame[y:y+h,x:x+w]
        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
	cv2.imshow('face', imgCrop)
	name = personne + "/" + personne + str(i) + ".png"	
	i=i+1
	cv2.imwrite(name,imgCrop)
	visual_recognition = VisualRecognitionV3(
		'2018-10-15',
		iam_apikey='KK3iYxGWs-cSirde1O5eJyT9WOTw0KsQA_rmT8ZyBCzy')

	with open(name, 'rb') as images_file:
		classes = visual_recognition.classify(
			images_file,
			threshold='0.6',
			classifier_ids='DefaultCustomModel_1145080004').get_result()
		result=json.dumps(classes['images'][0]['classifiers'][0]['classes'], indent=4)
		if result == '[]':
			print("Inconnu")
			name2 = "Inconnu/" + str(j) + ".png"
			j=j+1
			cv2.imwrite(name2,imgCrop)
		else:
			#print(json.dumps(classes['images'][0]['classifiers'][0]['classes'][0]['class'], indent=4))
			score=json.dumps(classes['images'][0]['classifiers'][0]['classes'][0]['score'], indent=4)			
			if locale.atof(score) > 0.9:
				print(json.dumps(classes['images'][0]['classifiers'][0]['classes'][0]['class'], indent=4))
    # Display the resulting frame
    cv2.imshow('Video', frame)
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
