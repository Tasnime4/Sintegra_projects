#This code is for extract and create a dataset of personnes from webcam
import cv2
import sys

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(1)
i=0
personne="Achref"
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
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), 2)
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

    # Display the resulting frame
    cv2.imshow('Video', frame)
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
