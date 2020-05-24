import cv2

videoCaptureObject = cv2.VideoCapture(0)
result = True
while(result):
	name = input("Enter your name :")
	ret,frame = videoCaptureObject.read()
	cv2.imwrite(name + ".jpg",frame)
	result = False
videoCaptureObject.release()
cv2.destroyAllWindows()