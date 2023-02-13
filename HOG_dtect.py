import numpy as np
import cv2
from dlib import get_frontal_face_detector
import imutils
import copy
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import os
from matplotlib import image
import time
import face_recognition



class CentroidTracker():
	def __init__(self, maxDisappeared=50):
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			return self.objects
		inputCentroids = np.zeros((len(rects), 2), dtype="int")
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())
			D = dist.cdist(np.array(objectCentroids), inputCentroids)
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]
			usedRows = set()
			usedCols = set()
			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				usedRows.add(row)
				usedCols.add(col)
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])
		return self.objects



ct = CentroidTracker(maxDisappeared=5)

# load dlib's HOG + Linear SVM face 
detector = get_frontal_face_detector()
'''OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}'''


name = 'Kotilingesh'
og_name = name
for root,dirs,folder in os.walk(r'D://Python_docs//Face Recognition//faces'):
    for files in folder:
        if os.path.splitext(files)[0] == name:
            if os.path.splitext(files)[1] == '.jpg':
                og_img_path = os.path.join(files,root,name+'.jpg')
            elif os.path.splitext(files)[1] == '.jpeg':
                og_img_path = os.path.join(files,root,name+'.jpeg')
            elif os.path.splitext(files)[1] == '.png':
                og_img_path = os.path.join(files,root,name+'.png')  
            else:
                print("No such person registered.")
                break            
            og_img = face_recognition.load_image_file(og_img_path)
            og_img_encoding = face_recognition.face_encodings(og_img)[0]
            break


known_face_encodings = [
    og_img_encoding
]
known_face_names = [
    name
]

face_locations = []
face_encodings = []
face_names = []

cap = cv2.VideoCapture("D:\\Python_docs\\Face Recognition\\video_20220929_194235.mp4")
cap.set(cv2.CAP_PROP_FRAME_COUNT,0)
i = 0
count = 0
not_same = 0
same = 0
objects = None
process_this_frame = True
detector_count = 0
while True:
    success, frame = cap.read()
    if success:
        cap.set(1,i)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        frame = cv2.resize(frame,(360,640))
        rects = detector(frame)
        rect = [(face.left(),face.top(),face.right(),face.bottom()) for face in rects]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        last_frame = copy.deepcopy(objects)
        
        objects = ct.update(rect)
        face_names = []
        for face in rects:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0, 255, 0), 2)
    
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        process_this_frame = not process_this_frame

        for (top, right, bottom, left), name in zip(face_locations, face_names):

            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            if name != 'Unknown':
                same += 1  
        for (objectID, centroid) in objects.items():
            x,y = centroid
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0], centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            try:
                pass
            except KeyError:
                continue
            if x<180 and last_frame[objectID][0]>180:
                count +=1
                if count>=2:
                    print("Tailgating")
                    cv2.imwrite("New_picture.jpg",frame)
                    break
                elif len(face_locations) == 0:
                    detector_count += 1
                    if detector_count >= 10:
                        break
                
        cv2.putText(frame, "Count : {}".format(count), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.line(frame, (180,0), (180,1280), (255,255,0) ,3)

        frame = imutils.resize(frame, width = 400)
        cv2.imshow('frame',frame)


        if cv2.waitKey(1) == 50:
            break
    if not success:
        break
    i = i + 10

if same <= 8:
    print("Not the same person in the directory")
else:
    name = og_name
    print(name,"entered the room",time.ctime(time.time())) 
cv2.waitKey(0)