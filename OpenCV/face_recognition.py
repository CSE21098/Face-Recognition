import numpy as np
import cv2
import os


def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))


def knn(X,Y,k=5):
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        ix = X[i,:-1]
        iy = X[i, -1]


        d = dist(Y,ix)
        vals.append([d,iy])
        
    vals = sorted(vals, key= lambda x: x[0])[:k]

    vals = np.array(vals)[:,-1]
    
    new_vals = np.unique(vals,return_counts=True)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred

cap  = cv2.VideoCapture(1)
fac = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


face_section = None
skip = 0
face_data =[]
dp = 'D:/HTML/'

label = []
class_id = 0
names = {}

for fx in os.listdir(dp):
    if fx.endswith('.npy'):

        names[class_id] = fx [:-4]
        item = np.load(dp+fx)
        face_data.append(item)

        target = class_id*np.ones((item.shape[0]))
        class_id += 1
        label.append(target)

face_datasets = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(label,axis=0).reshape((-1,1))

print(face_datasets.shape)
print(face_labels.shape)

trainset = np.concatenate((face_datasets,face_labels), axis=1)
print(trainset.shape)

while True:
    ret,frame = cap.read()
    
    if ret == False:
        continue
    
    faces = fac.detectMultiScale(frame,1.3,5)   

    for(face) in faces[-1:]:
        x,y,w,h = face
        offset =10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))


        out = knn(trainset,face_section.flatten())

        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2, cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Faces",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()