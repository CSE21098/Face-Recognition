import cv2
import numpy as np

cap  = cv2.VideoCapture(1)
fac = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


face_section = None
skip = 0
face_data =[]
dp = 'D:/HTML/'

file_name  = input("Enter the name of the person : ")
while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
       
    if ret == False:
        continue
    
    faces = fac.detectMultiScale(gray,1.3,5)
    faces = sorted(faces,key=lambda f:f[2]*f[3])    

    for(face) in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        offset =10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        skip +=1
        if(skip%10 == 0):
            face_data.append(face_section)
            print(len(face_data))
        
    #cv2.imshow('Gray',gray)
    
    cv2.imshow('video',frame)
    if face_section is not None:
        cv2.imshow('Smaller',face_section)
    

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

face_data = np.asarray(face_data)
face_data  = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)


np.save(dp + file_name +'.npy',face_data)
print("Saved Successfully Saved")
    
    
cap.release()
cv2.destroyAllWindows()