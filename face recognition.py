import cv2
import face_recognition as FR
font=cv2.FONT_HERSHEY_SIMPLEX
width=640
height=360
cam=cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)



pavanFace=FR.load_image_file(r'C:\Users\Gv Pavan Sumanth\Downloads\pavan.jfif')
faceLoc=FR.face_locations(pavanFace)[0]
pavanFaceEncode=FR.face_encodings(pavanFace)[0]

vinayFace=FR.load_image_file(r'C:\Users\Gv Pavan Sumanth\Downloads\vinay.jfif')
faceLoc=FR.face_locations(vinayFace)[0]
vinayFaceEncode=FR.face_encodings(vinayFace)[0]


knownEncodings=[pavanFaceEncode,vinayFaceEncode]
names=['Pavan sumanth','Vinay Kumar']

while True:
    ignore,  unknownFace = cam.read()

    unknownFaceRGB=cv2.cvtColor(unknownFace,cv2.COLOR_RGB2BGR)
    faceLocations=FR.face_locations(unknownFaceRGB)
    unknownEncodings=FR.face_encodings(unknownFaceRGB,faceLocations)

    for faceLocation,unknownEncoding in zip(faceLocations,unknownEncodings):
        top,right,bottom,left=faceLocation
        print(faceLocation)
        cv2.rectangle(unknownFace,(left,top),(right,bottom),(255,0,0),3)
        name='Unknown Person'
        matches=FR.compare_faces(knownEncodings,unknownEncoding)
        print(matches)
        if True in matches:
            matchIndex=matches.index(True)
            print(matchIndex)
            print(names[matchIndex])
            name=names[matchIndex]
        cv2.putText(unknownFace,name,(left,top),font,.75,(0,0,255),2)

    cv2.imshow('My Faces',unknownFace)
    if cv2.waitKey(1) & 0xff ==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
