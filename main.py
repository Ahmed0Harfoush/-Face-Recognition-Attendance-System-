# Basic & Primitive code for facial recognition

import cv2
import numpy as np
import face_recognition

imgMessi = face_recognition.load_image_file('images/download (2).jpeg')
imgMessi = cv2.cvtColor(imgMessi, cv2.COLOR_BGR2RGB)

imgMessiTest = face_recognition.load_image_file('images/Messi_Test.jpeg')
imgMessiTest = cv2.cvtColor(imgMessiTest, cv2.COLOR_BGR2RGB)

faseLoc = face_recognition.face_locations(imgMessi)[0]
encodeMessi = face_recognition.face_encodings(imgMessi)[0]
cv2.rectangle(imgMessi, (faseLoc[3], faseLoc[0]), (faseLoc[1], faseLoc[2]), (255, 0, 255), 2)
print(faseLoc)


faseLocTest = face_recognition.face_locations(imgMessiTest)[0]
encodeMessiTest = face_recognition.face_encodings(imgMessiTest)[0]
cv2.rectangle(imgMessiTest, (faseLocTest[3], faseLocTest[0]), (faseLocTest[1], faseLocTest[2]), (255, 0, 255), 2)
print(faseLocTest)


results = face_recognition.compare_faces([encodeMessi], encodeMessiTest)
faceDis = face_recognition.face_distance([encodeMessi], encodeMessiTest)
print(results, faceDis)
cv2.putText(imgMessiTest, f"{results}, {round(faceDis[0], 2)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

cv2.imshow("Lionel Messi", imgMessi)
cv2.imshow("Lionel Messi Test", imgMessiTest)

cv2.waitKey(0)
cv2.destroyAllWindows()