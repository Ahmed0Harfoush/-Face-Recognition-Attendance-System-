# advanced code for facial recognition

import cv2
import numpy as np
import face_recognition
import os
import time
from datetime import datetime

path = 'imagesAttendence'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendence(name):
    with open('Record_Attendence.csv', 'r+') as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4 ,x2 * 4,y2 * 4,x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6 , y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendence(name)


    cv2.imshow('webcam', img)
    cv2.waitKey(1)























#
# ---
#
# ### **1. Importing Required Libraries**
# ```python
# import cv2
# import numpy as np
# import face_recognition
# import os
# import time
# from datetime import datetime
# ```
# - `cv2` (OpenCV) - For image processing and webcam access.
# - `numpy` - For numerical operations.
# - `face_recognition` - For face detection and encoding.
# - `os` - For file operations.
# - `time` and `datetime` - For time-related operations.
#
# ---
#
# ### **2. Loading Images for Recognition**
# ```python
# path = 'imagesAttendence'
# images = []
# classNames = []
# myList = os.listdir(path)  # Get list of image files in the directory
# ```
# - `path = 'imagesAttendence'`: Defines the folder containing known images.
# - `os.listdir(path)`: Lists all images in the folder.
#
# **Looping through the images:**
# ```python
# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')  # Read the image
#     images.append(curImg)  # Store the image
#     classNames.append(os.path.splitext(cl)[0])  # Store the name (without extension)
# ```
# - `cv2.imread(f'{path}/{cl}')` reads the image.
# - `os.path.splitext(cl)[0]` extracts the name of the person from the file name.
#
# ---
#
# ### **3. Function to Encode Faces**
# ```python
# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (needed for face_recognition)
#         encode = face_recognition.face_encodings(img)[0]  # Encode the face
#         encodeList.append(encode)
#     return encodeList
# ```
# - Converts images to **RGB** (required for `face_recognition`).
# - Extracts **face encodings** (unique features for each face).
# - Stores encodings in a list.
#
# ---
#
# ### **4. Function to Mark Attendance**
# ```python
# def markAttendence(name):
#     with open('Record_Attendence.csv', 'r+') as f:  # Open CSV file
#         myDataList = f.readline()  # Read existing data
#         nameList = []
#
#         for line in myDataList:  # Extract names from records
#             entry = line.split(',')
#             nameList.append(entry[0])
#
#         if name not in nameList:  # If name is not already recorded
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')  # Get current time
#             f.writelines(f'\n{name}, {dtString}')  # Write new attendance entry
# ```
# - Opens `Record_Attendence.csv`.
# - Reads existing attendance.
# - Checks if the person is already marked.
# - If not, it writes their name and timestamp.
#
# ---
#
# ### **5. Encoding Faces from Known Images**
# ```python
# encodeListKnown = findEncodings(images)
# print('Encoding Complete')
# ```
# - Calls `findEncodings()` to process all images.
# - Stores known face encodings in `encodeListKnown`.
# - Prints a message when encoding is complete.
#
# ---
#
# ### **6. Capturing Live Video for Face Recognition**
# ```python
# cap = cv2.VideoCapture(0)  # Open webcam
# ```
# - `0` refers to the default webcam.
#
# ---
#
# ### **7. Processing Frames from Webcam**
# ```python
# while True:
#     success, img = cap.read()  # Capture frame from webcam
#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Reduce size for faster processing
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert to RGB
# ```
# - Captures a frame from the webcam.
# - Resizes it to **1/4 of the original size** to improve performance.
# - Converts it to **RGB** for face recognition.
#
# ---
#
# ### **8. Detecting and Encoding Faces in the Webcam Frame**
# ```python
# facesCurFrame = face_recognition.face_locations(imgS)  # Detect faces
# encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)  # Encode detected faces
# ```
# - Detects **all faces** in the current frame.
# - Extracts **face encodings** from detected faces.
#
# ---
#
# ### **9. Comparing Detected Faces with Known Faces**
# ```python
# for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#     matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # Compare faces
#     faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # Compute distances
#     matchIndex = np.argmin(faceDis)  # Find best match
# ```
# - Compares detected faces with known face encodings.
# - Finds the **best match** using face distances.
#
# ---
#
# ### **10. Drawing Bounding Boxes and Marking Attendance**
# ```python
# if matches[matchIndex]:
#     name = classNames[matchIndex].upper()  # Get name of matched face
#     print(name)
#
#     y1, x2, y2, x1 = faceLoc
#     y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back to original size
#
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box
#     cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # Draw label background
#     cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Display name
#
#     markAttendence(name)  # Record attendance
# ```
# - If a match is found:
#   - Extracts **face location**.
#   - Draws a **bounding box** around the face.
#   - Displays the **name** of the person.
#   - Calls `markAttendence(name)` to store attendance.
#
# ---
#
# ### **11. Displaying the Webcam Feed**
# ```python
# cv2.imshow('webcam', img)
# cv2.waitKey(1)  # Refresh every frame
# ```
# - Shows the webcam feed with detected faces.
# - Runs in a loop until manually stopped.
#
# ---
#




#  Arab
#
# ---
#
# ### **1. استيراد المكتبات المطلوبة**
# ```python
# import cv2
# import numpy as np
# import face_recognition
# import os
# import time
# from datetime import datetime
# ```
# - `cv2` (OpenCV) - لمعالجة الصور والوصول إلى كاميرا الويب.
# - `numpy` - للعمليات الحسابية.
# - `face_recognition` - لاكتشاف الوجوه والتعرف عليها.
# - `os` - للتعامل مع الملفات والمجلدات.
# - `time` و `datetime` - للعمل مع التوقيت والتواريخ.
#
# ---
#
# ### **2. تحميل صور الأشخاص المعروفين للتعرف عليهم**
# ```python
# path = 'imagesAttendence'
# images = []
# classNames = []
# myList = os.listdir(path)  # الحصول على قائمة الصور في المجلد
# ```
# - `path = 'imagesAttendence'`: يحدد المجلد الذي يحتوي على صور الأشخاص المعروفين.
# - `os.listdir(path)`: يجلب جميع الصور من المجلد.
#
# **تحميل الصور وتحليل أسمائها:**
# ```python
# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')  # قراءة الصورة
#     images.append(curImg)  # تخزين الصورة
#     classNames.append(os.path.splitext(cl)[0])  # استخراج الاسم بدون الامتداد
# ```
# - `cv2.imread(f'{path}/{cl}')` يقوم بقراءة الصورة.
# - `os.path.splitext(cl)[0]` يستخرج اسم الشخص من اسم الملف.
#
# ---
#
# ### **3. دالة لتحليل وتشفير الوجوه**
# ```python
# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # تحويل الصورة إلى RGB
#         encode = face_recognition.face_encodings(img)[0]  # استخراج ميزات الوجه
#         encodeList.append(encode)
#     return encodeList
# ```
# - يحول الصور إلى **RGB** (مطلوب من `face_recognition`).
# - يستخرج **ترميز الوجه** (مجموعة من الميزات الفريدة لكل وجه).
# - يخزن الترميزات في قائمة.
#
# ---
#
# ### **4. دالة لتسجيل الحضور**
# ```python
# def markAttendence(name):
#     with open('Record_Attendence.csv', 'r+') as f:  # فتح ملف الحضور
#         myDataList = f.readline()  # قراءة البيانات الموجودة
#         nameList = []
#
#         for line in myDataList:  # استخراج الأسماء من السجل
#             entry = line.split(',')
#             nameList.append(entry[0])
#
#         if name not in nameList:  # إذا لم يكن الاسم موجودًا بالفعل
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')  # الحصول على الوقت الحالي
#             f.writelines(f'\n{name}, {dtString}')  # تسجيل الحضور
# ```
# - يفتح ملف `Record_Attendence.csv`.
# - يقرأ الأسماء المسجلة بالفعل.
# - إذا لم يكن الاسم مسجلًا، يتم **إضافة الاسم مع توقيت الحضور**.
#
# ---
#
# ### **5. تشفير جميع الوجوه المخزنة**
# ```python
# encodeListKnown = findEncodings(images)
# print('تم الانتهاء من التشفير')
# ```
# - يستدعي `findEncodings()` لمعالجة جميع الصور.
# - يخزن الترميزات المعروفة في `encodeListKnown`.
# - يطبع رسالة عند انتهاء التشفير.
#
# ---
#
# ### **6. تشغيل كاميرا الويب والتقاط الفيديو**
# ```python
# cap = cv2.VideoCapture(0)  # فتح كاميرا الويب
# ```
# - `0` يشير إلى الكاميرا الافتراضية.
#
# ---
#
# ### **7. معالجة الإطارات من الكاميرا**
# ```python
# while True:
#     success, img = cap.read()  # التقاط صورة من الكاميرا
#     imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # تصغير حجم الصورة لتسريع الأداء
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # تحويل الصورة إلى RGB
# ```
# - يلتقط صورة من كاميرا الويب.
# - يقلل حجم الصورة إلى **1/4 من الحجم الأصلي** لتسريع الأداء.
# - يحولها إلى **RGB** لتناسب `face_recognition`.
#
# ---
#
# ### **8. اكتشاف وتشفير الوجوه في الإطار الحالي**
# ```python
# facesCurFrame = face_recognition.face_locations(imgS)  # اكتشاف الوجوه
# encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)  # تشفير الوجوه المكتشفة
# ```
# - يكتشف **كل الوجوه** في الصورة الحالية.
# - يستخرج **ترميزات الوجوه** المكتشفة.
#
# ---
#
# ### **9. مقارنة الوجوه المكتشفة مع الوجوه المعروفة**
# ```python
# for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#     matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # مقارنة الوجوه
#     faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # حساب المسافة بين الوجوه
#     matchIndex = np.argmin(faceDis)  # اختيار أفضل تطابق
# ```
# - يقارن الوجوه المكتشفة مع **الوجوه المعروفة**.
# - يجد **أفضل تطابق** باستخدام مسافة الوجه.
#
# ---
#
# ### **10. رسم مستطيل حول الوجه وتسجيل الحضور**
# ```python
# if matches[matchIndex]:
#     name = classNames[matchIndex].upper()  # جلب اسم الشخص المطابق
#     print(name)
#
#     y1, x2, y2, x1 = faceLoc
#     y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # إعادة الحجم الأصلي
#
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # رسم مستطيل حول الوجه
#     cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # خلفية الاسم
#     cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # عرض الاسم
#
#     markAttendence(name)  # تسجيل الحضور
# ```
# - إذا تم العثور على تطابق:
#   - يستخرج **موقع الوجه**.
#   - يرسم **مستطيلًا حول الوجه**.
#   - يعرض **اسم الشخص**.
#   - يسجل الحضور في ملف `Record_Attendence.csv`.
#
# ---
#
# ### **11. عرض الفيديو المباشر**
# ```python
# cv2.imshow('webcam', img)
# cv2.waitKey(1)  # تحديث كل إطار
# ```
# - يعرض فيديو مباشر مع تحديد الأشخاص المعروفين.
# - يستمر في التحديث حتى يتم إيقافه يدويًا.
#
# ---