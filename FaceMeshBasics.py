import cv2 as cv
import mediapipe as mp
import time
cap = cv.VideoCapture("a/Me.mp4")
pTime=0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for FaceLandmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, FaceLandmarks, mpFaceMesh.FACE_CONNECTIONS, drawSpecs, drawSpecs)

            for id,lm in enumerate(FaceLandmarks.landmark):
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y * ih)
                print(id, x, y)



    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.QT_FONT_NORMAL, 3, (0, 0, 255), 3)
    cv.imshow("Image", img)
    cv.waitKey(10)
