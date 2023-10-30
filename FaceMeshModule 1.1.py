import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import math

numfaceCap = 5



class FacialLandmarkDetector():

    def __init__(self, staticMode=False, maxFace=numfaceCap, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace = maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectionCon, self.minTrackCon)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)
        self.distance = 0

    def findFaceMesh(self, img, draw=False):

        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for FaceLandmarks in self.results.multi_face_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, FaceLandmarks, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpecs,
                                               self.drawSpecs)

                face = []

                for id, lm in enumerate(FaceLandmarks.landmark):
                    if id == 464:
                        self.distanceAdjuster = lm.z

                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    cv.putText(img, str(id), (x, y), cv.QT_FONT_NORMAL, 0.3, (128, 0, 128), 1)
                    # print(id, x, y)
                    face.append([x, y])

                self.distance = self.distance_calc(face[280], face[464])
                print(self.distance, self.distanceAdjuster)
                # 173: Left eye corner 123:left cheek
                # 464: right eye corner 280; right cheek

                faces.append(face)
        return img, faces

    def distance_calc(self, a, b):
        a = np.array(a)  # Top
        b = np.array(b)  # Bottom

        distance = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        return round(distance)

    def average(self, lst):
        return sum(lst) / len(lst)


def main():
    cap = cv.VideoCapture(0)#"a/Me.mp4"
    pTime = 0
    detector = FacialLandmarkDetector()
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    DistanceList = []

    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        DistanceList.append(detector.distance)
        if len(faces) != 0:
            cv.putText(img, 'Face counter:' + str(len(faces)), (int(width) - 150, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 197), 1, cv.LINE_AA)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.QT_FONT_NORMAL, 3, (0, 0, 255), 3)
        cv.imshow("Hello", img)
        cv.waitKey(1)
        avgD = round(detector.average(DistanceList))#detector.distanceAdjuster
        print(avgD)


if __name__ == "__main__":
    main()
