import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import math
from matplotlib import pyplot as plt

numfaceCap = 100
distanceAdjuster = 0


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
        self.angle = 0
        self.zList = []
        self.rangecheck = False
        self.listZ = []
        self.listDistance = []

    def findFaceMesh(self, img, draw=True):

        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []


        if self.results.multi_face_landmarks:
            for FaceLandmarks in self.results.multi_face_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, FaceLandmarks, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpecs,
                                               self.drawSpecs)

                self.face = []

                for id, lm in enumerate(FaceLandmarks.landmark):
                    if id == 9:


                        self.zList.append(lm.z)
                        self.avgZperTen = self.avgz(self.zList)
                        if self.avgZperTen:

                            self.listDistance.append(self.distance)
                            self.avgZperTen = (np.e**(self.avgZperTen)-0.96)*100

                            self.listZ.append(self.avgZperTen)
                            self.zList.clear()
                            print(self.distance,((-10*self.avgZperTen)-80),((10*self.avgZperTen)-80) - self.distance)
                            # print(round((self.avgZperTen*-10+80) * self.distance,3))
                            # print(round(self.avgZperTen * ((self.distance-80)/-10), 3))
                            if 60 < (self.avgZperTen * self.distance) < 70:
                                self.rangecheck = True
                            else:
                                self.rangecheck = False


                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv.putText(img, str(id), (x, y), cv.QT_FONT_NORMAL, 0.3, (128, 0, 128), 1)
                    # print(id, x, y)
                    self.face.append([x, y])

                self.distance = self.distance_calc(self.face[280], self.face[464])
                cv.putText(img,"Distance:"+str(self.distance),(int((self.face[280][0]+self.face[464][0])/2),
                                                   int((self.face[280][1]+self.face[464][1])/2)),
                           cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2, cv.LINE_4)
                cv.line(img,self.face[280],self.face[464],(0,0,0),2,cv.LINE_AA)
                cv.line(img, self.face[280], self.face[0], (0, 0, 0), 2, cv.LINE_AA)
                self.angle = self.calc_angle(self.face[0], self.face[280], self.face[464])

                # 173: Left eye corner 123:left cheek
                # 464: right eye corner 280; right cheek

                faces.append(self.face)
        return img, faces

    def distance_calc(self, a, b):
        a = np.array(a)  # Top
        b = np.array(b)  # Bottom

        distance = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        return round(distance)

    def average(self, lst):
        return sum(lst) / len(lst)

    def calc_angle(self, a, b, c):
        a = np.array(a)  # Starting joint
        b = np.array(b)  # Middle joint
        c = np.array(c)  # End joint

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return round(angle,2)

    def avgz(self, zlist):
        if len(zlist) == 10:
            avgZ = self.average(zlist)
            for i in zlist:
                if i > avgZ*2 or i < avgZ*0.33:
                    zlist.remove(i)
            return self.average(zlist)
        else:
            return

def main():
    pTime = 0
    cap = cv.VideoCapture(0)#"a/Me.mp4"
    detector = FacialLandmarkDetector()
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    DistanceList = []
    AngleList = []
    x=0

    while True:
        success, img = cap.read()
        # img = cv.flip(img,1)
        img, faces = detector.findFaceMesh(img)
        DistanceList.append(detector.distance)
        AngleList.append(detector.angle)
        if len(faces) != 0:
            cv.putText(img, 'Face counter:' + str(len(faces)), (int(width) - 150, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 197), 1, cv.LINE_AA)
        if detector.rangecheck:
            cv.putText(img, 'In range', (int(width) - 150, 42),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 140, 197), 1, cv.LINE_AA)
        else:
            cv.putText(img, 'Out of range', (int(width) - 150, 42),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 140, 197), 1, cv.LINE_AA)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.QT_FONT_NORMAL, 3, (0, 0, 255), 3)
        cv.putText(img, str(detector.angle),
                   detector.face[280], cv.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 0, 255), 2, cv.LINE_4)
        cv.imshow("Hello", img)
        cv.waitKey(1)
        # print("Distance:", detector.distance, "Angle:", detector.angle)
        # avgD = round(detector.average(DistanceList))
        # avgA = round(detector.average(AngleList), 2)
        # print("Average Distance:", avgD, "Average Angle:", avgA)

        x+=1
    plt.plot(detector.listZ, detector.listDistance)
    plt.show()


if __name__ == "__main__":
    main()
