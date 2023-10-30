import cv2 as cv


import mediapipe as mp
import time
numface = 1
class FacialLandmarkDetector():

    def __init__(self, staticMode=False, maxFace=numface, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFace =  maxFace
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFace, self.minDetectionCon, self.minTrackCon)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw = True):
        self.imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for FaceLandmarks in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, FaceLandmarks, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpecs, self.drawSpecs)

                face = []

                # for id, lm in enumerate(FaceLandmarks.landmark):
                #     ih, iw, ic = img.shape
                #     x, y = int(lm.x * iw), int(lm.y * ih)
                #     cv.putText(img, str(id), (x, y), cv.QT_FONT_NORMAL, 0.3, (128, 0, 128), 1)
                #     #print(id, x, y)
                #     face.append([x,y])
                faces.append(face)
        return img, faces

def main():
    cap = cv.VideoCapture("a/Me.mp4")
    pTime = 0
    detector = FacialLandmarkDetector()
    while True:
        success, img = cap.read()




        img, faces = detector.findFaceMesh(img)
        if len(faces)!=0:
            print(len(faces))
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(img, f'FPS: {int(fps)}', (20, 70), cv.QT_FONT_NORMAL, 3, (0, 0, 255), 3)
        cv.imshow("Image", img)
        cv.waitKey(10)

if __name__ == "__main__":

    main()