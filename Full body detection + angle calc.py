import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


#Face, Pose, Hand Draw specs
FdrawSpecs = mp_drawing.DrawingSpec(color=(0,0,0),thickness=1, circle_radius=2)
PdrawSpecs = mp_drawing.DrawingSpec(color=(0,0,0),thickness=1,circle_radius=2)
HdrawSpecs = mp_drawing.DrawingSpec(color=(0,0,0),thickness=2,circle_radius=1)
cap = cv2.VideoCapture(0)
#find resolution of video
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

counter = 0
stage = None

#Functions
def calc_angle(a, b, c):
    a = np.array(a)  # Starting joint
    b = np.array(b)  # Middle joint
    c = np.array(c)  # End joint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Save memory improve performance
        image.flags.writeable = False

        # Make Detections
        results = holistic.process(image)

        #Landmarks extraction
        try:
            landmarks = results.pose_landmarks.landmark

            #Get coords
            shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]



            #find angle

            angle = round(calc_angle(shoulder, elbow,wrist))
            # Visualize angle
            cv2.putText(image, str(angle),tuple(np.multiply(elbow, [width, height]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_4)

            # Curl counter logic
            if angle > 125:
                stage = "down"
            if angle < 45 and stage == 'down':
                stage = "up"
                counter += 1

        except:
            pass

        # Find pose land marks names
        # for lndmarks in mp_holistic.PoseLandmark:
        #     print(lndmarks)


        #To extract specific landmark values do: landmarks[mp_holistic.PoseLandmark.a landmark name.value]


        # Rep data
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (46, 24, 197), 2, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Save memory improve performance
        image.flags.writeable = True
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,FdrawSpecs,FdrawSpecs)


        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,HdrawSpecs,HdrawSpecs)

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=1, circle_radius=1)
                                  )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,PdrawSpecs,PdrawSpecs)

        cv2.imshow('Video', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
