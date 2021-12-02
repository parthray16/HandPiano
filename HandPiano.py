import cv2
import mediapipe as mp
import time
from playsound import playsound

def main():
    cap = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    pTime = 0
    cTime = 0
    finger_points = {3, 4, 7, 8, 11, 12, 15, 16, 19, 20}
    finger_dict = {4: "thumb", 8: "index finger", 12: "middle finger", 16: "ring finger", 20: "pinky finger"}
    while True:
        success, img = cap.read()
        # when using an inverted camera
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                prev_id = -2
                prev_y = None
                for id, lm in enumerate(handLms.landmark):
                    #This will draw finger_tips and the points right below it
                    if id in finger_points:
                        h, w, c = img.shape
                        cx, cy = int(lm.x *w), int(lm.y*h)
                        # draw a circle at the landmark
                        cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
                        #This set of if statements will check if a finger has been put down
                        # and it will print which finger (more testing needed)
                        if id == prev_id + 1:
                            if cy > prev_y:
                                print(f"{id=}")
                                if results.multi_handedness[0].classification[0].label == 'Left':
                                    print("The " + "left " + finger_dict[id] + " was put down")
                                else:
                                    print("The " + "right " + finger_dict[id] + " was put down")
                                playsound(f"Notes/key{id}.mp3", block=False)
                        prev_id = id
                        prev_y = cy
                # connect the landmarks
                #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
if __name__ == "__main__":
    main()