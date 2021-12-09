import cv2
import mediapipe as mp
import time
from playsound import playsound
import sys
import pandas as pd
from sklearn.metrics import precision_score

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.50,
                    min_tracking_confidence=0.50)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
finger_points = {3, 4, 7, 8, 11, 12, 15, 16, 19, 20}
finger_dict = {4: "thumb", 8: "index finger", 12: "middle finger", 16: "ring finger", 20: "pinky finger"}
pressed = {"4Left": 0, "4Right": 0, "8Left": 0, "8Right" : 0, "12Left": 0, "12Right": 0,"16Left": 0, "16Right": 0, "20Left": 0, "20Right": 0}
idx_map = {"4Left": 0, "4Right": 1, "8Left": 2, "8Right" : 3, "12Left": 4, "12Right": 5,"16Left": 6, "16Right": 7, "20Left": 8, "20Right": 9}

def check_thumb(hand, prev_x, cur_x):
    if hand == "Left" and prev_x - 35 > cur_x:
        return True
    if hand == "Right" and prev_x + 35 < cur_x:
        return True
    return False


def evaluate_data(video_path, counts):
    count = pd.read_csv("count.csv")
    video = int(video_path[5])
    
    counts_expected = count[count.video == video]["count"].tolist()
    tot = 0
    for i in range(len(counts)):
        tot += (abs(counts[i] - counts_expected[i]) / counts_expected[i])
    mean_pct_error = tot / len(counts)

    print(f"Micro Average Precision: {precision_score(counts, counts_expected, average='micro')}")
    print(f"Mean percent error: {mean_pct_error}")


def process_recorded_video(video_path):
    presses = []
    counts = [0]*10
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        success, img = video.read()
        if not success:
            break
        process_video(img, presses, counts)
        cv2.imshow("Video", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    evaluate_data(video_path, counts)


def process_live_video():
    presses = []
    # counts = pressed.copy()
    counts = [0]*10
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break
        # when using an inverted camera
        img = cv2.flip(img, 1)
        process_video(img, presses, counts)
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def process_video(img, presses, counts):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        #print(results.multi_hand_landmarks)
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            hand = results.multi_handedness[idx].classification[0].label
            prev_id = -2
            prev_y = None
            prev_x = None
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
                        #hand = results.multi_handedness[0].classification[0].label
                        thumb = False
                        if id == 4:
                            thumb = check_thumb(hand, prev_x, cx)
                        p_index = str(id) + hand
                        if cy > prev_y or thumb:
                            if pressed[p_index] == 0:
                                #print(p_index)
                                pressed[p_index] = 1
                                #print(f"{id=}")
                                presses.append(p_index)
                                counts[idx_map[p_index]] += 1
                                if hand == 'Left':
                                    print("The " + "left " + finger_dict[id] + " was put down")
                                    playsound(f"Notes/key{id-1}.mp3", block=False)
                                else:
                                    print("The " + "right " + finger_dict[id] + " was put down")
                                    playsound(f"Notes/key{id}.mp3", block=False)
                        else:
                            pressed[p_index] = 0
                    prev_id = id
                    prev_y = cy
                    prev_x = cx
            # connect the landmarks
            #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # cTime = time.time()
    # fps = 1/(cTime-pTime)
    # pTime = cTime

    # cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


def main():
    # evaluate_data()
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        process_recorded_video(video_path)
        return
    process_live_video()


if __name__ == "__main__":
    main()