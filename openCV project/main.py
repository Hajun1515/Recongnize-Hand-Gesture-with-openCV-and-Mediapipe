import mediapipe as mp
import cv2 

#inisialisasi mediapipe hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

#function for recognize the gesture hands
def recognize_gesture(hand_landmarks):
    ujung_jempol = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    ujung_telunjuk = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    ujung_jariTengah = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ujung_jariManis = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ujung_kelinking = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    #thumbs up
    if(ujung_jempol.y < ujung_jariManis.y and
        ujung_jempol.y < ujung_jariTengah.y and
        ujung_jempol.y < ujung_kelinking.y and
        ujung_jempol.y < ujung_telunjuk.y):
        return 'THUMBS UP'
        # Peace Sign (Index and Middle finger up)
    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_jariTengah.y < ujung_jempol.y and
        ujung_jariManis.y > ujung_jempol.y and
        ujung_kelinking.y > ujung_jempol.y):
        return "Peace Sign"

    # Fist
    if (ujung_jempol.y > ujung_telunjuk.y and
        ujung_jempol.y > ujung_jariTengah.y and
        ujung_jempol.y > ujung_jariManis.y and
        ujung_jempol.y > ujung_kelinking.y):
        return "Fist"

    # Metal Sign (Index and Pinky finger up)
    if (ujung_telunjuk.y < ujung_jempol.y and
        ujung_kelinking.y < ujung_jempol.y and
        ujung_jariTengah.y > ujung_jempol.y and
        ujung_jariManis.y > ujung_jempol.y):
        return "Metal"
    return 'The gesture is unknown'




#function for detect the gesture hand
def detect_hand_gesture(image,hand):
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    result = hand.process(image_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            #detection gesture
            gesture = recognize_gesture(hand_landmarks)
            #draw the detect hand image/frame
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(image,gesture,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2,cv2.LINE_AA)
    return image
#open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Cant open the camera')
    exit()

while(cap.isOpened):
    ret,frame = cap.read()
    if not ret:
        print('Failed to capture the frame')
        break
    
    frame = detect_hand_gesture(frame,hands)
    
    cv2.imshow('Hand Gesture Recognition',frame)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
