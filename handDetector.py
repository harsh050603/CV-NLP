import enum
import cv2 as cv #Computer Vision Package
import mediapipe as mp #Library for "noding" fingers (mapping finger joints)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class handDetector:
    def __init__(self,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.hands=mp_hands.Hands(max_num_hands=max_num_hands,min_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence)
        # Min_dete.... checks for distance of hand from the camera.. stops detecting if the hand is too far or too close

    #Find Hand nodes ka coordinates
    def findHandLandmarks(self,image,handnumber=0,draw=False):
        originalImage=image
        image=cv.cvtColor(image,cv.COLOR_BGR2RGB)

        #Using Mediapipe chupa hua code
        results=self.hands.process(image)

        landMarkList=[]
        if results.multi_hand_landmarks:
            hand=results.multi_hand_landmarks[handnumber]

            for id,landmark in enumerate(hand.landmark):
                imgH,imgW,imgC=originalImage.shape #Dimesions of Image
                xPos,yPos=[int(landmark.x*imgW),int(landmark.y*imgH)] # landmarks are ratios to dimesions se multiply karke pixel perfect position of hand
                landMarkList.append([id,xPos,yPos])

            #Drawing On Pic
            if draw:
                mp_draw.draw_landmarks(originalImage,hand,mp_hands.HAND_CONNECTIONS)
            
        return landMarkList