import cv2 as cv
import os
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Parameters
width, height = 1280, 720
gestureThreshold = 400
folderPath = "Presentation"

# Camera Setup
cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Hand Detector
detectorHand = HandDetector(detectionCon=0.8, maxHands=1)

# Variables
imgList = []
delay = 10
buttonPressed = False
counter = 0
imgNumber = 0
annotations = [[]]
annotationNumber = -1
annotationStart = False
hs, ws = int(120 * 1), int(213 * 1)  # width and height of small image

# Get list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

while True:
    # Get image frame
    ret, img = cap.read()
    img = cv.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv.imread(pathFullImage)

    # Find the hand and its landmarks
    hands, img = detectorHand.findHands(img)  # with draw
    # Draw Gesture Threshold line
    cv.line(img, (0, gestureThreshold), (width, gestureThreshold), (0, 255, 0), 10)

    if hands and buttonPressed is False:  # If hand is detected

        hand = hands[0]
        cx, cy = hand["center"]
        lmList = hand["lmList"]  # List of 21 Landmark points
        fingers = detectorHand.fingersUp(hand)  # List of which fingers are up

        # Constrain values for easier drawing
        xVal = int(np.interp(lmList[8][0], [width // 2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal

        if cy <= gestureThreshold:  # If hand is at the height of the face
            #Gesture 1: Left Slide
            if fingers == [1, 0, 0, 0, 0]:
                print("Left")
                buttonPressed = True
                if imgNumber > 0:
                    imgNumber -= 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
            
            #Gesture 2: Right Slide
            if fingers == [0, 0, 0, 0, 1]:
                print("Right")
                buttonPressed = True
                if imgNumber < len(pathImages) - 1:
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
        
        #Gesture 3: Show Pointer
        if fingers == [0, 1, 0, 0, 0]:
            cv.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv.FILLED)

        #Gesture 4: Draw Pointers
        if fingers == [0, 1, 1, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            print(annotationNumber)
            annotations[annotationNumber].append(indexFinger)
            cv.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv.FILLED)

        else:
            annotationStart = False

        #Gesture 5: Erase
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

    else:
        annotationStart = False

    #Whenever changing slides, ensuring delay.
    if buttonPressed:
        counter += 1
        if counter > delay:
            counter = 0
            buttonPressed = False

    #Ensuring each annotation is different from the other
    for i, annotation in enumerate(annotations):
        for j in range(len(annotation)):
            if j != 0:
                cv.line(imgCurrent, annotation[j - 1], annotation[j], (0, 0, 200), 12)

    imgSmall = cv.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws: w] = imgSmall

    cv.imshow("Slides", imgCurrent)
    cv.imshow("Image", img)

    #To exit
    key = cv.waitKey(1)
    if key == ord('q'):
        break
