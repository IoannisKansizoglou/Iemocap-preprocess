###################### LIBRARIES ######################

import cv2
import numpy as np
import os, os.path, time
import pandas as pd

##################### PARAMETERS #####################

# Starting time
t0 = time.time()
# Threshold for keeping a center movement per frame
c_threshold = 40
# Threshold for keeping a center vs other
d_threshold = 100000
# Values for first cropping
y1, y2, y3, y4, x1, x2, x3, x4 = (130, 230, 140, 240, 120, 240, 500, 630)
# Final size of cropped image
width, height = (96, 96)


###################### FUNCTIONS #####################


### Function for creating folders ###
def create_folder(PATH):
    
    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
        print ('Error: Creating directory' + PATH)


### Function to crop image tracking the face ###
def crop_image(image, left_precenter, right_precenter, sheet, speaker):

    # Set face tracking type
    cascPATH = '/home/gryphonlab/Ioannis/Works/IEMOCAP/CodeFaceDetection/haarcascade_frontalface_alt2.xml'
    face_cascade = cv2.CascadeClassifier(cascPATH)
    # First crop of image to simplify face-detection
    img = image.copy()
    img1 = img[y1:y2, x1:x2]
    img2 = img[y3:y4, x3:x4]
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # Detect faces (-eyes) in the image
    left_faces = face_cascade.detectMultiScale(gray1, scaleFactor=1.01, minNeighbors=5)
    right_faces = face_cascade.detectMultiScale(gray2, scaleFactor=1.01, minNeighbors=5)
    # Track left speaker
    left_center, left_index = select_window(left_faces, left_precenter)
    # Track right speaker
    right_center, right_index = select_window(right_faces, right_precenter)
    # Calculate left crop rectangle
    print(left_center)
    left_a = left_center[0] - int(width/2)
    left_b = left_center[1] - int(height/2)
    # Draw proposed left center, face rectangle and crop rectangle
    cv2.circle(img1,tuple(left_center),2,(0,255,0),2)
    if left_index >= 0:
        x, y, w, h = left_faces[left_index]
        cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.rectangle(img1,(left_a,left_b),(left_a+width,left_b+height),(0,0,255),2)
    # Calculate right crop rectangle
    print(right_center)
    right_a = right_center[0] - int(width/2)
    right_b = right_center[1] - int(height/2)
    # Draw proposed left center, face rectangle and crop rectangle
    cv2.circle(img2,tuple(right_center),2,(0,255,0),2)
    if right_index >= 0:
        x, y, w, h = right_faces[right_index]
        cv2.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.rectangle(img2,(right_a,right_b),(right_a+width,right_b+height),(0,0,255),2)
    # Concatenate left and right images
    img = np.concatenate((img1, img2), axis=1)
    # Show results
    cv2.imshow('Image',img)
    cv2.waitKey(1)
    # Select speaker to crop according to extractionmap.xlsx
    if speaker == 'L':
        
        # Crop image
        left_a += x1
        left_b += y1
        image = image[left_b:left_b+width, left_a:left_a+height]
        
    elif speaker == 'R':

        # Crop image
        right_a += x3
        right_b += y3
        image = image[right_b:right_b+width, right_a:right_a+height]
        #image = image[int((y3+y4)/2)-int(width/2):int((y3+y4)/2)-int(width/2)+width, int((x3+x4)/2)-int(height/2):int((x3+x4)/2)-int(height/2)+height]
        
    else:

        image = np.array([0, 0])

    # Return cropped image and next precenters
    return image, left_center, right_center


### Function to find current speaker ###
def find_speaker(frame, row, xl, sheet):

    # Create DataFrame from current excel's sheet
    df = xl.parse(sheet)
    # Check current frame's position in xlsx row
    if frame < df['iframe'][row]:
        speaker = 'None'
    elif frame <= df['fframe'][row]:
        speaker = df['speaker'][row]
    else:
        speaker = 'None'
        if (df.index == row+1).any():
            row += 1
    
    return speaker, row


### Function to select the best window before cropping
def select_window(faces, precenter):

    # Case[1] of detecting no 'face'
    if np.shape(faces)[0] == 0:
        # Proposed center
        center = precenter
        # False value for index in faces of selected window
        index = -1
    
    else:

        # Index in faces of selected window
        index = 0
        # Case[2] of detecting many 'faces'
        if np.shape(faces)[0] > 1:
            
            # Starting default distance
            dmin = d_threshold
            i = 0

            # Decide which to keep
            for (x,y,w,h) in faces:

                # Compute center
                xc = int(round((2*x+w)/2))
                yc = int(round((2*y+h)/2))
                d = np.linalg.norm(np.array([xc, yc])-np.array(precenter))
                # Change appropriately min and index
                if d < dmin:
                    dmin = d
                    index = i

                i += 1

            # Take values for proposed center
            index = int(index)
            x, y, w, h = faces[index]

        # Case[3] of detecting exactly one 'face'
        else:

            # Take values for proposed center
            x, y, w, h = faces[0]
            xc = int(round((2*x+w)/2))
            yc = int(round((2*y+h)/2))
            dmin = np.linalg.norm(np.array([xc, yc])-np.array(precenter))
            
        # Proposed centre
        xc = int(round((2*x+w)/2))
        yc = int(round((2*y+h)/2))
        # Check distance with precenter threshold
        if dmin < c_threshold:
            # Proposed center is accepted
            center = [xc, yc]
        else:
            # Proposed center is discarded, keep precenter
            center = precenter

        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # Compute big axis of face-detection rectangle
        #s1 = np.array([x, y])
        #s2 = np.array([x+w, y+h])
        #print(np.linalg.norm(s1-s2))

    if precenter == [0, 0]:
        if np.shape(faces)[0]>0:
            x, y, w, h = faces[0]
            xc = int(round((2*x+w)/2))
            yc = int(round((2*y+h)/2))
            center = [xc, yc]
        else:
            center = [int((x2-x1)/2), int((y2-y1)/2)]

    return center, index



######################## MAIN ########################


def main():

    # Folder to store all .jpg files
    imagePATH = '/home/gryphonlab/Ioannis/Works/IEMOCAP/InputFaces'
    create_folder(imagePATH)

    # For printing effect
    counter = 1

    for ses in range(1,2):

        # CHANGE THIS if dataset IEMOCAP is moved
        sessionPATH = '/media/gryphonlab/8847-9F6F/Ioannis/IEMOCAP_full_release/Session'+str(ses)+'/dialog/avi/DivX'
        videos = os.listdir(sessionPATH)

        # Current session folder
        image_sessionPATH = imagePATH + '/Session' + str(ses)
        create_folder(image_sessionPATH)

        # Path to session's excel file
        extractionmapPATH = '/home/gryphonlab/Ioannis/Works/IEMOCAP/Core/cut_extractionmap'+str(ses)+'.xlsx'
        # Read excel file
        xl = pd.ExcelFile(extractionmapPATH)

        # First precenters don't exist
        left_precenter = [0, 0]
        right_precenter = [0, 0]

        for vid in range(28,29):#len(videos)):

            # Current video path
            videoPATH = sessionPATH + '/' + str(videos[vid])

            # Current sheet name
            sheet = videos[vid][:-4]
            print(sheet)

            print(str(counter)+ ') Creating video...' + videoPATH)
            counter += 1

            if os.path.isfile(videoPATH):

                # Folder for frames of current video
                image_videoPATH = image_sessionPATH + '/' + videos[vid]
                create_folder(image_videoPATH)
                # Playing video from file
                cap = cv2.VideoCapture(videoPATH)
                # First frame0.jpg
                currentFrame = 0
                # Row of extractionmap to search according to currentFrame
                currentRow = 0
                # Capture frame-by-frame
                success, frame = cap.read()

                while success:

                    if currentFrame > 9550 and currentFrame < 10074:

                        # Find current speaker
                        currentSpeaker, currentRow = find_speaker(currentFrame, currentRow, xl, sheet)
                        # Crop frame
                        frame, left_precenter, right_precenter = crop_image(frame, left_precenter, right_precenter, sheet, currentSpeaker)
                        # Check that current frame refers to speaker
                        if np.shape(frame)[0]>2:
                            # Save current cropped image in .jpg file
                            name = str(image_videoPATH) + '/frame' + str(currentFrame) + '.jpg'
                            cv2.imwrite(name, frame)

                    # To stop duplicate images
                    print('Current Frame: Frame'+str(currentFrame))
                    currentFrame += 1
                    # Try capture next frame
                    success, frame = cap.read()

        
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        print('Done"')

        # Execution time
        print( 'Execution time of extractFace.py [sec]: ' + str(time.time() - t0) )

# Control runtime
if __name__ == '__main__':
    main()