
########################### LIBRARIES ##########################

import tensorflow as tf
import os, os.path
import pandas as pd
import numpy as np

########################## PARAMETERS ##########################

np.random.seed(10)
datasetPATH = '/home/gryphonlab/Ioannis/Works/IEMOCAP/'
person = ['M', 1]

########################### FUNCTIONS ##########################

# Function to calculate and transpose emotion to number
def transpose_emotion(emotion):

    for i in range(np.shape(emotion)[0]):

        # Check emotion to transpose
        if emotion[i] == 'neu':
            emotion[i] = 0
        elif emotion[i] == 'hap':
            emotion[i] = 1
        elif emotion[i] == 'sur':
            emotion[i] = 2
        elif emotion[i] == 'exc':
            emotion[i] = 3
        elif emotion[i] == 'sad':
            emotion[i] = 4
        elif emotion[i] == 'fru':
            emotion[i] = 5
        elif emotion[i] == 'fea':
            emotion[i] = 6
        elif emotion[i] == 'ang':
            emotion[i] = 7
        elif emotion[i] == 'xxx' or emotion[i] == 'oth' or emotion[i] == 'dis':
            emotion[i] = 100
        
    return emotion


# Function to read values from excel file
def read_excel(sheet, xl):

    # Create DataFrame from current excel's sheet
    df = xl.parse(sheet)
    # Calculate and transpose emotions to number
    #emotion = transpose_emotion( np.array(df['emotion']) )
    # Merge all emovector columns into a numpy array
    valence = np.array(df['emovectorA'])
    arousal = np.array(df['emovectorB'])
    # Normalize emovector values to [-1,1]
    valence = np.add(valence,-3)
    valence = np.divide(valence,2)
    arousal = np.add(arousal,-3)
    arousal = np.divide(arousal,2)
    #print(arousal)
    # Unify every row to a single numpy array: [[emovectorA[0], emovectorB[0],emovectorC[0]], [...], ...] 
    #emovector = np.array([emovector[:,i] for i in range(np.shape(emovector)[1])])
    speaker = np.array(df['speaker'])
    # Return 4 numpy arrays
    return np.array(df['iframe']), np.array(df['fframe']), valence, arousal, speaker


def leave_one_out(inputPATH, person):

    train_facesPATH, train_spectrogramsPATH, train_valence, train_arousal = list(), list(), list(), list()
    test_facesPATH, test_spectrogramsPATH, test_valence, test_arousal = list(), list(), list(), list()

    for ses in range(1,6):

        extractionmapPATH = inputPATH+'Core/cut_extractionmap'+str(ses)+'.xlsx'
        xl = pd.ExcelFile(extractionmapPATH)
        sessionPATH = inputPATH+'InputFaces/Session'+str(ses)+'/'
        videos = os.listdir(sessionPATH)
        
        for vid in videos:

            if vid != '.DS_Store' and vid != 'Thumbs.db':

                sheet = vid[:-4]
                videoPATH = sessionPATH+vid+'/'
                audioPATH = inputPATH+'InputSpectrograms/Session'+str(ses)+'/'+sheet+'.wav/'
                frames = os.listdir(videoPATH)
                # Take values from current excel file
                iframe, fframe, emovectorA, emovectorB, speaker = read_excel(sheet, xl)
                

                for frame in frames:

                    row = 0
                    frame_id = frame[5:-4]
                    
                    while int(frame_id) > fframe[row]:
                        row += 1
                        
                    if (ses==person[1]) and ((vid[5]==person[0] and speaker[row]=='L') or (vid[5]!=person[0] and speaker[row]=='R')):

                        test_facesPATH.append(videoPATH+frame)
                        test_spectrogramsPATH.append(audioPATH+'frame'+frame_id+'.npy')
                        if int(frame_id) >= iframe[row] and int(frame_id) <= fframe[row]:
                            test_valence.append(emovectorA[row])
                            test_arousal.append(emovectorB[row])

                    else:

                        train_facesPATH.append(videoPATH+frame)
                        train_spectrogramsPATH.append(audioPATH+'frame'+frame_id+'.npy')
                        if int(frame_id) >= iframe[row] and int(frame_id) <= fframe[row]:
                            train_valence.append(emovectorA[row])
                            train_arousal.append(emovectorB[row])
                    
    #facesPATH = tf.convert_to_tensor(facesPATH, dtype=tf.string)
    #labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    train_data = np.array([train_facesPATH, train_spectrogramsPATH, train_valence, train_arousal])
    train_data = train_data.T
    test_data = np.array([test_facesPATH, test_spectrogramsPATH, test_valence, test_arousal])
    test_data = test_data.T

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)

    eval_data = test_data[:-int(0.5*np.shape(test_data)[0])]
    pred_data = test_data[-int(0.5*np.shape(test_data)[0]):]

    return train_data, eval_data, pred_data


def read_data(inputPATH):

    facesPATH, spectrogramsPATH, valence, arousal = list(), list(), list(), list()
    
    for ses in range(1,6):

        extractionmapPATH = inputPATH+'Core/cut_extractionmap'+str(ses)+'.xlsx'
        xl = pd.ExcelFile(extractionmapPATH)
        sessionPATH = inputPATH+'InputFaces/Session'+str(ses)+'/'
        videos = os.listdir(sessionPATH)
        
        for vid in videos:

            if vid != '.DS_Store' and vid != 'Thumbs.db':

                sheet = vid[:-4]
                videoPATH = sessionPATH+vid+'/'
                audioPATH = inputPATH+'InputSpectrograms/Session'+str(ses)+'/'+sheet+'.wav/'
                frames = os.listdir(videoPATH)
                # Take values from current excel file
                iframe, fframe, emovectorA, emovectorB, _ = read_excel(sheet, xl)
                

                for frame in frames:

                    row = 0
                    frame_id = frame[5:-4]
                    
                    while int(frame_id) > fframe[row]:
                        row += 1
                        
                    facesPATH.append(videoPATH+frame)
                    spectrogramsPATH.append(audioPATH+'frame'+frame_id+'.npy')
                    if int(frame_id) >= iframe[row] and int(frame_id) <= fframe[row]:
                        valence.append(emovectorA[row])
                        arousal.append(emovectorB[row])
                    
    #facesPATH = tf.convert_to_tensor(facesPATH, dtype=tf.string)
    #labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    data = np.array([facesPATH, spectrogramsPATH, valence, arousal])
    data = data.T

    return data


def split_data(data):

    data = data[:-1]
    np.random.shuffle(data)
    train_data = data[0:int(0.6*np.shape(data)[0])]
    eval_data = data[int(0.6*np.shape(data)[0]):-int(0.2*np.shape(data)[0])]
    pred_data = data[-int(0.2*np.shape(data)[0]):]
    return train_data, eval_data, pred_data


############################# MAIN ############################


def main():

    # Split data completely randomly
    #data = read_data(datasetPATH)
    #train_data, eval_data, pred_data = split_data(data)

    # Split leaving one speaker out
    train_data, eval_data, pred_data = leave_one_out(datasetPATH, person)

    # Extract faces, spectrograms, valence and arousal
    train_faces = train_data.T[0]
    train_spectrograms = train_data.T[1]
    train_valence = train_data.T[2]
    train_arousal = train_data.T[3]
    eval_faces = eval_data.T[0]
    eval_spectrograms = eval_data.T[1]
    eval_valence = eval_data.T[2]
    eval_arousal = eval_data.T[3]
    pred_faces = pred_data.T[0]
    pred_spectrograms = pred_data.T[1]
    pred_valence = pred_data.T[2]
    pred_arousal = pred_data.T[3]

    d1 = { 'train_faces': train_faces, 'train_spectrograms': train_spectrograms,
           'train_valence': train_valence, 'train_arousal': train_arousal }
    d2 = {'eval_faces': eval_faces, 'eval_spectrograms': eval_spectrograms,
          'eval_valence': eval_valence, 'eval_arousal': eval_arousal }
    d3 = { 'pred_faces': pred_faces, 'pred_spectrograms': pred_spectrograms,
           'pred_valence': pred_valence, 'pred_arousal': pred_arousal }

    df1 = pd.DataFrame(data=d1)
    df2 = pd.DataFrame(data=d2)
    df3 = pd.DataFrame(data=d3)

    df1.to_csv(datasetPATH+'Core/regression/training_data.csv')
    df2.to_csv(datasetPATH+'Core/regression/evaluation_data.csv')
    df3.to_csv(datasetPATH+'Core/regression/prediction_data.csv')
    #dataset = tf.data.Dataset.from_tensor_slices(train_data)
    #print(dataset)

# Control runtime
if __name__ == '__main__':
    main()
