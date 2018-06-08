
########################### LIBRARIES ##########################

import tensorflow as tf
import os, os.path
import pandas as pd
import numpy as np

########################## PARAMETERS ##########################

np.random.seed(10)

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
    emotion = transpose_emotion( np.array(df['emotion']) )
    # Merge all emovector columns into a numpy array
    #emovector = np.array([np.array(df['emovectorA']), np.array(df['emovectorB']), np.array(df['emovectorC'])])
    # Unify every row to a single numpy array: [[emovectorA[0], emovectorB[0],emovectorC[0]], [...], ...] 
    #emovector = np.array([emovector[:,i] for i in range(np.shape(emovector)[1])])
    # Return 4 numpy arrays
    return np.array(df['iframe']), np.array(df['fframe']), emotion


def read_data(datasetPATH):

    facesPATH, spectrogramsPATH, labels = list(), list(), list()
    
    for ses in range(1,6):

        extractionmapPATH = datasetPATH+'Core/cut_extractionmap'+str(ses)+'.xlsx'
        xl = pd.ExcelFile(extractionmapPATH)
        sessionPATH = datasetPATH+'InputFaces/Session'+str(ses)+'/'
        videos = os.listdir(sessionPATH)
        
        for vid in videos:

            if vid != '.DS_Store' and vid != 'Thumbs.db':

                sheet = vid[:-4]
                videoPATH = sessionPATH+vid+'/'
                audioPATH = datasetPATH+'InputSpectrograms/Session'+str(ses)+'/'+sheet+'.wav/'
                frames = os.listdir(videoPATH)
                # Take values from current excel file
                iframe, fframe, emotion = read_excel(sheet, xl)
                

                for frame in frames:

                    row = 0
                    frame_id = frame[5:-4]
                    
                    while int(frame_id) > fframe[row]:
                        row += 1
                        
                    if emotion[row] != 100:

                        facesPATH.append(videoPATH+frame)
                        spectrogramsPATH.append(audioPATH+'frame'+frame_id+'.npy')
                        if int(frame_id) >= iframe[row] and int(frame_id) <= fframe[row]:
                            labels.append(emotion[row])
                    
    #facesPATH = tf.convert_to_tensor(facesPATH, dtype=tf.string)
    #labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    data = np.array([facesPATH, spectrogramsPATH, labels])
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

   
    data = read_data('/home/gryphonlab/Ioannis/Works/IEMOCAP/')
    train_data, eval_data, pred_data = split_data(data)

    train_faces = train_data.T[0]
    train_spectrograms = train_data.T[1]
    train_labels = train_data.T[2]
    eval_faces = eval_data.T[0]
    eval_spectrograms = eval_data.T[1]
    eval_labels = eval_data.T[2]
    pred_faces = pred_data.T[0]
    pred_spectrograms = pred_data.T[1]
    pred_labels = pred_data.T[2]

    d1 = {'train_faces': train_faces, 'train_spectrograms': train_spectrograms, 'train_labels': train_labels}
    d2 = {'eval_faces': eval_faces, 'eval_spectrograms': eval_spectrograms, 'eval_labels': eval_labels}
    d3 = {'pred_faces': pred_faces, 'pred_spectrograms': pred_spectrograms, 'pred_labels': pred_labels}

    df1 = pd.DataFrame(data=d1)
    df2 = pd.DataFrame(data=d2)
    df3 = pd.DataFrame(data=d3)
    df1.to_csv('/home/gryphonlab/Ioannis/Works/IEMOCAP/Core/training_data.csv')
    df2.to_csv('/home/gryphonlab/Ioannis/Works/IEMOCAP/Core/evaluation_data.csv')
    df3.to_csv('/home/gryphonlab/Ioannis/Works/IEMOCAP/Core/prediction_data.csv')
    #dataset = tf.data.Dataset.from_tensor_slices(train_data)
    #print(dataset)

# Control runtime
if __name__ == '__main__':
    main()
