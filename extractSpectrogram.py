
########################### LIBRARIES ##########################

import numpy as np
import os, os.path, time, subprocess
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
import pandas as pd

########################## PARAMETERS ##########################

# Starting time
t0 = time.time()
# Number of Sessions
n_sessions = 5
# Video's frame per second
video_rate = 30
# Window size for the FFT
fft_size = 192
# Distance to slide along the window
step_size = fft_size/13.8
# Threshold for spectrograms (lower filters out more noise)
spec_threshold = 4
# Low cut for the butter bandbass filter (Hz)
lowcut = 500
# High cut for the butter bandbass filter (Hz)
highcut = 15000

########################### FUNCTIONS ##########################


### Function for creating folders ###
def create_folder(PATH):

    try:
        if not os.path.exists(PATH):
            os.makedirs(PATH)
    except OSError:
        print ('Error: Creating directory' + PATH)


# Functions for the butter filter
def butter_bandbass(lowcut, highcut, fs, order=5):

    nyq = 0.5*fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandbass_filter(data, lowcut, highcut, fs, order=5):

    b, a = butter_bandbass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Function to create an overlapped version of X
def overlap(X, window_size, window_step):

    if window_size%2 != 0:
        raise ValueError('Window size must be even!!')
    #Make sure there is an even number of windows before stridetricks
    append = np.zeros((window_size - len(X)%window_size))
    X = np.hstack((X,append))

    ws = window_size
    ss = window_step
    a = X

    valid = len(a) - ws
    nw = int((valid)//ss)
    out = np.ndarray((nw,ws), dtype=a.dtype)

    for i in range(nw):
        # "slide" the window along the samples
        start = int(i*ss)
        stop = int(start+ws)
        out[i] = a[start : stop]

    return out


# Function to compute Short-Time Fourier Transform (STFT)
def stft(X, fftsize=128, step=65, mean_normalize=True, real=False, compute_onesided=True):

    if real:
        local_fft = np.fft.rfft
        cut = -1
    else:
        local_fft = np.fft.fft
        cut = None
    
    if compute_onesided:
        cut = int(fftsize // 2)
    
    if mean_normalize:
        X -= X.mean()

    X = overlap(X, fftsize, step)

    size = fftsize
    win = 0.54 - 0.46 * np.cos(2*np.pi*np.arange(size)/(size-1))
    X = X*win[None]
    X = local_fft(X)[:, :cut]
    return X


# Function to create spectrogram
def pretty_spectrogram(d, log=True, thresh=5, fft_size=512, step_size=64):

    specgram = np.abs(stft(d, fftsize=fft_size, step=step_size, real=False, compute_onesided=True))

    if log == True:
        # Normalize volume to max 1
        specgram /= specgram.max()
        # Take log
        specgram = np.log10(specgram)
        # Set anything less than threshold equal to threshold
        specgram[specgram < -thresh] = -thresh
    else:
        # Set anything less than threshold equal to threshold
        specgram[specgram < thresh] = thresh

    return specgram


############################# MAIN #############################

def main():

    # Folder to store all the spectrograms (.txt files)
    spectrogramPATH = '/home/gryphonlab/Ioannis/Works/IEMOCAP/InputSpectrograms'
    create_folder(spectrogramPATH)

    # Number of sessions to iterate
    for ses in range(1,n_sessions+1):
        
        # Path to session's excel file
        extractionmapPATH = '/home/gryphonlab/Ioannis/Works/IEMOCAP/Core/cut_extractionmap'+str(ses)+'.xlsx'
        # Read excel file
        xl = pd.ExcelFile(extractionmapPATH)
        # List of excel's sheet names
        sheets = xl.sheet_names
        # Path of current session's spectrograms 
        spectrogramPATH = spectrogramPATH+'/Session'+str(ses)+'/'
        create_folder(spectrogramPATH)
        # Directory of mono-wav files
        wavPATH = '/media/gryphonlab/8847-9F6F/Ioannis/IEMOCAP_full_release/Session'+str(ses)+'/dialog/monowav/'
        ###create_folder(wavPATH)
        #videos = os.listdir(sessionPATH)
        
        for sheet in sheets:

                
            # Current video path
            ###videoPATH = sessionPATH + '/' + str(vid)
            # Current audio file name
            fileNAME = sheet+'.wav'
            # Path of current video's spectrograms
            spectrogramPATH = spectrogramPATH+fileNAME+'/'
            create_folder(spectrogramPATH)
            # Extract mono audio-file (.wav) from video-file (.avi)
            ###command = 'ffmpeg -i '+videoPATH+' -ab 160k -ac 1 -ar 44100 -vn '+wavPATH+fileNAME
            ###subprocess.call(command, shell=True)
            # Open wav file
            audio_rate, data = wavfile.read(wavPATH+fileNAME)
            # Filter the audio file
            #data = butter_bandbass_filter(data, lowcut, highcut, audio_rate, order=1)
            # Extract spectrogam frames according to video frames

            # Create DataFrame from current excel's sheet
            sheet_df = xl.parse(sheet)
            # Copy iframe to numpy array
            iframe = np.array(sheet_df['iframe'])
            # Copy fframe to numpy array
            fframe = np.array(sheet_df['fframe'])

            for i in range(0,np.shape(iframe)[0]):

                for frame in range(iframe[i], fframe[i]+1):
                    
                    # Take data simultaneous with current video frame
                    framedata = data[int(frame*audio_rate/video_rate):int((frame+1)*audio_rate/video_rate)]
                    if np.shape(framedata)[0] == 1470:
                        # Compute spectrogram array for current frame
                        wav_spectrogram = pretty_spectrogram(framedata.astype('float64'), fft_size=fft_size, step_size=step_size, \
                                                                log=True, thresh=spec_threshold)
                        # Save spectrogram array to .txt file
                        #np.savetxt(spectrogramPATH+'frame'+str(frame)+'.txt', wav_spectrogram, fmt='%1.5f')
                        # Save to binary
                        wav_spectrogram *= -10000
                        np.save(spectrogramPATH+'frame'+str(frame),wav_spectrogram.astype('uint16'))
                        #wav_spectrogram.astype('uint16').tofile(spectrogramPATH+'frame'+str(frame)+'.txt')
                        #df = pd.DataFrame(wav_spectrogram).astype('float16')
                        #df.to_excel(spectrogramPATH+'frame'+str(frame)+'.xlsx')

            # Return back to path of current session's spectrograms 
            spectrogramPATH = '/home/gryphonlab/Ioannis/Works/IEMOCAP/InputSpectrograms/Session'+str(ses)+'/'

        # Return back to InputSpectograms folder
        spectrogramPATH = '/home/gryphonlab/Ioannis/Works/IEMOCAP/InputSpectrograms'
        print('End of Session: '+str(ses))
                    

        
    # Execution time
    print( 'Execution time of extractSpectrogram.py [sec]: '+ str(time.time() - t0) )


# Control runtime
if __name__ == '__main__':
    main()
