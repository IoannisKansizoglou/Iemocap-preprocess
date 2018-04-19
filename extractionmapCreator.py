
########################### LIBRARIES ##########################

import numpy as np
import os, os.path, time
import pandas as pd

########################## PARAMETERS ##########################

# Starting time
t0 = time.time()
# Number of Sessions
n_sessions = 5
# Video's frame per second
video_rate = 30
# Skip lines when reading from EmoEvaluation files
first_line = 2

########################### FUNCTIONS ##########################


### Function for creating folders ###
def create_folder(PATH):
    
	# If the path doesn't exist try to make folder
	try:
		if not os.path.exists(PATH):
			os.makedirs(PATH)
	except OSError:
		print ('Error: Creating directory ' + PATH)


### Function for reading time values from EmoEvaluation files ###
def read_time_values(line, i, Ti, Tf):

	# Move one character where number starts
	i += 1
	# Counter to the last position in Region of Interest
	f = i
	# Stop when space is found
	while line[f] != ' ':
		f += 1
	# Append this string as float to Ti list
	Ti.append(float(line[i:f]))
	# Pass the string ' - '
	i = f + 3
	f = i
	# Key-character for stopping recognizing numbers
	while line[f] != ']':
		f += 1
	# Append this string as float to Tf list
	Tf.append(float(line[i:f]))
	# Go to next character
	i = f + 1
	# Return time lists and current posiiton in line
	return Ti, Tf, i

'''
### Function for reading emotion and transpose to values from EmoEvaluation files ###
def read_emotion_values1(line, i, emotion_elements):
	
	# Skip string 'C-xx:\t'
	i += 6
	# Counter to the last position in Region of Interest
	f = i
	# Stop when semicolon is found
	while line[f] != ';':
		f += 1
	# Append this string as string to emotion_list
	emotion_elements.append(str(line[i:f]))
	# Return emotions' list and current position in line
	return emotion_elements, i
'''

### Function for reading emotion and transpose to values from EmoEvaluation files ###
def read_emotion_values(line, i, emotion_elements):
	
	# Go to first letter of emotion
	i += 1
	# Last position in Region of Interest
	f = i+3
	# Append this string as string to emotion_list
	emotion_elements.append(str(line[i:f]))
	# Move i to the end
	i = f
	# Return emotions' list and current position in line
	return emotion_elements, i

### Function for reading emovector values from EmoEvaluation files ###
def read_emovector_values(line, i, emovector):

	# List to store 3 values of current emovector
	emovector_list = []
	# Move one character where number starts
	i += 1
	# Counter to the last position in Region of Interest
	f = i
	# Stop when comma is found
	while line[f] != ',':
		f += 1
	# Append this string as first float to emovector_list
	emovector_list.append(float(line[i:f]))
	# Pass the string ', '
	i = f + 2
	f = i
	# Stop when comma is found
	while line[f] != ',':
		f += 1
	# Append this string as second float to emovector_list
	emovector_list.append(float(line[i:f]))
	# Pass the string ', '
	i = f + 2
	f = i
	# Stop when ending close is found
	while line[f] != ']':
		f += 1
	# Append this string as third float to emovector_list
	emovector_list.append(float(line[i:f]))
	# Append current emovector_list to total emoctors' list
	emovector.append(emovector_list)
	# Go to next character
	i = f + 1
	# Return emovectors' list and current position in line
	return emovector, i


### Function for merging emotion values to full emotion list ###
def merge_emotion_elements(emotion_elements, emotion):
	# Append emotion elements as an element of emotion list
	emotion.append(emotion_elements)
	# Return emotion list
	return emotion


### Function to specify who is speaking for every frame
def find_speaker(iframe, fileNAME):

	# Initialize speaker list
	speaker = []
	# Check from file name if basic speaker is Female
	if fileNAME[5] == 'F':
		# Female speaker is on the left side
		speaker.append('L')
		# Check next frame period
		i = 1
		# Iterate while iframe keeps to increase
		while iframe[i] > iframe[i-1]:
			# Add female speaker to left side
			speaker.append('L')
			i += 1
		# Iterate until end of frames
		while i < np.shape(iframe)[0]:
			# Add male speaker to right side
			speaker.append('R')
			i += 1
	# From file name basic speaker is male
	else:
		# Female speaker is on the right side
		speaker.append('R')
		# Check next frame period
		i = 1
		# Iterate while iframe keeps to increase
		while iframe[i] > iframe[i-1]:
			# Add female speaker to right side
			speaker.append('R')
			i += 1
		# Iterate until end of frames
		while i < np.shape(iframe)[0]:
			# Add male speaker to left side
			speaker.append('L')
			i += 1
	
	# Return speaker sequence as numpy array
	return np.array(speaker)


# Function to cut frames in which both speakers talk
def cut_dual_frames(iframe, fframe, speaker):

    # Search all the sheet
    for i in range(np.shape(iframe)[0]-1):
        # Check that the speaker changes and there is an overlap in speakers' utterances
        if speaker[i] != speaker[i+1] and fframe[i] > iframe[i+1]:
            # Swap values of iframe and fframe which corresponds to cut of dual-speaker frames
            fframe[i], iframe[i+1] = iframe[i+1], fframe[i]
        
    # Return final cut iframe and fframe
    return iframe, fframe

'''
### Function for reading values from EmoEvaluation files ###
def handle_values1(fileNAME, PATH, writer, rate):
	
	# Open text-file
	with open(PATH, 'r', encoding='latin-1') as fid:
		# Read and save text-file to line list
		lines = fid.readlines()
		# Skip first lines
		lines = lines[first_line:len(lines)]
		
		######################
		### Initial States ###
		######################
		# Lists of starting and ending times of interest
		Ti = []
		Tf = []
		# Lists of emotions and emotion vectors
		emotion = []
		emovector = []
		# List to store emotion elements until a full emotion list is accomplished
		emotion_elements = []
		######################
		
		# Scan all lines
		for line in lines:
			# Check first character of current line
			i = 0
			# Key-character for recognizing line with time and emovector values
			if line[i] == '[':
				# First read time values
				Ti, Tf, i = read_time_values(line , i, Ti, Tf)
				# Pass next string until emovector values
				i += 24
				while line[i] != '[':
					i += 1
				# Read emovector values
				emovector, i = read_emovector_values(line, i, emovector)
			# Key-character for recognizing line with emotion values
			elif len(line) > 1 and line[i:i+2] == 'C-':
				# Read emotion values
				emotion_elements, i = read_emotion_values1(line, i, emotion_elements)
				# Check if a full emotion list is accomplished
			elif line[0] == '\n':
					# Append emotion elements as one element to emotion list
					emotion = merge_emotion_elements(emotion_elements, emotion)
					# Initialize emotionm elements
					emotion_elements = []

		# Initial frame is the first frame (int) AFTER Ti*rate  
		iframe = np.add( np.multiply(Ti,rate).astype(int),1 )
		# Final frame is the last frame (int) BEFORE Tf*rate
		fframe = np.multiply(Tf,rate).astype(int)
		# Save values to DataFrame
		save_dataFrame1(iframe, fframe, emotion, emovector, fileNAME, writer)
	
	# Function DOESN'T return anything			
	pass


### Function to create dataFrame using pandas ###
def save_dataFrame1(iframe, fframe, emotion, emovector, fileNAME, writer):
	
	# Append all data to columns as shown below
	data =	{
    'iframe': iframe, 'fframe': fframe, 'emotionA': [emotion[i][0] for i in range(np.shape(emotion)[0])], \
    'emotionB': [emotion[i][1] for i in range(np.shape(emotion)[0])], 'emotionC': [emotion[i][2] for i in range(np.shape(emotion)[0])], \
    'emovectorA': [emovector[i][0] for i in range(np.shape(emovector)[0])], \
	'emovectorB': [emovector[i][1] for i in range(np.shape(emovector)[0])], 'emovectorC': [emovector[i][2] for i in range(np.shape(emovector)[0])]
       		} #'emotionD': [emotion[i][3] for i in range(np.shape(emotion)[0])],
	# Create DataFrame from data
	df = pd.DataFrame(data)
	# Write Dataframe to session's excel and save
	df.to_excel(writer, fileNAME)
	writer.save()
	# Function DOESN'T return anything
	pass
'''

### Function for reading values from EmoEvaluation files ###
def handle_values(fileNAME, PATH, writer, rate):
	
	# Open text-file
	with open(PATH, 'r', encoding='latin-1') as fid:
		# Read and save text-file to line list
		lines = fid.readlines()
		# Skip first lines
		lines = lines[first_line:len(lines)]
		
		######################
		### Initial States ###
		######################
		# Lists of starting and ending times of interest
		Ti = []
		Tf = []
		# Lists of emotions and emotion vectors
		emotion = []
		emovector = []
		######################
		
		# Scan all lines
		for line in lines:
			# Check first character of current line
			i = 0
			# Key-character for recognizing line with time and emovector values
			if line[i] == '[':
				# First read time values
				Ti, Tf, i = read_time_values(line , i, Ti, Tf)
				# Pass next string until emotion value
				i += 20
				while line[i] != '\t':
					i += 1
				# Read sum emotion 
				emotion, i = read_emotion_values(line, i, emotion)
				# Pass string until emovector values
				while line[i] != '[':
					i += 1
				# Read emovector values
				emovector, i = read_emovector_values(line, i, emovector)

		# Initial frame is the first frame (int) AFTER Ti*rate  
		iframe = np.add( np.multiply(Ti,rate).astype(int),1 )
		# Final frame is the last frame (int) BEFORE Tf*rate
		fframe = np.multiply(Tf,rate).astype(int)
		# Find who is the speaker of every frame
		speaker = find_speaker(iframe, fileNAME)
		# Save values to DataFrame
		save_dataFrame(iframe, fframe, emotion, emovector, speaker, fileNAME, writer)
	
	# Function DOESN'T return anything			
	pass


### Function to create dataFrame using pandas ###
def save_dataFrame(iframe, fframe, emotion, emovector, speaker, fileNAME, writer):
	
	# Append all data to columns as shown below
	data =	{
    'iframe': iframe, 'fframe': fframe, 'emotion': emotion, 'speaker': speaker, \
    'emovectorA': [emovector[i][0] for i in range(np.shape(emovector)[0])], \
	'emovectorB': [emovector[i][1] for i in range(np.shape(emovector)[0])], \
	'emovectorC': [emovector[i][2] for i in range(np.shape(emovector)[0])]
       		}
	# Create DataFrame from data
	df = pd.DataFrame(data)
	# Sort DataFrame according to 'iframe' column
	sorted_df = df.sort_values( by=['iframe'] )
	
	sorted_df = sorted_df.reset_index(drop=True)
	sorted_df['iframe'], sorted_df['fframe'] = cut_dual_frames(sorted_df['iframe'], sorted_df['fframe'], sorted_df['speaker'])
	# Write Dataframe to session's excel and save
	sorted_df.to_excel(writer, fileNAME)
	writer.save()
	# Function DOESN'T return anything
	pass


############################# MAIN #############################


def main():

	# Folder to store the extractionmaps
	create_folder('/home/gryphonlab/Ioannis/Works/IEMOCAP/Core')

	# Number of sessions to iterate
	for ses in range(1,n_sessions+1):
		
		########################## CHECK PATH??? ##########################
		# CHANGE THIS if dataset IEMOCAP is moved
		EMO_EVAL_PATH = '/media/gryphonlab/8847-9F6F/Ioannis/IEMOCAP_full_release/Session'+str(ses)+'/dialog/EmoEvaluation/'
		# Take list of files in EMO_EVAL_PATH
		evaluations = os.listdir(EMO_EVAL_PATH)
		# Skip non .txt files-folders
		evaluations=[i for i in evaluations if i[-4:]=='.txt']
		# Create session's extractionmap to write the values 
		writer = pd.ExcelWriter( '/home/gryphonlab/Ioannis/Works/IEMOCAP/Core/cut_extractionmap'+str(ses)+'.xlsx', engine='openpyxl' )
		# Iterate for all EmoEvaluation files in EMO_EVAL_PATH
		for eval in evaluations:
			
			# Get values from current EmoEvaluation file 
			handle_values(eval[:-4], EMO_EVAL_PATH+eval, writer, video_rate)
			
		
	# Execution time
	print( 'Execution time of extractionmapCreator.py [sec]: '+ str(time.time() - t0) )


# Control runtime
if __name__ == '__main__':
    main()
