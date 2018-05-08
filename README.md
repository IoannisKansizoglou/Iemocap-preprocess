# iemocap_preprocess
Multimodal preprocessing on IEMOCAP dataset

# Upload_3 08.05.18

- multiCNN.py

    Added: EVAL mode

# Upload_2: 07.05.18

- extractionmapCreator.py

    UNCHANGED
    
- extractSpectrogram.py

    UNCHANGED
  
- extractFace.py

    UNCHANGED
    
- createInputs.py
    
    Produce training_data.csv, evaluation_data.csv and prediction_data.csv
    input maps for multiCNN
    
- multiCNN.py

    Includes: data import methods, multiCNN model, TRAIN method, model save

# Upload_1: 19.04.18

- extractionmapCreator.py 
    
    Produce extractionmap_cut.xlsx 
    
    execution time: 252.5967 sec
    
- extractSpectrogram.py

    Produce specified -from extractionmaps- spectrograms on .npy files
    
    execution time: 5 x 170 sec
    
    total capacity: 5 x 3.3 GB
    
    image size:     (96,96)
    
 - extractFace.py
 
    Produce specified -from extractionmaps- face-images on .jpg files
    
    execution time: --
    
    total capacity: --
    
    image size:     (96,96)
