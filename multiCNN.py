
########################### LIBRARIES ###########################

import numpy as np
import tensorflow as tf
import pandas as pd
import time

########################## PARAMETERS ###########################

t0 = time.time()    # Starting time
dense_units = 2048  # Dense layers' units
dropout_rate = 0  # Dropout rate: percentage of neurons to drop
n_classes = 8       # Output's units
eta = 0.001         # Learning rate
n_epochs = 1        # Number of epochs
batches_size = 100  # Size of mini-batches
img_height = 96     # Height of spectrograms
img_width = 96      # Width of spectrograms

########################### CNN MODEL ###########################

tf.logging.set_verbosity(tf.logging.INFO)

def multi_cnn_model(features, labels, mode):


    print(features['faces'])
    print(features['spectrograms'])
    print(labels)
    ### FACE SIDE ###
    # First input layer of multi-CNN
    input_layer_1 = tf.reshape( features['faces'], [-1, 96, 96, 3])
    # Convolutional layer #1.1
    conv1_1 = tf.layers.conv2d( inputs=input_layer_1, filters=32, kernel_size=[3,3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=tf.nn.relu )
    # Batch-normalization layer 1.1
    batch_norm1_1 = tf.layers.batch_normalization( inputs=conv1_1, training=False )
    # Maxpooling layer #1.1
    pool1_1 = tf.layers.max_pooling2d( inputs=batch_norm1_1, pool_size=[2,2], strides=2 )
    # Convolutional layer #1.2
    conv1_2 = tf.layers.conv2d( inputs=pool1_1, filters=64, kernel_size=[3,3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=tf.nn.relu )
    # Batch-normalization layer 1.2
    batch_norm1_2 = tf.layers.batch_normalization( inputs=conv1_2, training=False )
    # Maxpooling layer #1.2
    pool1_2 = tf.layers.max_pooling2d( inputs=batch_norm1_2, pool_size=[2,2], strides=2 )
    # Convolutional layer #1.3
    conv1_3 = tf.layers.conv2d( inputs=pool1_2, filters=128, kernel_size=[3,3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=tf.nn.relu )
    # Batch-normalization layer 1.3
    batch_norm1_3 = tf.layers.batch_normalization( inputs=conv1_3, training=False )
    # Maxpooling layer #1.3
    pool1_3 = tf.layers.max_pooling2d( inputs=batch_norm1_3, pool_size=[2,2], strides=2 )
    # Convolutional layer #1.4
    conv1_4 = tf.layers.conv2d( inputs=pool1_3, filters=256, kernel_size=[3,3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=tf.nn.relu )
    # Batch-normalization layer 1.4
    batch_norm1_4 = tf.layers.batch_normalization( inputs=conv1_4, training=False )
    # Maxpooling layer #1.4
    pool1_4 = tf.layers.max_pooling2d( inputs=batch_norm1_4, pool_size=[2,2], strides=2 )
    
    ### SPECTROGRAM SIDE ###
    # Second input layer of multi-CNN
    input_layer_2 = tf.reshape( features['spectrograms'], [-1, 96, 96, 1])
    # Convolutional layer #2.1
    conv2_1 = tf.layers.conv2d( inputs=input_layer_2, filters=32, kernel_size=[3,3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=tf.nn.relu )
    # Batch-normalization layer 2.1
    batch_norm2_1 = tf.layers.batch_normalization( inputs=conv2_1, training=False )
    # Maxpooling layer #2.1
    pool2_1 = tf.layers.max_pooling2d( inputs=batch_norm2_1, pool_size=[2,2], strides=2 )
    # Convolutional layer #2.2
    conv2_2 = tf.layers.conv2d( inputs=pool2_1, filters=64, kernel_size=[3,3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=tf.nn.relu )
    # Batch-normalization layer 2.2
    batch_norm2_2 = tf.layers.batch_normalization( inputs=conv2_2, training=False )
    # Maxpooling layer #2.2
    pool2_2 = tf.layers.max_pooling2d( inputs=batch_norm2_2, pool_size=[2,2], strides=2 )
    # Convolutional layer #2.3
    conv2_3 = tf.layers.conv2d( inputs=pool2_2, filters=128, kernel_size=[3,3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=tf.nn.relu )
    # Batch-normalization layer 2.3
    batch_norm2_3 = tf.layers.batch_normalization( inputs=conv2_3, training=False )
    # Maxpooling layer #2.3
    pool2_3 = tf.layers.max_pooling2d( inputs=batch_norm2_3, pool_size=[2,2], strides=2 )
    # Convolutional layer #2.4
    conv2_4 = tf.layers.conv2d( inputs=pool2_3, filters=256, kernel_size=[3,3], kernel_initializer=tf.contrib.layers.xavier_initializer(), padding='same', activation=tf.nn.relu )
    # Batch-normalization layer 2.4
    batch_norm2_4 = tf.layers.batch_normalization( inputs=conv2_4, training=False )
    # Maxpooling layer #2.4
    pool2_4 = tf.layers.max_pooling2d( inputs=batch_norm2_4, pool_size=[2,2], strides=2 )

    ### CONCATENATE BOTH SIDES ###
    pool1_4_flat = tf.reshape( pool1_4, [-1, 6*6*256] )
    pool2_4_flat = tf.reshape( pool2_4, [-1, 6*6*256] )
    total_pool_flat = tf.concat( [pool1_4_flat, pool2_4_flat], axis=1 )

    ### DENSE LAYERS ###
    dense_1 = tf.layers.dense( inputs=total_pool_flat, units=dense_units, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu )
    batch_norm_1 = tf.layers.batch_normalization( inputs=dense_1, training=False )
    dropout_1 = tf.layers.dropout( inputs=batch_norm_1, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN )
    dense_2 = tf.layers.dense( inputs=dropout_1, units=dense_units, kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu )
    batch_norm_2 = tf.layers.batch_normalization( inputs=dense_2, training=False )
    dropout_2 = tf.layers.dropout( inputs=batch_norm_2, rate=dropout_rate, training=mode == tf.estimator.ModeKeys.TRAIN )

    ### LOGITS LAYER ###
    logits = tf.layers.dense( inputs=dropout_2, units=n_classes )

    # Generate predictions for PREDICT and EVAL modes
    predictions = {
        'classes': tf.argmax( input=logits, axis=1 ),
        # Add 'softmax_tensor' to the graph, used for PREDICT mode
        'propabilities': tf.nn.softmax( logits, name='softmax_tensor' )
    }

    # Configure the Prediction Op for PREDICT mode 
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec( mode=mode, predictions=predictions)

    # Calculate loss for TRAIN and EVAL modes
    print(logits)
    onehot_labels = tf.one_hot( indices=tf.cast(labels, tf.int32), depth=n_classes )
    print(onehot_labels)
    loss = tf.losses.softmax_cross_entropy( onehot_labels=onehot_labels, logits=logits )

    # Configure the Training mode for TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer( learning_rate=eta )
        train_op = optimizer.minimize( loss=loss, global_step=tf.train.get_global_step() )
        return tf.estimator.EstimatorSpec( mode=mode, loss=loss, train_op=train_op )

    # Add ealuation metrics for EVAL mode
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy( labels=labels, predictions=predictions['classes'] )
    }
    return tf.estimator.EstimatorSpec( mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


########################### FUNCTIONS ###########################


def _parse_function( facePATH, spectrogramPATH, label):
    face_string = tf.read_file(facePATH)
    face_decoded = tf.image.decode_jpeg(face_string, channels=3)
    face_decoded = tf.to_float(face_decoded)
    face_decoded = tf.reshape(face_decoded,[img_height,img_width,3])
    face_decoded = tf.image.per_image_standardization(face_decoded)
    
    spectrogram_string = tf.read_file(spectrogramPATH)
    spectrogram_decoded = tf.decode_raw(spectrogram_string,tf.uint16)
    spectrogram_decoded = spectrogram_decoded[64:]
    spectrogram_decoded = tf.to_float(spectrogram_decoded)
    spectrogram_decoded = tf.divide(spectrogram_decoded, 10000)
    spectrogram_decoded = tf.reshape(spectrogram_decoded,[img_height,img_width,1])
    
    #label = tf.to_float(label)
    print(face_decoded)
    print(spectrogram_decoded)
    return face_decoded, spectrogram_decoded, label


def train_input_fn():

    # Load train data
    df = pd.read_csv('/home/gryphonlab/Ioannis/Works/IEMOCAP/Core/training_data.csv')
    # Faces for training (numpy array)
    train_faces = np.array(df['train_faces'])
    # Spectrograms for training (numpy array)
    train_spectrograms = np.array(df['train_spectrograms'])
    train_labels = np.array(df['train_labels']).astype(np.int32)

    #faces_placeholder = tf.placeholder(train_faces.dtype, train_faces.shape)
    #spectrograms_placeholder = tf.placeholder(train_spectrograms.dtype, train_spectrograms.shape)
    #labels_placeholder = tf.placeholder(train_labels.dtype, train_labels.shape)
    
    training_dataset = tf.data.Dataset.from_tensor_slices((train_faces, train_spectrograms, train_labels))
    training_dataset = training_dataset.map(_parse_function)
    batched_dataset = training_dataset.batch(batches_size)
    
    iterator = batched_dataset.make_one_shot_iterator()
    faces, spectrograms, labels = iterator.get_next()
    #training_init_op = iterator.make_initializer(training_dataset)

    features_dict = { 'faces': faces, 'spectrograms': spectrograms}

    return features_dict, labels


############################# MAIN #############################


def main(unused_argv): 
    
    
    # Create the Estimator
    multi_cnn_classifier = tf.estimator.Estimator( model_fn=multi_cnn_model, model_dir='/home/gryphonlab/Ioannis/Works/IEMOCAP/Core/multi_cnn_eta.001' )
    # Set up logging for predictions
    tensors_to_log = { 'probabilities': 'softmax_tensor' }
    logging_hook = tf.train.LoggingTensorHook( tensors=tensors_to_log, every_n_iter=50 )
    
    # TRAIN the model
    #train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={'x1': train_faces, 'x2': train_spectrograms}, y=train_labels, batch_size=batches_size, num_epochs=1, shuffle=True
    #)
    multi_cnn_classifier.train( input_fn=train_input_fn, steps=20000, hooks=[logging_hook])

    # EVAL the model
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x={'x1': eval_faces, 'x2': eval_spectrograms}, y=eval_labels, num_epochs=1, shuffle=False
    #)
    #eval_results = multi_cnn_classifier.evaluate( input_fn=eval_input_fn )
    #print(eval_results)

    print('Execution time: '+ str(time.time() - t0))



if __name__ == '__main__':
    tf.app.run()