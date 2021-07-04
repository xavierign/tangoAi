import os
import pickle
import numpy
from music21 import note, chord, instrument, stream, duration
import time

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from models.RNNAttention import get_distinct, create_lookups, prepare_sequences, get_music_list, create_network
from models.RNNAttention import create_network, sample_with_temp
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

#load data and mappings

melody_lengths = [12,16,24]
data_dir = 'data/'

notes_in_dict = {}
notes_out_dict = {}
durations_in_dict = {}
durations_out_dict = {}
for l in melody_lengths:
    notes_in_dict[l] = np.load(data_dir + 'SEQ_' +str(l) + '/notes_in_' + str(l) + '.npy')
    notes_out_dict[l] = np.load(data_dir + 'SEQ_' +str(l) + '/notes_out_' + str(l) + '.npy')
    durations_in_dict[l] = np.load(data_dir + 'SEQ_' +str(l) +'/durations_in_' + str(l) + '.npy')
    durations_out_dict[l] = np.load(data_dir + 'SEQ_' +str(l)  + '/durations_out_' + str(l) + '.npy')
    
n_durations = 9
n_notes = 42

import os
import pickle
import numpy
from music21 import note, chord, instrument, stream, duration
import time

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from models.RNNAttention import get_distinct, create_lookups, prepare_sequences, get_music_list, create_network
from models.RNNAttention import create_network, sample_with_temp
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

#define the grid
melody_lengths = [12, 16, 24] #
use_attention = True # cambiar a False

embed_size_array = [24,36,48]
rnn_units_array = [32,64,128]

# running parameters
additional_epochs = 4

## loop
for melody_length in melody_lengths:
    
    #load data
    notes_in = notes_in_dict[melody_length]
    notes_out = notes_out_dict[melody_length]
    durations_in = durations_in_dict[melody_length]
    durations_out = durations_out_dict[melody_length]
    
    network_input = [notes_in, durations_in]
    network_output = [notes_out, durations_out]

    
    # loop parameters
    for embed_size in embed_size_array:
        for rnn_units in rnn_units_array:
            run_folder = 'run/{}_{}_{}_{}/'.format(melody_length,embed_size,rnn_units,use_attention)
            store_folder = os.path.join(run_folder, 'store')
            weights_folder = os.path.join(run_folder, 'weights')
            
            #define checkpoints
            checkpoint2 = ModelCheckpoint(os.path.join(weights_folder, "weights.h5"),
                            monitor='val_loss',
                            verbose=0,
                            save_best_only=True,
                            mode='min')
            early_stopping = EarlyStopping(monitor='val_loss'
                                , restore_best_weights=True
                                , patience = 5 )
            callbacks_list = [checkpoint2, early_stopping]
            
            #create directories
            if not os.path.exists(run_folder):
                os.mkdir(run_folder)
                os.mkdir(os.path.join(run_folder, 'store'))
                os.mkdir(os.path.join(run_folder, 'output'))
                os.mkdir(os.path.join(run_folder, 'weights'))
                os.mkdir(os.path.join(run_folder, 'viz'))

            #train model to capture validation and n_epochs
            model, att_model = create_network(n_notes, n_durations, embed_size, rnn_units, use_attention)
            trainHistory = model.fit(network_input, network_output
                  , epochs=1000, batch_size=128
                  , validation_split = 0.2
                  , callbacks=callbacks_list
                  , shuffle=True)

            #retrain with all data n_epochs from above 
            model, att_model = create_network(n_notes, n_durations, embed_size, rnn_units, use_attention)
            fullTrainHistory = model.fit(network_input, network_output
                  , epochs=early_stopping.stopped_epoch + additional_epochs, batch_size=128
                  , validation_split = 0
                  , shuffle=False)
            
            #save fullTrained model
            model.save(store_folder + '/model.h5')

            #save training history
            with open(store_folder + '/trainHistory.json', mode='w') as f:
                pd.DataFrame(trainHistory.history).to_json(f)

            with open(store_folder + '/fullTrainHistory.json', mode='w') as f:
                pd.DataFrame(fullTrainHistory.history).to_json(f)

            #write the parameters
            text_file = open(store_folder + "/parameters.txt", "w")
            text_file.write(str(embed_size)+","+str(rnn_units)+","+str(use_attention)+ "," + str(len(notes_in[0])) +"\n")
            text_file.close()