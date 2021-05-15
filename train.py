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

melody_len = 16 # 12, 16, or 24
section = 'SEQ_' + str(melody_len)   
indir = 'data/' + section + "/"

notes_in = np.load(indir + 'notes_in.npy')
notes_out = np.load(indir + 'notes_out.npy')
durations_in = np.load(indir + 'durations_in.npy')
durations_out = np.load(indir + 'durations_out.npy')

network_input = [notes_in, durations_in]
network_output = [notes_out, durations_out]

# run params


### fixed by data
n_notes = 42
n_durations = 9 

#run experiments
n_experiments = 4
music_name = 'experiments'
additional_epochs= 4
n_initial = 1

for id_ in range(n_experiments):
    run_id = str(id_)
    run_folder = 'run/{}/'.format(section)
    run_folder += str(int(random.random()*999999999999))#'_'.join([run_id, music_name])
    store_folder = os.path.join(run_folder, 'store')
    data_folder = os.path.join('data', music_name)
    weights_folder = os.path.join(run_folder, 'weights')

    checkpoint1 = ModelCheckpoint(
        os.path.join(weights_folder, "weights-improvement-{epoch:02d}-{val_loss:.4f}-bigger.h5"),
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min')
    checkpoint2 = ModelCheckpoint(
        os.path.join(weights_folder, "weights.h5"),
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min')
    early_stopping = EarlyStopping(
        monitor='val_loss'
        , restore_best_weights=True
        , patience = 3 )
    callbacks_list = [
        #checkpoint1
        checkpoint2
        , early_stopping]

    if not os.path.exists(run_folder):
        os.mkdir(run_folder)
        os.mkdir(os.path.join(run_folder, 'store'))
        os.mkdir(os.path.join(run_folder, 'output'))
        os.mkdir(os.path.join(run_folder, 'weights'))
        os.mkdir(os.path.join(run_folder, 'viz'))

    #variable
    embed_size =random.choice([36,48,60])
    rnn_units = random.choice([50,100,200])
    use_attention = random.choice([True,False])
    model, att_model = create_network(n_notes, n_durations, embed_size, rnn_units, use_attention)
    trainHistory = model.fit(network_input, network_output
          , epochs=500, batch_size=128
          , validation_split = 0.2
          , callbacks=callbacks_list
          , shuffle=False   )
    
    #retrain with all data 
    model, att_model = create_network(n_notes, n_durations, embed_size, rnn_units, use_attention)
    fullTrainHistory = model.fit(network_input, network_output
          , epochs=early_stopping.stopped_epoch + additional_epochs, batch_size=64
          , validation_split = 0
          , shuffle=False  )
    
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