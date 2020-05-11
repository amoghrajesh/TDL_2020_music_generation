import sys
import re 
import numpy as np 
import pandas as pd
import music21
from glob import glob
import IPython
from tqdm import tqdm
import pickle
from keras.utils import np_utils
from music21 import converter, instrument, note, chord, stream
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, LSTM, Dropout, Flatten

songs = glob('../PES1201701295_data/Jazz/*.mid')
songs = songs[:3]

def get_notes():
    notes = []
    for file in songs:
        midi = converter.parse(file)
        notes_to_parse = []
        try:
            parts = instrument.partitionByInstrument(midi)
        except:
            pass
        if parts: 
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = midi.flat.notes
    
        for element in notes_to_parse: 
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif(isinstance(element, chord.Chord)):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    with open('../data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    
    return notes

def prepare_sequences(notes, n_vocab): 
    sequence_length = 100
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = np_utils.to_categorical(network_output)
    print("n_patterns----------------->",n_patterns, network_input.shape)
    return (network_input, network_output)

def create_network(network_in, n_vocab): 
    model = Sequential()
    #print("---------------------------------input shape = ------------------------", network_in.shape[1:])
    model.add(LSTM(128, input_shape=network_in.shape[1:], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())

    return model

def train(model, network_input, network_output, epochs): 
    filepath = 'weights.best.music3.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True)
    
    model.fit(network_input, network_output, epochs=epochs, batch_size=32, callbacks=[checkpoint])

def train_network():

    epochs = 10
    notes = get_notes()
    print('Notes processed')
    n_vocab = len(set(notes))
    print('Vocab generated')
    network_in, network_out = prepare_sequences(notes, n_vocab)
    print('Input and Output processed')
    print("------------------lentth of network in and out--------------", len(network_in) ,len(network_out))
    model = create_network(network_in, n_vocab)
    print('Model created')
    print('Training in progress')
    train(model, network_in, network_out, epochs)
    print('Training completed')

def generate():
    with open('../data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)
    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))
    
    print('Initiating music generation process.......')
    
    network_input = get_inputSequences(notes, pitchnames, n_vocab)
    normalized_input = network_input / float(n_vocab)
    model = create_network(normalized_input, n_vocab)
    print('Loading Model weights.....')
    model.load_weights('weights.best.music3.hdf5')
    print('Model Loaded')
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def get_inputSequences(notes, pitchnames, n_vocab):
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    sequence_length = 100
    network_input = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
    
    network_input = np.reshape(network_input, (len(network_input), 100, 1))
    
    return (network_input)

def generate_notes(model, network_input, pitchnames, n_vocab):
    start = np.random.randint(0, len(network_input)-1)
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    pattern = list(network_input[start])
    prediction_output = []
    
    print('Generating notes........')
    for note_index in range(500):
        if note_index%100==0:
            print("in for loop")
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print('Notes Generated...')
    return prediction_output

def create_midi(prediction_output):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    
    print('Saving Output file as midi....')

    midi_stream.write('midi', fp='test_output.mid')
generate()
