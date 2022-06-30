#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Display some predictions."""
import os
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
from nmp import dataset
# from tensorflow.keras.layers import LSTM
import tensorflow as tf
import copy
import pandas as pd
from nmp.dataset import pyplot_piano_roll
from nmp import model as mod
import matplotlib.pyplot as plt
import time
import pypianoroll
import random

P = Path(os.path.abspath(''))  # Compatible with Jupyter Notebook

PLOTS = P / 'plots'  # Plots path
BS = 64
FS = 24  # Sampling frequency. 10 Hz = 100 ms
Q = 0  # Quantize?
st = 10  # Past timesteps
num_ts = 10  # Predicted timesteps
DOWN = 12  # Downsampling factor
D = "data/POP909" # Dataset

MODEL = 'lstm-z-de.h5'

LOW_LIM = 33  # A1
HIGH_LIM = 97  # C7

# LOW_LIM = 36  # A1
# HIGH_LIM = 85  # C7

NUM_NOTES = HIGH_LIM - LOW_LIM
CROP = [LOW_LIM, HIGH_LIM]  # Crop plots

# TensorFlow stuff
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE = 0
hm = 4

# Get midi file 
# import argparse

# parser = argparse.ArgumentParser()

# parser.add_argument("input_file", help="input Midi file for predictions")
# args = parser.parse_args()
# FILE = args.input_file
FILE = "twinkle-twinkle"
# FILE = "ShapeofYou"
# FILE = "twinkle-cut"
# FILE = "mary_had_lamb"
# FILE = "photograph"

pr = pypianoroll.parse("midi_tests/"+str(FILE)+".mid", 24)
tempo_arr = pd.DataFrame(pr.tempo)
tempo = tempo_arr[0][1]
print("tempo:", tempo)
# print(pd.DataFrame(resolution.tempo))

merged = pr.get_merged_pianoroll()
test_file = np.savetxt(FILE, merged)
test_file = np.save(FILE, merged)

# print(arr.shape)
