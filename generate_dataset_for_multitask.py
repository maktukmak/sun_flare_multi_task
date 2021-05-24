import numpy as np
import matplotlib.pyplot as plt
import pickle
from dataset_prep import dataset_struct
import tensorflow as tf

import seaborn as sns
sns.set_theme()

tf.autograph.set_verbosity(0)

tf.config.experimental_run_functions_eagerly(False)

class dataset_cfg_struct():
    def __init__(self):
        self.use_cached_preprocessing = False
        self.use_cached_harps = False
        
        self.len_seq = 12 * 5 * 10 # 10 hours of features
        self.len_pred = 24 * 60 # prediction for 24 hours after
        self.max_flare_filtering = True  # Filter flares within 24 hours window 
        self.max_flare_window_drop = False # If true: Drop filtered flares else: replace with max flare
        self.remove_C = True


dataset_cfg = dataset_cfg_struct()

dataset = dataset_struct(dataset_cfg)
dataset.load_datasets()
dataset.preprocess_datasets()

x_h = []
y_h = []
for i in range(len(dataset.dataset_hmi.valid_events_train)):
    feats = dataset.read_features(dataset.dataset_hmi.valid_events_train[i], dataset.dataset_hmi)[0]
    y_h.append(dataset.dataset_hmi.valid_events_train[i][1])
    x_h.append(feats[-1])
    
x_h = np.array(x_h)   
y_h = [0 if (y_h[i][:1] == 'B') or (y_h[i][:1] == 'C') else 1 for i in range(len(y_h))]
y_h = np.array(y_h)

x_m = []
y_m = []
for i in range(len(dataset.dataset_mdi.valid_events_train)):
    feats = dataset.read_features(dataset.dataset_mdi.valid_events_train[i], dataset.dataset_mdi)[0]
    y_m.append(dataset.dataset_mdi.valid_events_train[i][1])
    x_m.append(feats[-1])
    
x_m = np.array(x_m)   
y_m = [0 if (y_m[i][:1] == 'B') or (y_m[i][:1] == 'C') else 1 for i in range(len(y_m))]
y_m = np.array(y_m)


x_te = []
y_te = []
for i in range(len(dataset.dataset_hmi.valid_events_test)):
    feats = dataset.read_features(dataset.dataset_hmi.valid_events_test[i], dataset.dataset_hmi)[0]
    y_te.append(dataset.dataset_hmi.valid_events_test[i][1])
    x_te.append(feats[-1])
    
x_te= np.array(x_te)   
y_te = [0 if (y_te[i][:1] == 'B') or (y_te[i][:1] == 'C') else 1 for i in range(len(y_te))]
y_te = np.array(y_te)

with open("data_for_multitask.txt", "wb") as fp:
    pickle.dump([x_m, y_m, x_h, y_h, x_te, y_te], fp)

