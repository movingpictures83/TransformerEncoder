#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

# Step 1: Get current directory 
current_directory = os.getcwd()        # '/home/bear-b/users/jshi008/IvyProjects/FIDLA/train_WaLeF_models'

# Step 2: Get path of parent directory (one level up)
parent_directory = os.path.dirname(current_directory)    # '/home/bear-b/users/jshi008/IvyProjects/FIDLA'

# Step 3: Append parent directory to sys.path
sys.path.append(parent_directory)


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from tensorflow import keras
from tensorflow.keras import Model, Input, layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *




# In[3]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ### Preprocess

# In[4]:
class TransformerEncoderPlugin:
 def input(self, inputfile):
     self.inputfile = inputfile
 def run(self):
     pass
 def output(self, outputfile):
  from preprocess.BaselinePrerocess import baseline_process
  from baselines.transformer import transformer_encoder
  from sklearn.metrics import mean_squared_error as mse
  from sklearn.metrics import mean_absolute_error as mae
  from tensorflow.keras.models import load_model
  from math import sqrt
  n_hours = 72
  K = 24 
  masked_value = 1e-10
  split_1 = 0.8
  split_2 = 0.9
  train_X_mask, val_X_mask, test_X_mask, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler = baseline_process(n_hours, K, masked_value, split_1, split_2, self.inputfile)
  print(train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, train_ws_y.shape, val_ws_y.shape, test_ws_y.shape)
  input_shape = train_X_mask.shape[1:]
  head_size = 64*2
  ff_dim = 64
  num_transformer_blocks = 1
  num_heads = 1
  mlp_units = 32
  dropout = 0.4
  mlp_dropout = 0.3
  masked_value = 1e-10
  learning_rate = 5e-4
  decay_steps = 10000
  decay_rate = 0.95
  PATIENCE = 500
  EPOCHS = 5
  BATCH = 512
  model = transformer_encoder(input_shape,
                            head_size=head_size, 
                            num_heads=num_heads, 
                            ff_dim=ff_dim, 
                            num_transformer_blocks=num_transformer_blocks, 
                            mlp_units=mlp_units, 
                            dropout=dropout, 
                            mlp_dropout=mlp_dropout,
                            masked_value=masked_value
                           )
  model.summary()
  lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=learning_rate, 
                                                          decay_steps=decay_steps,
                                                          decay_rate=decay_rate)
  model.compile(optimizer=Adam(learning_rate=lr_schedule),
              loss='mse',
              metrics=['mae']
             )
  es = EarlyStopping(monitor='val_mae', mode='min', verbose=2, patience=PATIENCE)
  mc = ModelCheckpoint(outputfile, 
                     monitor='val_mae', 
                     mode='min', 
                     verbose=2, 
                     save_best_only=True,
                     
                    )
  model.fit(train_X_mask, train_ws_y,
          validation_data=(test_X_mask, test_ws_y),
          epochs=EPOCHS,
          batch_size=BATCH,
          verbose=2,
          shuffle=True,
          callbacks=[es, mc]
         )
  saved_model = load_model(outputfile)
  yhat = saved_model.predict(test_X_mask)

  inv_yhat = ws_scaler.inverse_transform(yhat)
  inv_y = ws_scaler.inverse_transform(test_ws_y)

  print("inv_y.shape, inv_yhat.shape", inv_y.shape, inv_yhat.shape)
  print('MAE = {}'.format(float("{:.4f}".format(mae(inv_y, inv_yhat)))))
  print('RMSE = {}'.format(float("{:.4f}".format(sqrt(mse(inv_y, inv_yhat))))))
  wss = ['WS_S1', 'TWS_S25A', 'TWS_S25B', 'TWS_S26']
  inv_yhat_reshape = inv_yhat.reshape((-1, 24, 4))
  inv_y_reshape = inv_y.reshape((-1, 24, 4))

  for i in range(len(wss)):
    plt.rcParams["figure.figsize"] = (15, 3)
    error = inv_yhat_reshape - inv_y_reshape
    plt.plot(error[:, -1, i], linewidth=1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Time steps', fontsize=18)
    plt.ylabel('Actual Error (ft)', fontsize=18)
    plt.title('{}'.format(wss[i]), fontsize=18)
    plt.text(9000, -0.6, 'MAE = {}'.format(float("{:.3f}".format(sum(abs(error[:, -1, i]))/len(error)))), fontsize=12)
    plt.show()
  up_thre = 0.5
  low_thre = -0.5

  for i in range(len(wss)):
    plt.rcParams["figure.figsize"] = (15, 3)
    error = inv_yhat_reshape - inv_y_reshape
    error_24h = error[:, -1, i]
    print(error_24h.shape)
    print(np.sum(error[:, -1, i] > up_thre) + np.sum(error[:, -1, i] < low_thre))
  for i in range(len(wss)):
    plt.rcParams["figure.figsize"] = (15, 3)
    error = inv_yhat_reshape - inv_y_reshape
    error_24h = error[:, -1, i]
    print(error_24h.shape)
    print(np.sum(error[:, -1, i] > up_thre))
  for i in range(len(wss)):
    plt.rcParams["figure.figsize"] = (15, 3)
    error = inv_yhat_reshape - inv_y_reshape
    error_24h = error[:, -1, i]
    print(error_24h.shape)
    print(np.sum(error[:, -1, i] < low_thre))

  from preprocess.BaselinePrerocess import baseline_process, gcn_process, baseline_process_for_gate_predictor
  from preprocess.GraphTransformerPrerocess import graph_water_transformer_cov_process
  from preprocess.GraphTransformerPrerocess import graph_global_transformer_local_process
  from preprocess.GraphTransformerPrerocess import graph_water_transformer_cov_process_for_gate_predictor
  from preprocess.graph import graph_topology, graph_topology_5
  from tensorflow.keras.models import load_model
  from postprocess.threshold import flood_threshold, drought_threshold, flood_threshold_t1, drought_threshold_t1
  from postprocess.errors import estimate_error
  from sklearn.metrics import mean_squared_error as mse
  from sklearn.metrics import mean_absolute_error as mae
  from math import sqrt
  from spektral.layers import GCNConv
  from baselines.tcn import TCN
  from preprocess.helper import series_to_supervised
  import time

  n_hours = 72
  K = 24
  masked_value = 1e-10
  split_1 = 0.8
  split_2 = 0.9
  sigma2 = 0.1
  epsilon = 0.5

  train_X_mask, val_X_mask, test_X_mask, train_ws_y, val_ws_y, test_ws_y, scaler, ws_scaler = baseline_process(n_hours, K, masked_value, split_1, split_2, self.inputfile)

  print(train_X_mask.shape, val_X_mask.shape, test_X_mask.shape, train_ws_y.shape, val_ws_y.shape, test_ws_y.shape)
  #saved_models = ['WaLeF_transformer']

  #for i in range(len(saved_models)):
  model = load_model(outputfile)
  start_time = time.perf_counter()
  yhat = model.predict(test_X_mask)
  end_time = time.perf_counter()
  used_time = end_time - start_time
  print(f"Usded time: {used_time} seconds")
  # inverse transformation
  inv_yhat = ws_scaler.inverse_transform(yhat)
  inv_yhat = inv_yhat[:, [0, 24, 48, 72]]
  inv_y = ws_scaler.inverse_transform(test_ws_y)
  inv_y = inv_y[:, [0, 24, 48, 72]]

  print(inv_y.shape)
  # compute MAE and RMSE
  print('MAE = {}'.format(float("{:.4f}".format(mae(inv_y, inv_yhat)))))
  print('RMSE = {}'.format(float("{:.4f}".format(sqrt(mse(inv_y, inv_yhat))))))

  errors = inv_yhat - inv_y
  print('Numbers of over/under estimate:', estimate_error(errors))


