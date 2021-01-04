import warnings

warnings.filterwarnings("ignore")

import configparser
import sys
from model_build import concatenating, loss_func
from Create_data import Create_data #need to add codes to merge image in train.py
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

print('=================Config====================')

#read config files
config = configparser.ConfigParser()
config.read('./Config.config')

model_info = config['Model_INFO']
data_info = config['Data_INFO']

#model_info
model_num = int(model_info['model_num'])
optimizer = model_info['optimizer']
epoch = int(model_info['epoch'])
batch_size = int(model_info['batch_size'])
loss = model_info['loss']
metrics=model_info['metrics']
print('1. Model')
print(' model num : ' + str(model_num))
print(' optimizer : ' + str(optimizer))
print(' epochs : ' +str(epoch))
print(' batch size : ' +str(batch_size))
print(' loss function : ' +str(loss))
print('===========================================')
#data info
width = int(data_info['img_width'])
height = int(data_info['img_height'])
depth = int(data_info['img_depth'])
train_dir = data_info['img_dir']
ev_model = config['Evaluation']
model_path = ev_model['model_path']
test_data_path = ev_model['test_data_path']

print('2. Data')
print(' width : ' + str(width))
print(' height : ' +str(height))
print(' depth : ' +str(depth))
print('===========================================')

#load model
load_model = config['Load_MODEL']
ckpt_load_path = load_model['model_path']
ckpt_save_path = load_model['checkpoint_path']

#read data
ch1, ch2, ch3, ch4, coord = Create_data.read_img(train_dir, depth)
ch1_t, ch2_t, ch3_t, ch4_t, coord_t = Create_data.read_img(test_data_path, depth)
if (ch1[0]==ch2[0]).all() :
	print('error')
else :
	print('continue')

#check point format
ckpt_path = ckpt_save_path + "/model-{epoch:06d}.ckpt"
ckpt_dir = os.path.dirname(ckpt_path)

#Build the model
model = concatenating.multi_input
lf = loss_func.loss_function
md = model.build(width, height, depth, model_num)
if loss == 'custom' :
    loss_f = lf.normalized_mse
else :
    loss_f = loss
md.compile(optimizer=optimizer, loss=loss_f)

#check point callback
if ckpt_load_path == 'Empty' :
    md.save_weights(ckpt_path.format(epoch=0))
    global_epoch = epoch
else :
    print("=================================================")
    print("loading.....")
    global_epoch = int(ckpt_load_path[-11:-5]) + epoch
    md.load_weights(ckpt_load_path)
    
    
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path, verbose = 1, save_weights_only = True, period = 5000)
md.fit([ch1, ch2, ch3, ch4], coord, epochs = epoch, callbacks = [ckpt_callback], batch_size = batch_size, validation_data = ([ch1_t, ch2_t, ch3_t, ch4_t], coord_t))


md.save_weights(ckpt_path.format(epoch=global_epoch))
md.save('./trained_model/md' + str(model_num) + '_optimizer' +str(optimizer) + '_loss' + str(loss) + '_batch' + str(batch_size) + '_epoch' + str(epoch) + '.h5')
train_pred = md.predict([ch1, ch2, ch3, ch4])

print("=================Prediction====================")
print(md.predict([ch1, ch2, ch3, ch4]))
print("===================Answer======================")
print(coord)
print('Train Loss', md.evaluate([ch1, ch2, ch3, ch4], coord))

plt.figure(1)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.scatter(coord[:,0], coord[:,1], c = 'blue', label = 'Answer')
plt.scatter(train_pred[:, 0], train_pred[:, 1], c = 'red', label = 'Model Prediction')
plt.savefig('./train_result_' + str(model_num) + '.jpg')

test_pred = md.predict([ch1_t, ch2_t, ch3_t, ch4_t])
plt.figure(2)
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.scatter(coord_t[:,0], coord_t[:,1], c = 'blue', label = 'Answer')
plt.scatter(test_pred[:, 0], test_pred[:, 1], c = 'red', label = 'Model Prediction')
plt.savefig('./test_result_' + str(model_num) + '.jpg')
