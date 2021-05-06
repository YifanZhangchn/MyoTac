#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
from collections import deque
from threading import Lock, Thread

from math import atan2,asin,sqrt
import time
import myo
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()

class EmgCollector(myo.DeviceListener):
     #Collects EMG data in a queue with *n* maximum number of elements.

    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)
        self.imu_data_queue = deque(maxlen=(int(n)//4+1))

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)
    
    def get_imu_data(self):
        with self.lock:
            return list(self.imu_data_queue)

  # myo.DeviceListener

    def on_connected(self, event):
        event.device.stream_emg(True)
        print("\r connected！！")
        event.device.request_battery_level()

    def on_battery_level(self, event):
        print("Your battery level is:", event.battery_level)

    def on_emg(self, event):
        with self.lock:
            self.emg_data_queue.append((event.emg))
            
    
    def on_orientation(self,event):
        with self.lock:
            rotation = event.orientation
            roll = atan2(2.0 * (rotation[3] * rotation[0] + rotation[1] * rotation[2]),1.0-2.0 * (rotation[0] * rotation[0] + rotation[1] * rotation[1]))
            pitch = asin(max(-1.0, min(1.0, 2.0 * (rotation[3] * rotation[1] - rotation[2] * rotation[0]))))
            yaw = atan2(2.0 * (rotation[3] * rotation[2] + rotation[0] * rotation[1]),
                        1.0-2.0 * (rotation[1] * rotation[1] + rotation[2] * rotation[2]))

            self.imu_data_queue.append([event.acceleration[0],event.acceleration[1],event.acceleration[2],
                                        event.gyroscope[0],event.gyroscope[1],event.gyroscope[2],roll,pitch,yaw])
                    


class Analysis(object):

    def __init__(self, listener,sess,y,preValue,x,keep_prob):
        self.n = listener.n
        self.listener = listener
        self.sess = sess
        self.preValue = preValue
        self.y = y
        self.x = x
        self.keep_prob = keep_prob
        self.imu = None
        self.emg = None
        self.gesture = ['男性','女性','指挥官','人质','嫌疑犯','你','我','来','听到','看到','推进',
                        '讯息收到','赶快','停止','掩护我','不明白','明白','蹲下','不用理会','手枪',
                        '步枪','自动武器','霰弹枪','汽车','门口','转角处','集合','单纵队','双向纵队','单向横队']


    def update_analysis(self):
        imu_data = np.array(self.listener.get_imu_data())
        emg_data = np.array(self.listener.get_emg_data())
        if imu_data.shape[0]!= self.n//4+1 or emg_data.shape[0]!=self.n:
            print("not full！！")
            return
        gyro_z = [x[5] for x in imu_data]
        gyro_y = [x[4] for x in imu_data]
        if (gyro_z[12]>30 and gyro_z[13]>60 and gyro_z[16]>100) or (abs(gyro_y[12])>30 and abs(gyro_y[13])>60 and abs(gyro_y[15])>100):
            self.imu = imu_data.T
            self.emg = emg_data.T
            print("---------------------")
            print("start：")
            time_start = time.time()
            self.data_processing()
            time_end = time.time()
            print('time：',time_end-time_start)
            time.sleep(2)
    
    def data_processing(self):
        tx = np.linspace(0,1,101)
        xnew = np.linspace(0,1,400)
        data_list = self.emg.tolist()
        for data in self.imu.tolist():
            ynew = np.interp(xnew,tx,data)
            data_list.append(ynew)
        
        #print(self.imu)
        
        data_list = np.expand_dims(data_list,axis=0)
        data_list = np.expand_dims(data_list,axis=3)
        #print('data_list.shape is',np.array(data_list).shape)   #(1,17,400,1)

        prevalue ,y = self.sess.run([self.preValue,self.y],feed_dict={self.x:data_list,self.keep_prob:1.0})

        print("answer is{}：{}".format(prevalue,self.gesture[prevalue[0]]))
        print("softmax ",y[0][prevalue[0]])

    def main(self):
        # Wait for 3 seconds until the queue is full
        time.sleep(3)
        while True:
            self.update_analysis()
            time.sleep(1/25)

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(input=x,filters=W,strides = [1,1,1,1],padding = 'VALID')

def max_pool_1x2(x):
    return tf.nn.max_pool2d(input=x,ksize=[1,1,2,1],strides=[1,1,2,1],padding = 'VALID')

def team_network(x,keep_prob):
    n_class = 30
    
    x = tf.reshape(x,[-1,17,5,80,1])
    x = tf.transpose(a=x,perm=[0,2,1,3,4])
    x = tf.reshape(x,[-1,17,80,1])
    
    W_conv1 = weight_variable([1,3,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x,W_conv1)+b_conv1)
    #h_conv1_drop = tf.nn.dropout(h_conv1,0.5)
    
    h_pool1 = max_pool_1x2(h_conv1)
    print('h_pool1 shape is ',h_pool1.shape)
    
    W_conv2 = weight_variable([1,3,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
    #h_conv2_drop = tf.nn.dropout(h_conv2,0.5)
    
    h_pool2 = max_pool_1x2(h_conv2)
    print('h_pool2 shape is ',h_pool2.shape)
    
    W_conv3 = weight_variable([1,1,64,32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
    #h_conv3_drop = tf.nn.dropout(h_conv3,0.5)
    
    h_pool3 = max_pool_1x2(h_conv3)
    print('h_pool3 shape is ',h_pool3.shape)
    
    dim = 17*9*32
    h_flat = tf.reshape(h_pool3,[-1,dim])
    W_fc1 = weight_variable([dim,256])
    b_fc1 = bias_variable([256])
    h_fc1 = tf.nn.relu(tf.matmul(h_flat,W_fc1)+b_fc1)    
    h_fc1_drop = tf.nn.dropout(h_fc1,1 - (keep_prob))
    print('h_fc1_drop shape is ',h_fc1_drop.shape)
    
    h_step = tf.reshape(h_fc1_drop,[-1,5,256])
    h_trans = tf.transpose(a=h_step,perm=[1,0,2])
    x = tf.reshape(h_trans,[-1,256])
    h_trans = tf.split(x,5)
    #print('h_trans shape is ',h_trans.shape)
    
    lstm_fw_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(256,forget_bias = 1.0)
    lstm_bw_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(256,forget_bias = 1.0)
    
    outputs,_,_ = tf.compat.v1.nn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,h_trans,dtype=tf.float32)
    
    #cell_layers = []
    #for l in range (3):
     #   cell_layers.append(tf.nn.rnn_cell.LSTMCell(128, forget_bias=1.0, state_is_tuple=True,reuse=tf.AUTO_REUSE))
        # cell_layers.append(tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True))

    #lstm_cells = tf.contrib.rnn.MultiRNNCell(cell_layers, state_is_tuple=True)
    #outputs,_ = tf.nn.dynamic_rnn(lstm_cells, h_trans, dtype=tf.float32, time_major=True)
        
    # "Many-to-one" style for the classifier. We get only the last output of the layer.
    lstm_last_output = outputs[-1]
    print('lstm_last_output shape is '+str(lstm_last_output.shape))
    lstm_last_dropout=tf.nn.dropout(lstm_last_output,1 - (keep_prob))
    
    W_fc2 = weight_variable([256*2,n_class])
    b_fc2 = bias_variable([n_class])
    y_conv = tf.nn.softmax(tf.matmul(lstm_last_dropout,W_fc2)+b_fc2)
    
    return y_conv

def main():
    
    sess = tf.compat.v1.InteractiveSession()
    x = tf.compat.v1.placeholder(tf.float32,[None,17,400,1])
    keep_prob = tf.compat.v1.placeholder(tf.float32)
    y = team_network(x,keep_prob)
    preValue = tf.argmax(y,1)
    saver = tf.compat.v1.train.Saver()

    ckpt = tf.train.get_checkpoint_state("./model/")
    # 载入模型，不需要提供模型的名字，会通过 checkpoint 文件定位到最新保存的模型
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("load success")   
    else :
        print("load failed")
    
    myo.init(sdk_path='D:/myo/myo-sdk-win-0.9.0/')
    hub = myo.Hub()
    listener = EmgCollector(400)
    with hub.run_in_background(listener.on_event):
        Analysis(listener,sess,y,preValue,x,keep_prob).main()


if __name__ == '__main__':
    main()


# In[ ]:




