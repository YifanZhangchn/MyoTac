# MyoTac: High Real-time Recognition of Tactical Sign Language Based on Lightweight Deep Neural Network

Realtime_application.py is the code for the real-time running part of the paper MyoTac: High Real-time Recognition of Tactical Sign Language Based on Lightweight Deep Neural Network.
Interested researchers can contact Zhang Yifan (E-mail: yifan.zhang.chn@foxmail.com) to get all the codes.

## Requirements

-Python 3.7+
-Tensorflow 2.0+
-collections
-numpy
-threading
-math
-time
-myo
-numpy

## Dataset
Data_tactical is the tactical sign language instruction data set we collected, which contains 30 tactical sign language instructions. We gathered a total of 25 volunteers, and each volunteer performed each sign language gesture 50 times. In the data set file format, p00-p24 represent 25 volunteers, 00-29 represent 30 tactical sign language instructions, sub-directory emg represents muscle electrical signal data, and accelerometer, gyro, and orientation are inertial sensor data. The 30 tactical sign language commands we selected include Male, female, commander, hostage, suspect, you, me, come on, hear, see, advance, message received, hurry up, stop, cover me, not understand, understand, squat down , ignore, pistol, rifle, automatic weapon, shotgun, car, door, corner, assemble, single column, two-way column, one-way line.


## Operation instructions
When running, you first need to wear the myo armband correctly, and keep myo_connect.exe running and myo connected.
The results can be output by making corresponding gestures during operation, as shown in  https://github.com/YifanZhangchn/MyoTac/blob/master/picture/myo.jpg. 

Note: The neural network used in realtime_application.py is not the final optimization. Interested researchers can contact Zhang Yifan (email: yifan.zhang.chn@foxmail.com) to get the final optimized neural network.
