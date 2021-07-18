> # ---------------------**CROWD ANOMALY DETECTION**---------------------

> ### Presented by 
> # Vishakha Bhat and Sambit Sanyal


># ****DISCLAIMER****
> ### This code is best run using Google colab. Thats where it was tried and tested

> #### The code should run fine in any new Google colab project as long as the right uploads are done.


> ## *Please ignore this if you are working with Google collab.*

> ## Before you run this code, some installations need to be done:
---
> ### `!pip install numpy` to install numpy

> ### `!pip install sklearn` to install sklearn

> ### `!pip install Keras` to install keras

> ### `!pip install tensorflow` to install tensorflow , But ofcourse for model training its recommended you use the GPU version so install 

> ### `!pip install tensorflow-gpu` 

> ### `!pip install h5py` to install h5py

> ### `!pip install scipy` to install scipy

> ### `!pip install skimage` to install skimage

> ### `!pip install ffmpeg` to install ffmpeg

# **STEP 1)** 
> ## So First lets create the `trainer.npy` with the help of the videos and the datasets. Please upload the Avenue training dataset and set the directory location in the code.

---
> ## In the likely secenario where you cannot find the Avenue Dataset. 
> ## [Please look for it in this link.](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html)


> ## So I shall uploading the data and keeping it in a folder called 
> ## `training_videos` and shall be copying the directory path into the code.
---




```python
'''
Hello. Word of advice. Please ensure you check the variable video_source_path refers to the folder with the dataset of training 
and also make sure you have uploaded the correct training videos and not the testing videos


'''


from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import glob
import os 
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import argparse
from PIL import Image
imagestore=[]



video_source_path='/content/training_videos'
fps=5
#fps refers to the number of seconds after which one frame will be taken . fps=5 means 1 frame after every 5 seconds. More like seconds per frame.

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def remove_old_images(path):
	filelist = glob.glob(os.path.join(path, "*.png"))
	for f in filelist:
		os.remove(f)

def store(image_path):
	img=load_img(image_path)
	img=img_to_array(img)


	#Resize the Image to (227,227,3) for the network to be able to process it. 


	img=resize(img,(227,227,3))

	#Convert the Image to Grayscale


	gray=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]

	imagestore.append(gray)



#List of all Videos in the Source Directory.
videos=os.listdir(video_source_path)
print("Found ",len(videos)," training videos")


#Make a temp dir to store all the frames
create_dir(video_source_path+'/frames')

#Remove old images
remove_old_images(video_source_path+'/frames')

framepath=video_source_path+'/frames'

for video in videos:
		os.system( 'ffmpeg -i {}/{} -r 1/{}  {}/frames/%03d.jpg'.format(video_source_path,video,fps,video_source_path))
		images=os.listdir(framepath)
		for image in images:
			image_path=framepath+ '/'+ image
			store(image_path)


imagestore=np.array(imagestore)
a,b,c=imagestore.shape
#Reshape to (227,227,batch_size)
imagestore.resize(b,c,a)
#Normalize
imagestore=(imagestore-imagestore.mean())/(imagestore.std())
#Clip negative Values
imagestore=np.clip(imagestore,0,1)
np.save('trainer.npy',imagestore)
#Remove Buffer Directory
os.system('rm -r {}'.format(framepath))
print("Program ended. Please wait while trainer.npy is created. \nRefresh when needed")
print('Number of frames created :', int(len(imagestore)))
```

    Found  16  training videos
    Program ended. Please wait while trainer.npy is created. 
    Refresh when needed
    Number of frames created : 227
    

 >## So right now a new model trainer called the `trainer.npy` should have been created! 

>## Please confirm its existence before you jump into the next section. 

---

# **STEP 2)**   
> ## Now that `trainer.npy` is created , we can run the below code and train the model using it.

> ## The model so created will be called `AnomalyDetector.h5`



```python
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np 
import argparse
from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose
from keras.models import Sequential

''' The following load_model function code has been taken from 
Abnormal Event Detection in Videos using Spatiotemporal Autoencoder
by Yong Shean Chong Yong Haur Tay
Lee Kong Chian Faculty of Engineering Science, Universiti Tunku Abdul Rahman, 43000 Kajang, Malaysia.
It's main purpose is to help us generate the anomaly detector model
'''

#load_model starts here :----------------------------------------------------
def load_model():
	"""
	Return the model used for abnormal event 
	detection in videos using spatiotemporal autoencoder

	"""
	model=Sequential()
	model.add(Conv3D(filters=128,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',input_shape=(227,227,10,1),activation='tanh'))
	model.add(Conv3D(filters=64,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))



	model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,padding='same',dropout=0.4,recurrent_dropout=0.3,return_sequences=True))

	
	model.add(ConvLSTM2D(filters=32,kernel_size=(3,3),strides=1,padding='same',dropout=0.3,return_sequences=True))


	model.add(ConvLSTM2D(filters=64,kernel_size=(3,3),strides=1,return_sequences=True, padding='same',dropout=0.5))




	model.add(Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='tanh'))
	model.add(Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='tanh'))

	model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

	return model

#load_model ends here :----------------------------------------------------



X_train=np.load('trainer.npy')
frames=X_train.shape[2]
#Need to make number of frames divisible by 10 to ease the load_model


frames=frames-frames%10

X_train=X_train[:,:,:frames]
X_train=X_train.reshape(-1,227,227,10)
X_train=np.expand_dims(X_train,axis=4)
Y_train=X_train.copy()


epochs=200
batch_size=1



if __name__=="__main__":

	model=load_model()

	callback_save = ModelCheckpoint("AnomalyDetector.h5",
									monitor="mean_squared_error")

	callback_early_stopping = EarlyStopping(monitor='val_loss', patience=3)

	print('Trainer has been loaded')
	model.fit(X_train,Y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  callbacks = [callback_save,callback_early_stopping]
			  )
print('Done\n Please wait while AnomalyDetector.h5 has been created \nRefresh when needed')




```

    Trainer has been loaded
    Epoch 1/200
    22/22 [==============================] - 47s 2s/step - loss: 0.2402 - accuracy: 0.5296
    Epoch 2/200
    

    /usr/local/lib/python3.6/dist-packages/keras/callbacks/callbacks.py:846: RuntimeWarning: Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,accuracy
      (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
    

    22/22 [==============================] - 47s 2s/step - loss: 0.2002 - accuracy: 0.5491
    Epoch 3/200
    22/22 [==============================] - 48s 2s/step - loss: 0.1813 - accuracy: 0.5735
    Epoch 4/200
    22/22 [==============================] - 48s 2s/step - loss: 0.1196 - accuracy: 0.6793
    Epoch 5/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0880 - accuracy: 0.7185
    Epoch 6/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0802 - accuracy: 0.7257
    Epoch 7/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0757 - accuracy: 0.7315
    Epoch 8/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0733 - accuracy: 0.7344
    Epoch 9/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0716 - accuracy: 0.7353
    Epoch 10/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0695 - accuracy: 0.7364
    Epoch 11/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0685 - accuracy: 0.7372
    Epoch 12/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0682 - accuracy: 0.7374
    Epoch 13/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0673 - accuracy: 0.7383
    Epoch 14/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0657 - accuracy: 0.7393
    Epoch 15/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0649 - accuracy: 0.7401
    Epoch 16/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0636 - accuracy: 0.7415
    Epoch 17/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0627 - accuracy: 0.7425
    Epoch 18/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0619 - accuracy: 0.7429
    Epoch 19/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0615 - accuracy: 0.7437
    Epoch 20/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0609 - accuracy: 0.7442
    Epoch 21/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0601 - accuracy: 0.7447
    Epoch 22/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0594 - accuracy: 0.7451
    Epoch 23/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0584 - accuracy: 0.7460
    Epoch 24/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0578 - accuracy: 0.7468
    Epoch 25/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0571 - accuracy: 0.7473
    Epoch 26/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0563 - accuracy: 0.7479
    Epoch 27/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0553 - accuracy: 0.7486
    Epoch 28/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0546 - accuracy: 0.7491
    Epoch 29/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0532 - accuracy: 0.7500
    Epoch 30/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0527 - accuracy: 0.7508
    Epoch 31/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0502 - accuracy: 0.7532
    Epoch 32/200
    22/22 [==============================] - 49s 2s/step - loss: 0.0482 - accuracy: 0.7554
    Epoch 33/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0463 - accuracy: 0.7574
    Epoch 34/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0455 - accuracy: 0.7588
    Epoch 35/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0428 - accuracy: 0.7617
    Epoch 36/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0408 - accuracy: 0.7641
    Epoch 37/200
    22/22 [==============================] - 49s 2s/step - loss: 0.0388 - accuracy: 0.7662
    Epoch 38/200
    22/22 [==============================] - 49s 2s/step - loss: 0.0376 - accuracy: 0.7680
    Epoch 39/200
    22/22 [==============================] - 49s 2s/step - loss: 0.0367 - accuracy: 0.7690
    Epoch 40/200
    22/22 [==============================] - 49s 2s/step - loss: 0.0357 - accuracy: 0.7701
    Epoch 41/200
    22/22 [==============================] - 54s 2s/step - loss: 0.0349 - accuracy: 0.7708
    Epoch 42/200
    22/22 [==============================] - 50s 2s/step - loss: 0.0337 - accuracy: 0.7718
    Epoch 43/200
    22/22 [==============================] - 50s 2s/step - loss: 0.0331 - accuracy: 0.7725
    Epoch 44/200
    22/22 [==============================] - 50s 2s/step - loss: 0.0335 - accuracy: 0.7728
    Epoch 45/200
    22/22 [==============================] - 49s 2s/step - loss: 0.0319 - accuracy: 0.7736
    Epoch 46/200
    22/22 [==============================] - 49s 2s/step - loss: 0.0311 - accuracy: 0.7742
    Epoch 47/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0305 - accuracy: 0.7746
    Epoch 48/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0302 - accuracy: 0.7750
    Epoch 49/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0300 - accuracy: 0.7751
    Epoch 50/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0293 - accuracy: 0.7756
    Epoch 51/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0289 - accuracy: 0.7758
    Epoch 52/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0287 - accuracy: 0.7760
    Epoch 53/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0284 - accuracy: 0.7764
    Epoch 54/200
    22/22 [==============================] - 52s 2s/step - loss: 0.0277 - accuracy: 0.7767
    Epoch 55/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0274 - accuracy: 0.7769
    Epoch 56/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0271 - accuracy: 0.7771
    Epoch 57/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0271 - accuracy: 0.7772
    Epoch 58/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0269 - accuracy: 0.7775
    Epoch 59/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0262 - accuracy: 0.7777
    Epoch 60/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0260 - accuracy: 0.7779
    Epoch 61/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0261 - accuracy: 0.7780
    Epoch 62/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0260 - accuracy: 0.7782
    Epoch 63/200
    22/22 [==============================] - 47s 2s/step - loss: 0.0255 - accuracy: 0.7784
    Epoch 64/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0252 - accuracy: 0.7787
    Epoch 65/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0248 - accuracy: 0.7790
    Epoch 66/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0245 - accuracy: 0.7793
    Epoch 67/200
    22/22 [==============================] - 51s 2s/step - loss: 0.0242 - accuracy: 0.7795
    Epoch 68/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0241 - accuracy: 0.7797
    Epoch 69/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0239 - accuracy: 0.7799
    Epoch 70/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0236 - accuracy: 0.7802
    Epoch 71/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0234 - accuracy: 0.7804
    Epoch 72/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0231 - accuracy: 0.7806
    Epoch 73/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0232 - accuracy: 0.7807
    Epoch 74/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0228 - accuracy: 0.7810
    Epoch 75/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0230 - accuracy: 0.7811
    Epoch 76/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0227 - accuracy: 0.7813
    Epoch 77/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0222 - accuracy: 0.7815
    Epoch 78/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0220 - accuracy: 0.7817
    Epoch 79/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0219 - accuracy: 0.7818
    Epoch 80/200
    22/22 [==============================] - 50s 2s/step - loss: 0.0218 - accuracy: 0.7819
    Epoch 81/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0216 - accuracy: 0.7821
    Epoch 82/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0214 - accuracy: 0.7823
    Epoch 83/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0213 - accuracy: 0.7825
    Epoch 84/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0213 - accuracy: 0.7826
    Epoch 85/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0210 - accuracy: 0.7828
    Epoch 86/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0209 - accuracy: 0.7829
    Epoch 87/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0209 - accuracy: 0.7830
    Epoch 88/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0205 - accuracy: 0.7832
    Epoch 89/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0204 - accuracy: 0.7833
    Epoch 90/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0203 - accuracy: 0.7833
    Epoch 91/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0201 - accuracy: 0.7835
    Epoch 92/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0200 - accuracy: 0.7836
    Epoch 93/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0203 - accuracy: 0.7837
    Epoch 94/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0202 - accuracy: 0.7837
    Epoch 95/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0198 - accuracy: 0.7839
    Epoch 96/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0198 - accuracy: 0.7840
    Epoch 97/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0196 - accuracy: 0.7841
    Epoch 98/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0195 - accuracy: 0.7841
    Epoch 99/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0194 - accuracy: 0.7842
    Epoch 100/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0194 - accuracy: 0.7843
    Epoch 101/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0191 - accuracy: 0.7845
    Epoch 102/200
    22/22 [==============================] - 50s 2s/step - loss: 0.0190 - accuracy: 0.7845
    Epoch 103/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0190 - accuracy: 0.7846
    Epoch 104/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0189 - accuracy: 0.7847
    Epoch 105/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0189 - accuracy: 0.7847
    Epoch 106/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0187 - accuracy: 0.7848
    Epoch 107/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0188 - accuracy: 0.7849
    Epoch 108/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0186 - accuracy: 0.7849
    Epoch 109/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0185 - accuracy: 0.7850
    Epoch 110/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0185 - accuracy: 0.7851
    Epoch 111/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0182 - accuracy: 0.7851
    Epoch 112/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0182 - accuracy: 0.7852
    Epoch 113/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0180 - accuracy: 0.7853
    Epoch 114/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0181 - accuracy: 0.7853
    Epoch 115/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0180 - accuracy: 0.7854
    Epoch 116/200
    22/22 [==============================] - 49s 2s/step - loss: 0.0178 - accuracy: 0.7855
    Epoch 117/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0179 - accuracy: 0.7854
    Epoch 118/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0178 - accuracy: 0.7856
    Epoch 119/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0178 - accuracy: 0.7856
    Epoch 120/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0178 - accuracy: 0.7857
    Epoch 121/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0175 - accuracy: 0.7857
    Epoch 122/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0174 - accuracy: 0.7858
    Epoch 123/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0174 - accuracy: 0.7858
    Epoch 124/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0172 - accuracy: 0.7859
    Epoch 125/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0172 - accuracy: 0.7859
    Epoch 126/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0172 - accuracy: 0.7859
    Epoch 127/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0171 - accuracy: 0.7860
    Epoch 128/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0171 - accuracy: 0.7861
    Epoch 129/200
    22/22 [==============================] - 50s 2s/step - loss: 0.0170 - accuracy: 0.7861
    Epoch 130/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0170 - accuracy: 0.7861
    Epoch 131/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0169 - accuracy: 0.7862
    Epoch 132/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0167 - accuracy: 0.7862
    Epoch 133/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0168 - accuracy: 0.7863
    Epoch 134/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0166 - accuracy: 0.7863
    Epoch 135/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0166 - accuracy: 0.7863
    Epoch 136/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0166 - accuracy: 0.7864
    Epoch 137/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0165 - accuracy: 0.7864
    Epoch 138/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0164 - accuracy: 0.7864
    Epoch 139/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0166 - accuracy: 0.7864
    Epoch 140/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0164 - accuracy: 0.7865
    Epoch 141/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0164 - accuracy: 0.7865
    Epoch 142/200
    22/22 [==============================] - 48s 2s/step - loss: 0.0162 - accuracy: 0.7866
    Epoch 143/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0162 - accuracy: 0.7866
    Epoch 144/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0162 - accuracy: 0.7866
    Epoch 145/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0162 - accuracy: 0.7867
    Epoch 146/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0161 - accuracy: 0.7867
    Epoch 147/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0162 - accuracy: 0.7867
    Epoch 148/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0160 - accuracy: 0.7868
    Epoch 149/200
    22/22 [==============================] - 39s 2s/step - loss: 0.0158 - accuracy: 0.7869
    Epoch 150/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0158 - accuracy: 0.7869
    Epoch 151/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0158 - accuracy: 0.7869
    Epoch 152/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0157 - accuracy: 0.7869
    Epoch 153/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0159 - accuracy: 0.7869
    Epoch 154/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0156 - accuracy: 0.7870
    Epoch 155/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0155 - accuracy: 0.7870
    Epoch 156/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0155 - accuracy: 0.7871
    Epoch 157/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0155 - accuracy: 0.7871
    Epoch 158/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0155 - accuracy: 0.7871
    Epoch 159/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0155 - accuracy: 0.7871
    Epoch 160/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0156 - accuracy: 0.7872
    Epoch 161/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0153 - accuracy: 0.7872
    Epoch 162/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0154 - accuracy: 0.7872
    Epoch 163/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0154 - accuracy: 0.7873
    Epoch 164/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0152 - accuracy: 0.7873
    Epoch 165/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0152 - accuracy: 0.7873
    Epoch 166/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0152 - accuracy: 0.7873
    Epoch 167/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0151 - accuracy: 0.7873
    Epoch 168/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0152 - accuracy: 0.7873
    Epoch 169/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0151 - accuracy: 0.7874
    Epoch 170/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0150 - accuracy: 0.7874
    Epoch 171/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0149 - accuracy: 0.7875
    Epoch 172/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0150 - accuracy: 0.7874
    Epoch 173/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0150 - accuracy: 0.7875
    Epoch 174/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0148 - accuracy: 0.7875
    Epoch 175/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0148 - accuracy: 0.7875
    Epoch 176/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0148 - accuracy: 0.7875
    Epoch 177/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0147 - accuracy: 0.7876
    Epoch 178/200
    22/22 [==============================] - 42s 2s/step - loss: 0.0147 - accuracy: 0.7876
    Epoch 179/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0146 - accuracy: 0.7877
    Epoch 180/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0146 - accuracy: 0.7877
    Epoch 181/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0146 - accuracy: 0.7876
    Epoch 182/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0147 - accuracy: 0.7877
    Epoch 183/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0147 - accuracy: 0.7877
    Epoch 184/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0147 - accuracy: 0.7877
    Epoch 185/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0146 - accuracy: 0.7877
    Epoch 186/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0145 - accuracy: 0.7878
    Epoch 187/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0145 - accuracy: 0.7878
    Epoch 188/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0144 - accuracy: 0.7878
    Epoch 189/200
    22/22 [==============================] - 43s 2s/step - loss: 0.0144 - accuracy: 0.7878
    Epoch 190/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0143 - accuracy: 0.7878
    Epoch 191/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0144 - accuracy: 0.7878
    Epoch 192/200
    22/22 [==============================] - 44s 2s/step - loss: 0.0143 - accuracy: 0.7879
    Epoch 193/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0143 - accuracy: 0.7879
    Epoch 194/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0142 - accuracy: 0.7879
    Epoch 195/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0141 - accuracy: 0.7879
    Epoch 196/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0142 - accuracy: 0.7879
    Epoch 197/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0141 - accuracy: 0.7879
    Epoch 198/200
    22/22 [==============================] - 45s 2s/step - loss: 0.0141 - accuracy: 0.7880
    Epoch 199/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0140 - accuracy: 0.7880
    Epoch 200/200
    22/22 [==============================] - 46s 2s/step - loss: 0.0139 - accuracy: 0.7880
    Done
     Please wait while AnomalyDetector.h5 has been created
    

> ### **If this is taking too long , please feel free to take this already ready trained model**
>  ### It has been with epoches as 200. Its final accuracy stands at 0.7880
> # [Download it from here](https://drive.google.com/drive/folders/1AbF68y7ofutgZObca8D4K6Z33s66GDCP?usp=sharing)

> ## Please do not run the next code untill you can confirm that `AnomalyDetector.h5` has been successfully created and its most accurate version has been updated. It takes a while. You can come back a century later.


---
**Please note that for demonstration of training the model please reduce the epoches to lower values. Running high value of epoches on CPU is not recommended.**



---

> ## Right now there is no longer a compulsion to continue the execution in Google colab. The files ```trainer.npy``` and ```AnomalyDetector.h5``` are enough to enable testing the videos in CPU.


> ## However for the completion of this document as a full fledged project the code below can be used to run on testing videos 


---



#**STEP 3)** 
> ## Upload the "Avenue Dataset" testing Videos in the folder called the `testing_videos`

---
> ## For testing the videos we need to create a `tester.npy` and run the trained model on it


> ## Make sure you have the `AnomalyDetector.h5` in the main folder otherwise you shall ofcourse get some errors. 


```python
'''
Hello. Word of advice. Please ensure you check the variable video_source_path refers to the folder with the dataset of testing videos
and also make sure you have uploaded the correct testing videos and not the training videos


'''


from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import glob
import os 
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import argparse
from PIL import Image
imagestore=[]



video_source_path='/content/testing_videos'
fps=5
#fps refers to the number of seconds after which one frame will be taken . fps=5 means 1 frame after every 5 seconds. More like seconds per frame.

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def remove_old_images(path):
	filelist = glob.glob(os.path.join(path, "*.png"))
	for f in filelist:
		os.remove(f)

def store(image_path):
	img=load_img(image_path)
	img=img_to_array(img)


	#Resize the Image to (227,227,3) for the network to be able to process it. 


	img=resize(img,(227,227,3))

	#Convert the Image to Grayscale


	gray=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]

	imagestore.append(gray)
#List of all Videos in the Source Directory.
videos=os.listdir(video_source_path)
print("Found ",len(videos)," testing videos")


#Make a temp dir to store all the frames
create_dir(video_source_path+'/frames')

#Remove old images
remove_old_images(video_source_path+'/frames')

framepath=video_source_path+'/frames'
total=0
video_count=0

for video in videos:
		video_count+=1
		print("Video number: ",video_count)
		print("Video:",str(video))
		image_count=0
		os.system( 'ffmpeg -i {}/{} -r 1/{}  {}/frames/%03d.jpg'.format(video_source_path,video,fps,video_source_path))
		images=os.listdir(framepath)
		image_count=len(images)
		for image in images:
			image_path=framepath+ '/'+ image
			store(image_path)
		total=len(images)+total
		print("Number of images:",image_count,"\n----------\n")


imagestore=np.array(imagestore)
a,b,c=imagestore.shape
#Reshape to (227,227,batch_size)
imagestore.resize(b,c,a)
#Normalize
imagestore=(imagestore-imagestore.mean())/(imagestore.std())
#Clip negative Values
imagestore=np.clip(imagestore,0,1)
np.save('tester.npy',imagestore)
#Remove Buffer Directory
os.system('rm -r {}'.format(framepath))

print("Program ended. All testing videos shall be stored in tester.npy \n Please wait while tester.npy is created. \nRefresh when needed")
print('Number of frames created :', int(total))
print ('Number of bunches=',int(total),"/10 = ",int(total/10))
print("\nCorrupted and unreadable bunches were ignored")
```

    Found  22  testing videos
    Video number:  1
    Video: 03.avi
    Number of images: 9 
    ----------
    
    Video number:  2
    Video: 01.avi
    Number of images: 13 
    ----------
    
    Video number:  3
    Video: 05.avi
    Number of images: 13 
    ----------
    
    Video number:  4
    Video: 15.avi
    Number of images: 13 
    ----------
    
    Video number:  5
    Video: 02.avi
    Number of images: 13 
    ----------
    
    Video number:  6
    Video: 13.avi
    Number of images: 13 
    ----------
    
    Video number:  7
    Video: 16.avi
    Number of images: 13 
    ----------
    
    Video number:  8
    Video: 09.avi
    Number of images: 13 
    ----------
    
    Video number:  9
    Video: 10.avi
    Number of images: 13 
    ----------
    
    Video number:  10
    Video: 17.avi
    Number of images: 13 
    ----------
    
    Video number:  11
    Video: 04.avi
    Number of images: 13 
    ----------
    
    Video number:  12
    Video: 06.avi
    Number of images: 13 
    ----------
    
    Video number:  13
    Video: .ipynb_checkpoints
    Number of images: 13 
    ----------
    
    Video number:  14
    Video: 07.avi
    Number of images: 13 
    ----------
    
    Video number:  15
    Video: 20.avi
    Number of images: 13 
    ----------
    
    Video number:  16
    Video: 14.avi
    Number of images: 13 
    ----------
    
    Video number:  17
    Video: 21.avi
    Number of images: 13 
    ----------
    
    Video number:  18
    Video: 08.avi
    Number of images: 13 
    ----------
    
    Video number:  19
    Video: 18.avi
    Number of images: 13 
    ----------
    
    Video number:  20
    Video: 11.avi
    Number of images: 13 
    ----------
    
    Video number:  21
    Video: 19.avi
    Number of images: 13 
    ----------
    
    Video number:  22
    Video: 12.avi
    Number of images: 13 
    ----------
    
    Program ended. All testing videos shall be stored in tester.npy 
     Please wait while tester.npy is created. 
    Refresh when needed
    Number of frames created : 282
    Number of bunches= 282 /10 =  28
    
    Corrupted and unreadable bunches were ignored
    

#**STEP 4)** 
> ## Right now wait for the `tester.npy` to generate and then run the below code


> ## Some errors may creep in the dataset, but they will be removed in the program because they will be corrupted and unreadable


```python

from keras.models import load_model
import numpy as np 




def mean_squared_loss(x1,x2):


	''' Compute Euclidean Distance Loss  between 
	input frame and the reconstructed frame'''


	diff=x1-x2
	a,b,c,d,e=diff.shape
	n_samples=a*b*c*d*e
	sq_diff=diff**2
	Sum=sq_diff.sum()
	dist=np.sqrt(Sum)
	mean_dist=dist/n_samples

	return mean_dist






'''Define threshold for Sensitivity
Lower the Threshhold,higher the chances that a bunch of frames will be flagged as Anomalous.

'''

#threshold=0.0004 #(Accuracy level 1)
#threshold=0.00042 #(Accuracy level 2)
threshold=0.0008#(Accuracy level Vishakha)

model=load_model('AnomalyDetector.h5')

X_test=np.load('tester.npy')
frames=X_test.shape[2]
#Need to make number of frames divisible by 10


flag=0 #Overall video flagq

frames=frames-frames%10

X_test=X_test[:,:,:frames]
X_test=X_test.reshape(-1,227,227,10)
X_test=np.expand_dims(X_test,axis=4)
counter =0
for number,bunch in enumerate(X_test):
	n_bunch=np.expand_dims(bunch,axis=0)
	reconstructed_bunch=model.predict(n_bunch)


	loss=mean_squared_loss(n_bunch,reconstructed_bunch)
	
	if loss>threshold:
		print("Anomalous bunch of frames at bunch number {}".format(number))
		counter=counter+1
		print("bunch number: ",counter)
		flag=1


	else:
		print('No anomaly')
		counter=counter+1
		print("bunch number: ",counter)



if flag==1:
	print("Anomalous Events detected")
else:
	print("No anomaly detected")
	
print("\nCorrupted and unreadable bunches were ignored")
```

    No anomaly
    bunch number:  1
    No anomaly
    bunch number:  2
    No anomaly
    bunch number:  3
    Anomalous bunch of frames at bunch number 3
    bunch number:  4
    Anomalous bunch of frames at bunch number 4
    bunch number:  5
    No anomaly
    bunch number:  6
    No anomaly
    bunch number:  7
    No anomaly
    bunch number:  8
    No anomaly
    bunch number:  9
    No anomaly
    bunch number:  10
    No anomaly
    bunch number:  11
    No anomaly
    bunch number:  12
    No anomaly
    bunch number:  13
    No anomaly
    bunch number:  14
    No anomaly
    bunch number:  15
    No anomaly
    bunch number:  16
    No anomaly
    bunch number:  17
    Anomalous bunch of frames at bunch number 17
    bunch number:  18
    Anomalous bunch of frames at bunch number 18
    bunch number:  19
    Anomalous bunch of frames at bunch number 19
    bunch number:  20
    No anomaly
    bunch number:  21
    No anomaly
    bunch number:  22
    No anomaly
    bunch number:  23
    No anomaly
    bunch number:  24
    No anomaly
    bunch number:  25
    No anomaly
    bunch number:  26
    No anomaly
    bunch number:  27
    No anomaly
    bunch number:  28
    Anomalous Events detected
    

> ## Now to run the code on chosen files, we have to run the following code . Please set the video file location in the code

> ## Please upload the `test.mp4` or `test.avi` as a testing video. Please ensure that the video isnt doctored and edited. It should be continous stream of frames


---




```python
'''
Hello. Word of advice. Please ensure you check the variable video_source_path refers to the folder with the dataset of training 
and also make sure you have uploaded the correct training videos and not the testing videos


'''
from keras.models import load_model
import numpy as np 

from keras.preprocessing.image import img_to_array,load_img
import numpy as np
import glob
import os 
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import argparse
from PIL import Image
imagestore=[]



video_source_path='/content/'
fps=5
#fps refers to the number of seconds after which one frame will be taken . fps=5 means 1 frame after every 5 seconds. More like seconds per frame.

def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def remove_old_images(path):
	filelist = glob.glob(os.path.join(path, "*.png"))
	for f in filelist:
		os.remove(f)

def store(image_path):
	img=load_img(image_path)
	img=img_to_array(img)


	#Resize the Image to (227,227,3) for the network to be able to process it. 


	img=resize(img,(227,227,3))

	#Convert the Image to Grayscale


	gray=0.2989*img[:,:,0]+0.5870*img[:,:,1]+0.1140*img[:,:,2]

	imagestore.append(gray)



#List of all Videos in the Source Directory.
videos=os.listdir(video_source_path)
print("Found ",len(videos)," files")


#Make a temp dir to store all the frames
create_dir(video_source_path+'/frames')

#Remove old images
remove_old_images(video_source_path+'/frames')

framepath=video_source_path+'/frames'
flag=0
for video in videos:
		if (video=="test.avi" or video=="test.mp4"):
			print("Test video found")
			flag=1
			os.system( 'ffmpeg -i {}/{} -r 1/{}  {}/frames/%03d.jpg'.format(video_source_path,video,fps,video_source_path))
			images=os.listdir(framepath)
			for image in images:
				image_path=framepath+ '/'+ image
				store(image_path)

if flag==0:
	print("Couldn't find test.mp4 or test.avi. Make sure you reupload and try this")
	exit()
imagestore=np.array(imagestore)
a,b,c=imagestore.shape
#Reshape to (227,227,batch_size)
imagestore.resize(b,c,a)
#Normalize
imagestore=(imagestore-imagestore.mean())/(imagestore.std())
#Clip negative Values
imagestore=np.clip(imagestore,0,1)
np.save('sample.npy',imagestore)
#Remove Buffer Directory
os.system('rm -r {}'.format(framepath))
print("Please wait while video is processed. \nRefresh when needed")


def mean_squared_loss(x1,x2):


	''' Compute Euclidean Distance Loss  between 
	input frame and the reconstructed frame'''


	diff=x1-x2
	a,b,c,d,e=diff.shape
	n_samples=a*b*c*d*e
	sq_diff=diff**2
	Sum=sq_diff.sum()
	dist=np.sqrt(Sum)
	mean_dist=dist/n_samples

	return mean_dist


'''Define threshold for Sensitivity
Lower the Threshhold,higher the chances that a bunch of frames will be flagged as Anomalous.

'''

#threshold=0.0004 #(Accuracy level 1)
#threshold=0.00042 #(Accuracy level 2)
threshold=0.0008#(Accuracy level 3)

model=load_model('AnomalyDetector.h5')

X_test=np.load('sample.npy')
frames=X_test.shape[2]
#Need to make number of frames divisible by 10


flag=0 #Overall video flagq

frames=frames-frames%10

X_test=X_test[:,:,:frames]
X_test=X_test.reshape(-1,227,227,10)
X_test=np.expand_dims(X_test,axis=4)
counter =0
for number,bunch in enumerate(X_test):
	n_bunch=np.expand_dims(bunch,axis=0)
	reconstructed_bunch=model.predict(n_bunch)


	loss=mean_squared_loss(n_bunch,reconstructed_bunch)
	
	if loss>threshold:
		print("Anomalous bunch of frames at bunch number {}".format(number))
		counter=counter+1
		print("bunch number: ",counter)
		flag=1


	else:
		print('No anomaly')
		counter=counter+1
		print("bunch number: ",counter)


print("----------------------------------------------------\nOUTPUT\n----------------------------------------------------\n")
if flag==1:
	print("Anomalous Events detected")
else:
	print("No anomaly detected")
	
print("\n----------------------------------------------------\nCorrupted and unreadable bunches were ignored")


```

    Found  6  files
    Test video found
    Please wait while video is processed. 
    Refresh when needed
    Anomalous bunch of frames at bunch number 0
    bunch number:  1
    ----------------------------------------------------
    OUTPUT
    ----------------------------------------------------
    
    Anomalous Events detected
    
    ----------------------------------------------------
    Corrupted and unreadable bunches were ignored
    

# **WE ARE DONE**
> ## Yipee. That ends our project.
> ## Queries and comments shall be addressed later I guess.
