import nibabel
import sys
import numpy as np
import keras.optimizers as optimizers
import time

from keras.models import Sequential,Model
from keras.layers import Input,Dense,Conv3D,MaxPooling3D,BatchNormalization,Dropout,Flatten
from keras.utils import np_utils
from keras import backend as K

t0=time.time()

seed=2
np.random.seed(seed)

path='/mnt/disk3/datasets_rm/data_set_dti/'

'''beginnings=['SW0033C', 'SW0045C', 'SW0081C', 'SW0085C', 'SW0186C', 'SW0242C', 'SW0247C', 'SW0283C', 'SW0288C', 'SW0291C', 'SW0295C', 'SW0312C', 'SW0317C', 'SW0318C', 'SW0397C', 'SW0410C', 'SW0450C', 'SW0457C', 'SW0486C', 'SW0516C', 'SW0528C', 'SW0544C', 'SW0597C', 'SW0598C', 'SW0614C', 'SW0655C', 'SW0668C', 'SW0701C', 'SW0746C', 'SW0753C', 'SW0773C', 'SW0816C', 'SW0822C', 'SW0866C', 'SW0878C', 'SW0879C', 'SW0930C', 'SW0994C', 'SW1055C', 'SW1081C', 'SW1133C', 'SW1144C', 'SW1222C', 'SW1228C', 'SW1391C', 'SW1442C', 'SW1450C', 'SW1468C', 'SW1507C', 'SW1536C', 'SW1553C', 'SW1584C', 'SW1590C', 'SW1636C', 'SW1647C', 'SW1717C', 'SW1720C', 'SW1861C', 'SW1930C', 'SW1938C', 'SW2100C', 'SW2154C', 'SW2158C', 'SW2173C', 'SW2181C', 'SW2183C', 'SW2189C', 'SW2220C', 'SW2269C', 'SW2295C', 'SW2379C', 'SW2380C', 'SW2382C', 'SW2389C', 'SW2416C', 'SW2503C', 'SW2525C', 'SW2544C', 'SW2571C', 'SW2578C', 'SW2581C', 'SW2604C', 'SW2639C', 'SW2644C', 'SW2696C', 'SW2703C', 'SW2714C', 'SW2741C', 'SW2750C', 'SW2888C', 'SW2906C', 'SW2957C', 'SW3075C', 'SW3081C', 'SW3082C', 'SW3231C', 'SW3445C', 'SW3482C', 'SW3483C', 'SW3491C', 'SW3592C', 'SW3651C', 'SW3747C', 'SW3833C', 'SW3881C', 'SW3944C']'''

endings = {'b0_mask':'_diff_b0_bet_mask.nii.gz', 'b0_bet':'_diff_b0_bet.nii.gz', 'b0':'_diff_b0.nii.gz', 'dti_FA':'_diff_dtifit_FA.nii.gz', 'dti_L1':'_diff_dtifit_L1.nii.gz', 'dti_L2':'_diff_dtifit_L2.nii.gz', 'dti_L3':'_diff_dtifit_L3.nii.gz', 'dti_MD':'_diff_dtifit_MD.nii.gz', 'dti_MO':'_diff_dtifit_MO.nii.gz', 'dti_RD':'_diff_dtifit_RD.nii.gz', 'dti_S0':'_diff_dtifit_S0.nii.gz', 'dti_tensor':'_diff_dtifit_tensor.nii.gz', 'dti_V1':'_diff_dtifit_V1.nii.gz', 'dti_V2':'_diff_dtifit_V2.nii.gz', 'dti_V3':'_diff_dtifit_V3.nii.gz', 'eddy_bet':'_diff_eddy_bet.nii.gz', 'eddy':'_diff_eddy.nii.gz', 'diff':'_diff.nii.gz'}

fd=open(path+"SW_groups.csv")
tmp=fd.readlines()
fd.close()
lines=[x.strip() for x in tmp[1:]]
splits=[line.split(',') for line in lines]
results={}
for split in splits:
    results[split[0]]=int(split[1])


input=[]
output=[]
for result in results:
    filename=path+result+endings['dti_V1']
    #print(filename)
    tempv=nibabel.load(filename).get_data()
    filename=path+result+endings['dti_L1']
    templ=nibabel.load(filename).get_data()
    templ=templ.reshape(templ.shape[0],templ.shape[1],templ.shape[2],1)
    temp=np.append(tempv,templ,axis=3)
    input.append(temp)
    output.append(results[result])

train_input=input[0:96]
train_output=output[0:96]

test_input=input[96:106]
test_output=output[96:106]

train_input=np.array(train_input)
test_input=np.array(test_input)


if K.image_data_format() == 'channels_first':
    train_input = train_input.reshape(train_input.shape[0], train_input.shape[4], train_input.shape[1], train_input.shape[2], train_input.shape[3])
    test_input = test_input.reshape(test_input.shape[0], test_input.shape[4], test_input.shape[1], test_input.shape[2], test_input.shape[3])
else:
    train_input = train_input.reshape(train_input.shape[0], train_input.shape[1], train_input.shape[2], train_input.shape[3],train_input.shape[4])
    test_input = test_input.reshape(test_input.shape[0], test_input.shape[1], test_input.shape[2], test_input.shape[3],test_input.shape[4])

print(train_input.shape)


model = Sequential()


model.add(Conv3D(32, (3, 3, 3), activation='relu', input_shape=train_input.shape[1:]))
model.add(MaxPooling3D(pool_size = (2, 2, 2)))
model.add(Conv3D(16, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size = (2, 2, 2)))
model.add(Conv3D(8, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size = (2, 2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(train_input[0])
coisa=model.fit(train_input, train_output, epochs=5, batch_size=2,shuffle=True,verbose=1, validation_split=0.1)

for x,y in zip(coisa.history['loss'],coisa.history['acc']):
	print("["+str(x)+"]   ["+str(y)+"]")



score = model.evaluate(test_input, test_output, verbose=1)
print(score)

sys.stdout.flush()
t1=time.time()

print("Time: %s seconds"%str(t1-t0))

