'''
CHINTA DINESH REDDY
EE15BTECH11007
'''

'''
the file is implementation of basic mlp on the data we received from biopat repository


'''


import numpy as np
import scipy.io as siop
import pandas as pd
import os
import glob
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt




print 'imports successful !'
np.random.seed(7)
def extract_useful(signal,fraction=1,actions=3):
    length_of_signal=len(signal)
    initial_offset=(length_of_signal/(3*2))*0.1
    initial_offset=int(initial_offset)
    final_offset=(length_of_signal/(3*2))*0.8
    final_offset=int(final_offset)
    first=signal[initial_offset:final_offset]
    second=signal[length_of_signal/3+initial_offset:length_of_signal/3+final_offset]
    third=signal[(length_of_signal/3)*2+initial_offset:(length_of_signal/3)*2+final_offset]
    return [first,second,third]


def windowing(signal,window_length=400,increment=100):
    quantity=len(signal)/100        #no of windows present
    signal=signal[:quantity*100]    #removing the end unwanted signal
    quantity=quantity-3             #no of times to run to loop to extract the samples
    output_samples=[]
    for i in range(quantity):
        output_samples.append(signal[:400])
        signal=np.roll(signal,100)
    return np.array(output_samples)


def rough_entrophy(one_sample):
    one_sample=list(one_sample)
    seted=list(set(one_sample))
    countarray=np.array([one_sample.count(i) for i in seted])
    entrophy=0
    universe_size=len(one_sample)
    entrophy_samples=countarray[countarray>1]           #to get elements which occured more than once
    for i in entrophy_samples:
        entrophy=entrophy+float(i)*np.log(float(i))/universe_size
    return entrophy

def corelation(signal1,signal2):
    signal1=np.array(signal1)
    signal2=np.array(signal2)
    mean1=signal1.mean()
    mean2=signal2.mean()
    signal1=signal1-mean1
    signal2=signal2-mean2
    numerator=sum(signal1*signal2)
    denominator=np.sqrt(sum((signal1**2)*(signal2**2)))
    return numerator/denominator


def zeroCrossigns(signal):
    signal=np.array(signal)
    length=len(signal)
    tmabs=sum(abs(signal))/length
    zc=0
    for i in range(length-1):
        fi=signal[i]
        fii=signal[i+1]
        if (fii*fi)<0 and abs(fii-fi)>tmabs:
            zc=zc+1
    return zc

def our_model():
    model=Sequential()
    model.add(Dense(28,input_dim=14,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


def main(file_name):
    subject_file=siop.loadmat(file_name)

    data_array=subject_file['recSession'][0][0][11]
    data_array_actions=[str(i[0][0]) for i in subject_file['recSession'][0][0][8]]
    samples_length=len(data_array)
    channel1=np.array([i[0] for i in data_array])
    channel2=np.array([i[1] for i in data_array])
    channel3=np.array([i[2] for i in data_array])
    channel4=np.array([i[3] for i in data_array])
    total_data=[channel1,channel2,channel3,channel4]


    '''
    the data has 3 minutes relaxation
    we need to take away the data
    '''
    print data_array_actions
    feature_array=[]
    for i in range(len(data_array_actions)):
        channel1_useful=extract_useful(channel1[:,i])
        channel1DataSamples=[windowing(j) for j in channel1_useful]
        channel1DataSamples=np.concatenate(channel1DataSamples,axis=0)
        channel2_useful=extract_useful(channel2[:,i])
        channel2DataSamples=[windowing(j) for j in channel2_useful]
        channel2DataSamples=np.concatenate(channel2DataSamples,axis=0)
        channel3_useful=extract_useful(channel3[:,i])
        channel3DataSamples=[windowing(j) for j in channel3_useful]
        channel3DataSamples=np.concatenate(channel3DataSamples,axis=0)
        channel4_useful=extract_useful(channel4[:,i])
        channel4DataSamples=[windowing(j) for j in channel4_useful]
        channel4DataSamples=np.concatenate(channel4DataSamples,axis=0)
        length=len(channel4DataSamples)
        actionArr=[]
        for j in range(length):
            entrophyArr=[]
            entrophyArr.append(rough_entrophy(channel1DataSamples[j]))
            entrophyArr.append(rough_entrophy(channel2DataSamples[j]))
            entrophyArr.append(rough_entrophy(channel3DataSamples[j]))
            entrophyArr.append(rough_entrophy(channel4DataSamples[j]))
            zcArr=[]
            zcArr.append(zeroCrossigns(channel1DataSamples[j]))
            zcArr.append(zeroCrossigns(channel2DataSamples[j]))
            zcArr.append(zeroCrossigns(channel3DataSamples[j]))
            zcArr.append(zeroCrossigns(channel4DataSamples[j]))
            corrArr=[]
            corrArr.append(corelation(channel1DataSamples[j],channel2DataSamples[j]))
            corrArr.append(corelation(channel1DataSamples[j],channel3DataSamples[j]))
            corrArr.append(corelation(channel1DataSamples[j],channel4DataSamples[j]))
            corrArr.append(corelation(channel2DataSamples[j],channel3DataSamples[j]))
            corrArr.append(corelation(channel2DataSamples[j],channel4DataSamples[j]))
            corrArr.append(corelation(channel3DataSamples[j],channel4DataSamples[j]))
            Arr=entrophyArr+zcArr+corrArr+[i]
            actionArr.append(Arr)
        feature_array.append(actionArr)
    print 'extracted features successfully !'
    feature_array=np.concatenate(feature_array,axis=0)
    #feature_array=np.concatenate(feature_array,axis=0)
    x=feature_array[:,:-1]
    y=feature_array[:,-1]
    dummy_y=np_utils.to_categorical(y)
    print 'building estimator !'
    estimator=KerasClassifier(build_fn=our_model,epochs=50,batch_size=5,verbose=0)
    kfold=KFold(n_splits=10,shuffle=True,random_state=7)
    print 'getting results !'
    results=cross_val_score(estimator,x,dummy_y,cv=kfold)
    return results

def get_file_names(path):
    file_array=[]
    for filename in glob.glob(os.path.join(path,'*.mat')):
        file_array.append(filename)
    return file_array


folder_name='/home/dinesh/Desktop/projects/amit/biopatrec-Data_Repository/10mov4chForearmUntargeted/'
file_array=get_file_names(folder_name)
client_result_array=[]
for filename in file_array:
    result=main(filename)
    acc_mean=result.mean()
    result=list(result)
    result.append(acc_mean)
    client_result_array.append(result)

pd.DataFrame(client_result_array,columns={'1','2','3','4','5','6','7','8','9','10','mean'}).to_csv('initial_mlp_results.csv')
