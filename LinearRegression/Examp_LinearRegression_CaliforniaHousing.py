#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sklearn
datasets = sklearn.datasets.fetch_california_housing()

#1. 문제데이터 x_data 정답데이터 y_data를 분리하시오
x_data = datasets["data"]
y_data = datasets["target"]
print(x_data.shape)
print(y_data.shape)
#2. feature_names 를 feature 변수로 분리하여 필드별 문제파일의 특성을 기재하시오
feature = datasets["feature_names"]
print(feature)
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
# MedInc- 평균 그룹의 중간 소득
# HouseAge- 평균 주택 년한
# AveRooms- 가구당 평균방의 갯수
# AveBedrms- 가구당 평균 침실수
# Population- 평균 인구수
# AveOccup- 평균 인원수
# Latitude- 위도
# Longitude- 경도
print(datasets["target_names"])#가격


# In[5]:


#위도와 경도는 평균값을 이용하고 모든 특성의 산점도 그래프를 그려 연관성을 시각화 하시오
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.subplot(3,3,1)
plt.scatter(x_data[:,0],y_data,s=1)
plt.title(feature[0])
plt.subplot(3,3,2)
plt.scatter(x_data[:,1],y_data,s=1)
plt.title(feature[1])
plt.subplot(3,3,3)
plt.scatter(x_data[:,2],y_data,s=1)
plt.title(feature[2])
plt.subplot(3,3,4)
plt.scatter(x_data[:,3],y_data,s=1)
plt.title(feature[3])
plt.subplot(3,3,5)
plt.scatter(x_data[:,4],y_data,s=1)
plt.title(feature[4])
plt.subplot(3,3,6)
plt.scatter(x_data[:,5],y_data,s=1)
plt.title(feature[5])
plt.subplot(3,3,7)
plt.scatter((x_data[:,6]+x_data[:,7])/len(x_data),y_data,s=1)
plt.title(feature[6])    


# In[6]:


import pandas as pd
df = pd.DataFrame(x_data,columns=feature)
print(df.describe())


# In[7]:


# 임계치 2(40) 3(5) 4(8000) 5(200)
plt.figure(figsize=(8,4))
plt.subplot(2,2,1)
plt.hist(x_data[:,2])
plt.subplot(2,2,2)
plt.hist(x_data[:,3])
plt.subplot(2,2,3)
plt.hist(x_data[:,4])
plt.subplot(2,2,4)
plt.hist(x_data[:,5])
plt.show()


# In[8]:


import numpy as np
#2(40) 3(5) 4(8000) 5(200)
def cutData(xdata,ydata):# 이상치 데이터 커팅
    tar2 = np.argwhere(xdata[:,2]>=40)
    xdata=np.delete(xdata,tar2,axis=0)
    ydata=np.delete(ydata,tar2,axis=0)

    tar3 = np.argwhere(xdata[:,3]>=5)
    xdata=np.delete(xdata,tar3,axis=0)
    ydata=np.delete(ydata,tar3,axis=0)   
    
    tar4 = np.argwhere(xdata[:,4]>=8000)
    xdata=np.delete(xdata,tar4,axis=0)
    ydata=np.delete(ydata,tar4,axis=0)
    
    tar5 = np.argwhere(xdata[:,5]>=200)
    xdata=np.delete(xdata,tar5,axis=0)
    ydata=np.delete(ydata,tar5,axis=0)    
    print(xdata.shape)
    print(ydata.shape)
    return xdata,ydata
x_data,y_data = cutData(x_data,y_data)


# In[9]:


#2(40) 3(5) 4(8000) 5(200)
plt.subplot(2,2,1)
plt.hist(x_data[:,2])
plt.subplot(2,2,2)
plt.hist(x_data[:,3])
plt.subplot(2,2,3)
plt.hist(x_data[:,4])
plt.subplot(2,2,4)
plt.hist(x_data[:,5])
plt.show()


# In[10]:


#데이터분할
x_train,x_test,y_train,y_test = \
    sklearn.model_selection.train_test_split(x_data,y_data,test_size=0.2,random_state=111)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[11]:


#3. x_train, x_test 를 정규화하시오
#  이때 x_train의 평균과 표준편차를 이용하세요, 정규화 후 데이터 한개를 확인합니다.
for ix in range(8):
    mean1 = np.mean(x_train[:,ix])
    std1 = np.std(x_train[:,ix])
    x_train[:,ix] = (x_train[:,ix]-mean1)/std1
    x_test[:,ix] = (x_test[:,ix]-mean1)/std1
print(x_train[0])
print(x_test[0])


# In[12]:


#4. 순서모델을 구성하고 입력층과 출력층을 작성하시오
#5. 모델을 컴파일하세요(손실함수는 mae를 사용하고 최적화함수는 경사하강법을 사용하세요)
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Input((8,)))
model.add(Dense(1))
model.compile(loss="MAE",optimizer="SGD")


# In[ ]:


fhist = model.fit(x_data,y_data,epochs=100)


# In[ ]:


#훈련결과 그래프를 그리시오
import matplotlib.pyplot as plt
plt.plot(fhist.history["loss"]


# In[ ]:




