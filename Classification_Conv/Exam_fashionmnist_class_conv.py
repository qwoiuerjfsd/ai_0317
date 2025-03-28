#!/usr/bin/env python
# coding: utf-8

# In[57]:


#Exam_fashionmnist_class_conv.ipynb
#1. fashion_mnist data receive
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("x_train:",x_train.shape," y_train:",y_train.shape)
print("x_test:",x_test.shape," y_test:",y_test.shape)
label_list = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal",
              "Shirt","Sneaker","Bag","Ankle boot"]


# In[58]:


#2. 데이터 구조 확인
print(x_train[1][14])#정규화
print(y_train[1])#원핫인코딩


# In[59]:


#3. 데이터 분할
from sklearn.model_selection import train_test_split
x_valid,x_test,y_valid,y_test=\
    train_test_split(x_test,y_test,test_size=0.4,random_state=123,stratify=y_test)
print(x_valid.shape)
print(x_test.shape)
print(y_valid.shape)
print(y_test.shape)


# In[60]:


#4. 데이터 셔플 및 전처리(정규화,원핫인코딩)
import sklearn
x_train,y_train = sklearn.utils.shuffle(x_train,y_train, random_state=123)
x_train=(x_train/255.).reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],-1)
x_test=(x_test/255.).reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],-1)
x_valid=(x_valid/255.).reshape(x_valid.shape[0],x_valid.shape[1],x_valid.shape[2],-1)
y_train=tf.one_hot(y_train,len(label_list))
y_valid=tf.one_hot(y_valid,len(label_list))
y_test=tf.one_hot(y_test,len(label_list))
print(x_train[0][14][:5])
print(x_valid[0][14][:5])
print(x_test[0][14][:5])
print(y_train[0])
print(y_valid[0])
print(y_test[0])


# In[61]:


#5. 정답과 이미지 일치성 확인
import numpy as np
import matplotlib.pyplot as plt
t_rarr = np.random.randint(0,len(x_train),5)
v_rarr = np.random.randint(0,len(x_valid),5)
s_rarr = np.random.randint(0,len(x_test),5)
print(t_rarr)
print(v_rarr)
print(s_rarr)
ix = 0
plt.rc("font",size=7)
plt.figure(figsize=(5,5))
plt.subplots_adjust(hspace=1)
for t,v,s in zip(t_rarr,v_rarr,s_rarr):
    plt.subplot(5,3,ix+1)
    #이미지와 타이틀(y값을 이용하여)을 그리세요
    plt.imshow(x_train[t],cmap="gray");plt.xticks([]);plt.yticks([]);
    plt.title(label_list[np.argmax(y_train[t])])
    plt.subplot(5,3,ix+2)
    plt.imshow(x_valid[v],cmap="gray");plt.xticks([]);plt.yticks([]);
    plt.title(label_list[np.argmax(y_valid[v])])
    plt.subplot(5,3,ix+3)
    plt.imshow(x_test[s],cmap="gray");plt.xticks([]);plt.yticks([]);
    plt.title(label_list[np.argmax(y_test[s])])
    ix+=3
    


# In[62]:


#6. 모델 구성
from tensorflow.keras import Input,Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten


# In[63]:


cmodel = Sequential()
cmodel.add(Input((28,28,1)))
cmodel.add(Conv2D(10,5,padding="same",activation="relu"))
cmodel.add(MaxPool2D(4,padding="same"))
cmodel.add(Conv2D(20,3,padding="same",activation="relu"))
cmodel.add(MaxPool2D(2,padding="same"))
cmodel.add(Flatten())
cmodel.add(Dropout(0.3))
cmodel.add(Dense(128,activation="relu"))
cmodel.add(Dropout(0.3))
cmodel.add(Dense(32,activation="relu"))
cmodel.add(Dense(10,activation="softmax"))
cmodel.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])


# In[64]:


fhist = cmodel.fit(x_train,y_train,validation_data=(x_valid,y_valid),\
           epochs=100,batch_size=3000)


# In[65]:


plt.figure(figsize=(6,2))
plt.subplot(1,2,1)
plt.plot(fhist.history["acc"],label="train_acc")
plt.plot(fhist.history["val_acc"],label="valid_acc")
plt.legend()
plt.title("VALIDATION")
plt.subplot(1,2,2)
plt.plot(fhist.history["loss"],label="train_loss")
plt.plot(fhist.history["val_loss"],label="valid_loss")
plt.legend()
plt.title("LOSSES")
plt.show()


# In[66]:


#모델 평가(test 데이터와 라벨을 활용) 손실도와 , 정확률 출력
lossval,accval = cmodel.evaluate(x_test,y_test)
print("손실도: ", int(lossval*10000)/10000," 정확률:",int(accval*10000)/100,"%")


# In[67]:


#예측값 시각화
y_pred = cmodel.predict(x_test)
rarr = np.random.randint(0,len(x_test),10)
plt.figure(figsize=(5,5))
for ix,rix in enumerate(rarr):
    plt.subplot(2,5,ix+1)
    plt.imshow(x_test[rix],cmap="gray")
    clr = "red"
    if np.argmax(y_test[rix])==np.argmax(y_pred[rix]):clr="blue"
    plt.title(label_list[np.argmax(y_test[rix])],color=clr)
    plt.xticks([]);plt.yticks([])
    plt.xlabel(label_list[np.argmax(y_pred[rix])],color=clr)
plt.show()
    


# In[74]:


#혼동행렬 - 예측정답과 실제정답을 일치-레이블로 변경
print(y_test.shape)
print(y_pred.shape)
print(y_test[0])
print(y_pred[0])
#정수로 변경
y_real = np.argmax(y_test,axis=1)
y_real_pred = np.argmax(y_pred,axis=1)
print(y_real[0])
print(y_real_pred[0])
y_real = [label_list[d] for d in y_real]
y_real_pred = [label_list[d] for d in y_real_pred]
print(y_real[:5])
print(y_real_pred[5])


# In[75]:


cm = sklearn.metrics.confusion_matrix(y_real,y_real_pred)
print(cm)


# In[76]:


#혼동행렬의 시각화
import seaborn as sns
plt.figure(figsize=(6,2))
sns.heatmap(cm,fmt="d",cmap="Blues",annot=True,xticklabels=label_list,yticklabels=label_list)
plt.show()


# In[78]:


#f1 score
print(sklearn.metrics.classification_report(y_real,y_real_pred))


# In[79]:


cmodel.save(r"./fashionmnist_convlution.keara")


# In[ ]:




