#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Examp_classification_fashionMnist.ipynb
#1. 필요라이브러리 임포트
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense


# In[8]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
y_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
            "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
#Label 	Description
# 0 	T-shirt/top
# 1 	Trouser
# 2 	Pullover
# 3 	Dress
# 4 	Coat
# 5 	Sandal
# 6 	Shirt
# 7 	Sneaker
# 8 	Bag
# 9 	Ankle boot


# In[9]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(x_train[1000,14])
print(np.max(x_train))
print(np.min(x_train))
print(np.max(y_train))
# one hot encoding
# 0  [1 0 0 0 0 0 0 0 0 0]
# 2  [0 0 1 0 0 0 0 0 0 0]
# 9  [0,0,0,0,0,0,0,0,0,1]
#문제데이터 표준화 필요
#정답은 원핫인코딩이 필요 #(sklean train_test_split 분할은 원핫 인코딩 전에 해야합니다.)


# In[10]:


# 2. 데이터 정규화 min max
x_train = x_train/255.
x_test = x_test/255.
print(x_train[1000,14])


# In[11]:


#3. np.random.shuffle
from sklearn.utils import shuffle
x_train,y_train=shuffle(x_train,y_train, random_state=123)#훈련데이터 셔플
x_train = np.array(x_train)
y_train = np.array(y_train)
print(x_train.shape)
print(y_train[1])



# In[12]:


import matplotlib.pyplot as plt
plt.subplots_adjust(wspace=1,hspace=0.001)
for ix,data in enumerate(rarr):
    plt.subplot(2,5,ix+1)
    plt.imshow(x_test[data],cmap="gray")
    clr = y_test_label[data]==y_pred_label[data]
    plt.title("True:"+y_test_label[data])
    plt.xlabel("Pred:"+y_pred_label[data],color= ("blue" if clr else "red"))
    plt.xticks([]);plt.yticks([])
plt.show()


# In[ ]:


#3. 정답데이터 원핫 인코딩
#tf.one_hot(데이터, 구분클래스수량)
#tf.keras.utils.to_categorical(데이터, num_classes=구분클래스수량)
import sklearn
# print(y_train[5])
# print(tf.one_hot(y_train,10)[5])
# 원핫인코딩 , 정수변경 모두 가능, 정답레이블로 변환기능
# encoder = sklearn.preprocessing.OneHotEncoder(sparse_output=False)
# test_fit_data =  np.array(["T-shirt/top","Trouser","Pullove"])
# test_y_data = np.array(["T-shirt/top","Trouser","Pullove","Pullove","T-shirt/top"])
# encoder = encoder.fit(test_fit_data.reshape(len(test_fit_data),1))
# #env = encoder.fit_transform(test_y_data.reshape(len(test_y_data),1))
# env = encoder.transform(test_y_data.reshape(len(test_y_data),1))
# print("---------")
# print(env)
# print(encoder.get_feature_names_out())
# # print(dir(encoder))
# print(encoder.inverse_transform(np.array([[0., 0., 1.,],[1., 0., 0.]])))
# encoder.set_params(sparse_output=True)
# env = encoder.transform(test_y_data.reshape(len(test_y_data),1))
# print(env)
# print(y_train[:10])


# In[ ]:


# y_train = tf.one_hot(y_train,10)
# y_test = tf.one_hot(y_test,10)
from custom_encoder import CustomEncoder
encoder = CustomEncoder()
print(y_train.shape)
y_train = np.array(encoder.integer_to_one_hot(y_train,y_labels))
y_test = np.array(encoder.integer_to_one_hot(y_test,y_labels))
print(y_train.shape)
res = y_train[0]
print(y_train[0])
print(encoder.one_hot_to_label([res])) 


# In[ ]:


from tensorflow.keras.layers import Flatten
model = Sequential()
model.add(Input((28,28)))
model.add(Flatten())# 728 개의 벡터로 변형 ( 완전연결층 )
model.add(Dense(10,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="SGD",metrics=["acc"])
fhist = model.fit(x_train,y_train,validation_split=0.2,epochs=200,batch_size=5000)


# In[ ]:


plt.subplot(1,2,1)
plt.plot(fhist.history["acc"],label="train_acc")
plt.plot(fhist.history["val_acc"],label="valid_acc")
plt.legend()
plt.subplot(1,2,2)
plt.plot(fhist.history["loss"],label="train_los")
plt.plot(fhist.history["val_loss"],label="valid_los")
plt.legend()
plt.show()


# In[ ]:


res = model.evaluate(x_test,y_test)
print("손실도:",res[0]," 정확률:",int(res[1]*10000)/100,"%")


# In[ ]:


y_pred = model.predict(x_test)
print(y_test.shape)
print(y_pred.shape)
y_test_label = encoder.one_hot_to_label(y_test)
y_pred_label = encoder.one_hot_to_label(y_pred)
print(y_test_label[:5])
print(y_pred_label[:5])


# In[ ]:


#정답레이블로 변경
# 실제정답 레이블링
def conv_label(c_data):
    y_ix = np.array([np.argmax(data) for data in c_data])#원핫인코딩을 정수로 변경
    y_conv = np.array([y_labels[d] for d in y_ix])#변경된 정수를 레이블 인덱스로 레이블 명 인출
    print(y_true[:10])
    return y_conv


# In[ ]:


#정답레이블로 변경
# 실제정답 레이블링
# def conv_label(c_data):
#     y_ix = np.array([np.argmax(data) for data in c_data])#원핫인코딩을 정수로 변경
#     y_conv = np.array([y_labels[d] for d in y_ix])#변경된 정수를 레이블 인덱스로 레이블 명 인출
#     print(y_true[:10])
#     return y_conv
# y_test_conv = conv_label(y_test)
# y_pred_conv = conv_label(y_pred)
# print(y_test_conv[:10])
# print(y_pred_conv[:10])


# In[ ]:


np.random.seed(123)
rarr = np.random.randint(0,len(y_test_label),10)
print(rarr)


# In[ ]:


plt.subplots_adjust(wspace=1,hspace=0.001)
for ix,data in enumerate(rarr):
    plt.subplot(2,5,ix+1)
    plt.imshow(x_test[data],cmap="gray")
    clr = y_test_label[data]==y_pred_label[data]
    plt.title("True:"+y_test_label[data])
    plt.xlabel("Pred:"+y_pred_label[data],color= ("blue" if clr else "red"))
    plt.xticks([]);plt.yticks([])
plt.show()


# In[ ]:


tf.keras.models.save_model(model, r"./save_model/fashion_mnist_classification.keras")


# In[ ]:


import pickle
with open(r"./save_model/fashion_mnist.classification_encoder","wb") as fp:
    pickle.dump(encoder,fp)


# In[ ]:


import pickle
with open(r"./save_model/fashion_mnist.classification_encoder","rb") as fp:
    encoder = pickle.load(fp)

