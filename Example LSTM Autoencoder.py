import matplotlib.pyplot as plt
import seaborn as sns

from numpy import dot
from numpy.linalg import norm

import pandas as pd
import numpy as np
from pylab import rcParams

import tensorflow as tf
from keras import optimizers, Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed
from keras.callbacks import ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import copy

from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(11)
from sklearn.model_selection import train_test_split


rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal","Break"]

#%%
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

def flatten(x):
    flattened_x = np.empty((x.shape[0], x.shape[2]))  # sample x features array.
    for i in range(x.shape[0]):
        flattened_x[i] = x[i, (x.shape[1]-1), :]
    return(flattened_x)


def scale(x, scaler): # feature 를 묶어서 정규화 하는 과정 - 정규화 함수 만들기
    for i in range(x.shape[0]):
        x[i, :, :] = scaler.transform(x[i, :, :])        
    return x

sc = MinMaxScaler()
def minmaxscaler(x):
    for i in range(x.shape[0]):
        x[i,:,:] = sc.fit_transform(x_train_y0[i,:,:])
    return x

def mean_absolute_percentage_error(x_valid_scaled, valid_x_predictions): # mape 구하는 함수
    return np.mean(np.abs(flatten(x_valid_scaled) - flatten(valid_x_predictions))*100 / 
                    ((sum(flatten(x_valid_scaled))/len(flatten(x_valid_scaled))) + 0.0001), axis=1)
#%%
# Load Data & Remove time column, and the categorical columns
df = pd.read_csv('C:\\Users\\T919\\Desktop\\processminer_rare_event_mts_data.csv').drop(['time', 'x28', 'x61'], axis=1)

#%%
# 코사인 유사도 이용해서 불필요한 요인 제거하기
y = copy.deepcopy(df.y.values)

fixed_label = ([0])*y.shape[0]

for i in range(y.shape[0]): # 사고가 일어날 것 같은 시간 설정(하나 늘릴 때마다 for문 밑에 추가하면 됨)
    if y[i] == 1:
        fixed_label[i] = 1

df_scale = np.array(fixed_label).reshape(-1,1)
# sc = MinMaxScaler()
df_drop_y = df.drop(['y'], axis=1)

for i in range(df_drop_y.shape[1]):
    # x = sc.fit_transform(df_drop_y.iloc[:,i].values.reshape(-1,1)) # 정규화할 때 필요
    x = df_drop_y.iloc[:,i].values.reshape(-1,1)
    df_scale = np.append(df_scale,x,axis=1)

df_scale = pd.DataFrame(df_scale,columns = df.columns)

cosine_similarity = []

for i in range(df_scale.shape[1]):
    cos_value = cos_sim(df_scale['y'].values, df_scale.iloc[:,i].values)
    cosine_similarity.append(cos_value)

np_cosine_similarity = np.array(cosine_similarity)
df_cosine_similarity = pd.DataFrame(cosine_similarity, index=df.columns) # 보여줄때 쓰는거

sorted_cosine_similaruty = copy.deepcopy(cosine_similarity)
sorted_cosine_similaruty.sort()

remove_count = 58 # 지우고자 하는 요인의 갯수

drop_index = []
for i in range(remove_count):
    drop_index.append(df.columns[cosine_similarity.index(sorted_cosine_similaruty[i])])

for j in range(len(drop_index)):
    df = df.drop([str(drop_index[j])], axis=1) # df = y값과 관계없는 요인들 20개 지운 후
df = df.drop(['y'],axis=1)
#%%
lookback = 5  # Equivalent to 10 min of past data. = timestamp
n_features = input_x.shape[1]
input_x = copy.deepcopy(df)

input_y = pd.DataFrame(y,columns=['y'])

#2분 간격이기 때문에 4분 이후를 예상하려면 2번 시프트
for s in range(1,3):
    input_y['{}'.format(s)] = input_y['y'].shift(s)

input_y = input_y.drop(['y', '1'], axis=1).iloc[lookback-1:]

# input_x는 2차원 배열이기 때문에 --> 3D 크기 어레이인 sample x lookback x features으로 변환해야 한다.
output_x = []
output_y = []

for i in range(len(input_x) - lookback + 1):
# for i in range(3):
    t = []
    for j in range(0, lookback):
        t.append(input_x.iloc[i + j])
    output_x.append(t)
    output_y.append(input_y.iloc[i])
    
x = np.squeeze(np.array(output_x))
y = np.array(output_y)

x_train = np.array(x)[:15000]
y_train = np.array(y)[:15000]
x_test = np.array(x)[13000:]
y_test = np.array(y)[13000:]

x_valid = x_train[12000:]
y_valid = y_train[12000:]
x_train = x_train[:12000]
y_train = y_train[:12000]

#%%

# 정상, 고장에 따른 데이터 분류

# 0 정상, 1 고장
x_train_y0 = []
x_train_y1 = []
x_valid_y0 = []
x_valid_y1 = []

# 샘플 중 정상만을 분류

for i in range(len(y_train)):
    if float(y_train[i]) == 0:
        x_train_y0.append(x_train[i])
    elif float(y_train[i]) == 1:
        x_train_y1.append(x_train[i])

for i in range(len(y_valid)):
    if float(y_valid[i]) == 0:
        x_valid_y0.append(x_valid[i])
    elif float(y_valid[i]) == 1:
        x_valid_y1.append(x_valid[i])

x_train_y0 = np.array(x_train_y0)
x_train_y1 = np.array(x_train_y1)
x_valid_y0 = np.array(x_valid_y0)
x_valid_y1 = np.array(x_valid_y1)

x_train = x_train.reshape(x_train.shape[0], lookback, n_features)
x_train_y0 = x_train_y0.reshape(x_train_y0.shape[0], lookback, n_features)
x_train_y1 = x_train_y1.reshape(x_train_y1.shape[0], lookback, n_features)
x_valid = x_valid.reshape(x_valid.shape[0], lookback, n_features)
x_valid_y0 = x_valid_y0.reshape(x_valid_y0.shape[0], lookback, n_features)
x_valid_y1 = x_valid_y1.reshape(x_valid_y1.shape[0], lookback, n_features)
x_test = x_test.reshape(x_test.shape[0], lookback, n_features)


x_train_y0_scaled = minmaxscaler(x_train_y0)
x_valid_scaled = minmaxscaler(x_valid)
x_valid_y0_scaled = minmaxscaler(x_valid_y0)
x_test_scaled = minmaxscaler(x_test)

timesteps =  x_train_y0_scaled.shape[1] # equal to the lookback
n_features =  x_train_y0_scaled.shape[2] # 59

epochs = 300
batch = 32
lr = 0.0001

import keras.backend as K

def custom_loss(y_true,y_pred):
    loss = K.abs((y_pred - y_true)/K.mean(y_true))*100   
    return loss

# 모델 생성
lstm_autoencoder = Sequential()
# Encoder
lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
lstm_autoencoder.add(RepeatVector(timesteps))
# Decoder
lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(n_features)))
lstm_autoencoder.summary()

# 오토인코더 훈련
adam = optimizers.Adam(lr)
lstm_autoencoder.compile(loss='mse', optimizer=adam)

cp = ModelCheckpoint(filepath="lstm_autoencoder_classifier.h5",save_best_only=True,verbose=0)
tb = TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True,write_images=True)
lstm_autoencoder_history = lstm_autoencoder.fit(x_train_y0_scaled, x_train_y0_scaled, 
                                                epochs=epochs, 
                                                batch_size=batch, 
                                                validation_data=(x_valid_y0_scaled, x_valid_y0_scaled),
                                                verbose=2).history

plt.plot(lstm_autoencoder_history['loss'], linewidth=2, label='Train')
plt.plot(lstm_autoencoder_history['val_loss'], linewidth=2, label='Valid')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()


#%% 
valid_x_predictions = lstm_autoencoder.predict(x_valid_scaled)
mape = mean_absolute_percentage_error(x_valid_scaled, valid_x_predictions)
mlb = MultiLabelBinarizer()

y_valid_t = []
for i in range(len(y_valid.tolist())):
    y_valid_t.append(y_valid.tolist()[i][0])
    
error_df = pd.DataFrame(mape, columns=['Reconstruction_error'])
# error_df = pd.DataFrame(mse, columns=['Reconstruction_error'])
error_df['True_class'] = y_valid_t

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

#%%

test_x_predictions = lstm_autoencoder.predict(x_test_scaled)
mape = mean_absolute_percentage_error(x_test_scaled, test_x_predictions)

y_test_t = []
for i in range(len(y_test.tolist())):
    y_test_t.append(y_test.tolist()[i][0])

error_test_df = pd.DataFrame(mape, columns=['Test_Reconstruction_error'])    
error_test_df['True_class'] = y_test_t
error_test_df[error_test_df.True_class.apply(lambda x : x == 1)]

threshold_fixed = 10 # 0에 가까울 수록 불량으로 예측하는 갯수가 늘음
error_test_df_1 = copy.deepcopy(error_test_df.iloc[700:850])
groups = error_test_df_1.groupby('True_class')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.index, group.Test_Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Break" if name == 1 else "Normal")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.set_ylim(0,12)
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Test_Reconstruction error")
plt.xlabel("Data point index")
plt.show()

#%%
# Precision and recall for different threshold values
precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_test_df.True_class, error_test_df.Test_Reconstruction_error)
plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()