import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import datetime
import copy
import random
from collections import Counter
from gensim.models import Word2Vec
from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
import math
import re
import sentencepiece as spm

#%matplotlib inline
warnings.filterwarnings('ignore')

#os.getcwd()
os.chdir('C:\\Users\\T919\\Desktop\\7th')
json_data1=open('train.json',encoding='UTF-8').read()
train = json.loads(json_data1)

#json_data2=open('test.json',encoding='UTF-8').read()
#test = json.loads(json_data2)

json_data3=open('val.json',encoding='UTF-8').read()
val_df = json.loads(json_data3)

json_data4=open('song_meta.json',encoding='UTF-8').read()
song_meta = json.loads(json_data4)

genre_gn_all = pd.read_json('genre_gn_all.json', typ = 'series',encoding='UTF-8')
# 장르코드 : gnr_code, 장르명 : gnr_name
genre_df = pd.DataFrame(genre_gn_all, columns = ['gnr_name']).reset_index().rename(columns = {'index' : 'gnr_code'})
train_columns=list(train[0].keys())
#test_columns=list(test[0].keys())
song_columns=list(song_meta[0].keys())
val_columns=list(val_df[0].keys())
train_columns=['id','plylst_title','tags','songs','like_cnt','updt_date']
#test_columns=['id','plylst_title','tags','songs','like_cnt','updt_date']
train=pd.DataFrame(train,columns=train_columns)
#test=pd.DataFrame(test,columns=test_columns)
song_meta=pd.DataFrame(song_meta)
val_df=pd.DataFrame(val_df)

song_dataset_1 = pd.read_csv('ans_song_df_7th_0_12000.csv',header=None)
song_dataset_2 = pd.read_csv('ans_song_df_7th_12000_23015.csv',header=None)
song_1 = pd.DataFrame(song_dataset_1)
song_2 = pd.DataFrame(song_dataset_2)
ans_song = pd.concat([song_1,song_2]).values

tag_dataset_1 = pd.read_csv('ans_tag_df_7th_0_12000.csv',header=None)
tag_dataset_2 = pd.read_csv('ans_tag_df_7th_12000_23015.csv',header=None)
tag_1 = pd.DataFrame(tag_dataset_1)
tag_2 = pd.DataFrame(tag_dataset_2)
ans_tag = pd.concat([tag_1,tag_2]).values

id_list=list(val_df.id)
answer=[]
for i in range(23015):
    dic={'id':id_list[i], 'songs':ans_song[i], 'tags':ans_tag[i]}
    answer.append(dic)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

with open('results.json', 'w', encoding='UTF-8') as file:
    file.write(json.dumps(answer,cls=NpEncoder, ensure_ascii=False,indent='\t'))
    
    
import zipfile
       
jungle_zip = zipfile.ZipFile('C:\\Users\\T919\\Desktop\\result.zip', 'w')
jungle_zip.write('results.json', compress_type=zipfile.ZIP_DEFLATED)
 
jungle_zip.close()

jungle_zip = zipfile.ZipFile('C:\\Users\\T919\\Desktop\\카카오 아레나 코드.zip', 'w')
jungle_zip.write('2020_07_17 카카오 아레나 최종본 7th 수정본.py', compress_type=zipfile.ZIP_DEFLATED)
jungle_zip.close()