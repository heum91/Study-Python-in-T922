__author__ = 'Administrator'
import xlrd
import os
import nltk
from konlpy.tag import Kkma
from konlpy.utils import pprint

#kkma 형태소 분석
#text_file=open("C:\\Users\\HIIE\\Desktop\\5월 17일 랩실모임 자료\\피오니.txt","r", encoding='utf-8').read()
text_file=open("C:\\Users\\Owner\\Documents\\workspace\\comments\\원더.txt","r",encoding='utf-8').read()
kkma=Kkma()
kkma.pos(text_file)

#감성사전 정리
def extract_polaritylist(filename):
    wb = xlrd.open_workbook('C:\\Users\\Owner\\Desktop\\감성사전\\'+filename+'.xlsx')
    ws = wb.sheet_by_index(0)
    ncol = ws.ncols
    nrow = ws.nrows
    temp1=''
    temp2=''
    temp3=''
    polarity_list=[]

    i = 0
    while i < nrow: # (여기서 nrow인데 뒤에 몇개 짤려요)
        #print(str(ws.row_values(i)))
        temp1=(str(ws.row_values(i)[0]))
        temp2=(str(ws.row_values(i)[1]))
        temp3=(float)(ws.row_values(i)[2])
        tuple1=(temp1,temp2,temp3)
        b = polarity_list.append(tuple1)
        i = i + 1
    return polarity_list
#print(extract_polaritylist('positive'))

#문서와 감성사전 비교, 같은 값에서 점수 뽑아내기
A=[]
F=kkma.pos(text_file)

for i in F:
    if i[1] in ['NNG','VA','VV','MAG','IC','VXA']:
        A.append(i)
        
print(A)
B=extract_polaritylist('positive')
C=extract_polaritylist('negative')


adj_file_posit = extract_polaritylist('형용사 명사 긍정')
adj_file_nega = extract_polaritylist('형용사 명사 부정')

count_dic = {}

for i in A:
    if i[0] not in count_dic:
        count_dic[i[0]]=1
    else:
        count_dic[i[0]]+=1


p_res1 = {i[0]:count_dic[i[0]] for i in B if (i[0], i[1]) in A}
n_res1 = {i[0]:count_dic[i[0]] for i in C if (i[0], i[1]) in A}

#p_res2 = {i[0]:count_dic[i[0]] for i in adj_file_posit if i[0] in count_dic.keys()}
p_res2={}
for i in adj_file_posit:
    if i[0] in count_dic.keys():
        p_res2[i[0]] = count_dic[i[0]]
print(p_res2)
n_res2 = {i[0]:count_dic[i[0]] for i in adj_file_nega if i[0] in count_dic.keys()}
p_res={}
n_res={}
p_res1.update(p_res2)
n_res1.update(n_res2)

print(p_res1,"p_res1")
print(n_res1,"n_res1")

#p_res = {(i[0], i[1]): i[2] for i in B if (i[0], i[1]) in A_set} #튜플안의 자료들을 2개의 Key 와 1개의 value로 나눌수 있다
#n_res = {(i[0], i[1]): i[2] for i in C if (i[0], i[1]) in A_set}


p_count=0
for i in p_res1.values():
    p_count+=i
n_count=0
for i in n_res1.values():
    n_count+=i

print('p_count',p_count)
print('n_count',n_count)

print(p_count/(p_count+n_count))
