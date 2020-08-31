from bs4 import BeautifulSoup
from konlpy.tag import Twitter
from collections import Counter
from urllib.request import HTTPError
import requests
import os
import pdf_textconverter
import nltk
import shutil

headers = {
    'User-Agent': 'Chrome/37.0.2049.0'}
work_directory = "C:\\Users\\home\\Documents\\workspace"
os.chdir(work_directory) #디렉토리 변경

def convfname(s): #논문 파일명 유효성검사
    templist = list(s)
    for i in range(0,len(templist)-1):
        if templist[i] =='\\' or templist[i] =='?' or templist[i] =='/' or templist[i] =='|' or templist[i] == ':' or templist[i] =='<' or templist[i] =='>':
            templist[i] = ''
    rstr = ''.join(templist)
    return rstr

def get_download_url(url): #구글 학술정보 검색 페이지 url에서 다운받을 pdf파일들의 url주소 와 파일명 리스트[url,파일명]를 리턴하는 함수
  req = requests.get(url,headers=headers)
  soup = BeautifulSoup(req.content, 'html.parser',from_encoding='utf-8')
  downloadurl_list=[]
  gs_r_gs_or_gs_scl_class = soup.find_all("div", class_='gs_r gs_or gs_scl')
  for item in gs_r_gs_or_gs_scl_class:
    if not item.find(class_='gs_or_ggsm'):
        continue
    else:
        furl = str(item.find(class_='gs_or_ggsm').a.get('href'))
        fname = str(item.find(class_='gs_rt').a.get_text())
        newfname=convfname(fname)
        downloadurl_list.append([furl, newfname])

  return downloadurl_list

def get_download(url,fname,directory): #pdf파일을 지정한 디렉토리에 다운받는 함수
    try:
        req = requests.get(url, stream=True, headers=headers)
        if req.headers.get("Content-Length") != None:
            with open(directory + '\\'+fname, 'wb') as f:
                req.raw.decode_content = True
                shutil.copyfileobj(req.raw, f)
            print('다운로드 완료')
            return True
        else:
            print("다운로드에러")
            return False

    except HTTPError as e:
        print(e)
        print("http 에러")
        return False

def get_tags(text, ntags=10000):

       spliter = Twitter()
       nouns = spliter.nouns(text)
       count = Counter(nouns)
       return_list = []
       for n, c in count.most_common(ntags):
            temp = {'tag': n, 'count': c}
            return_list.append(temp)
       if not return_list:
            return get_tags_en(text)
       print('태깅완료')
       return return_list


def get_tags_en(text, ntags=10000):
    return_list=[]
    nouns = nltk.word_tokenize(text)
    count = Counter(nouns)
    for n, c in count.most_common((ntags)):
        temp = {'tag': n, 'count': c}
        return_list.append(temp)
    if not return_list:
        return get_tags_en(text)
    print('태깅완료')
    return return_list

#def FindFileInDirectory(path):
#    fnameList = []
#    print('디렉토리 파일 목록: ')
#    for root, dirs, files in os.walk(path):
#        for fname in files:
#
#            if os.path.isfile(path+'\\'+fname) and (True == fname.endswith('.pdf')):
#                fnameList.append(fname)
#                print(fname)
#
#    return fnameList

def get_ctextlist(fnamelist):
   ctextlist = []  # [카운팅리스트,파일명] 리스트
   convtext = ''
   for fname in fnamelist:
       #pdf 파일 유효성 검사
      try:
       convtext = pdf_textconverter.convert(fname)
       ctextlist.append([get_tags_en(convtext),fname])
      except:
          return None
   for i in ctextlist:
        print(i)
   return ctextlist

def main():
    URL=''
    cDownUrlList=[]
    cFnameList=[]
    cList=[]
    for i in range(0,1):
        URL='https://scholar.google.co.kr/scholar?start=' + str(i*10) + '&q=microsensor&hl=ko&as_sdt=0,5'
        #<---URL유효성검사 가능하면 들어가야함--->
        cDownUrlList.extend(get_download_url(URL))
    print(cDownUrlList)
    print('다운로드를 실행할 url 개수: ',len(cDownUrlList))
    for url in cDownUrlList:
        fname = url[1] + '.pdf'
        if  get_download(url[0],fname,os.getcwd()): #파일다운이 성공하면True를 리턴
            cFnameList.append(fname) #다운로드에 성공한 파일명을 리스트에 추가

    for i in cFnameList:
        print(i)

    cList.extend(get_ctextlist(cFnameList))





if __name__ == '__main__':
    main()
