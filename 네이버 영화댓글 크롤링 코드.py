import urllib
import urllib.request
import urllib.parse
import bs4
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor
 
 
def deleteTag(x):
    return re.sub("<[^>]*>", "", x)
 
 
def getComments(code):
    def makeArgs(code, page):
        params = {
            'code': code,
            'type': 'after',
            'isActualPointWriteExecute': 'false',
            'isMileageSubscriptionAlready': 'false',
            'isMileageSubscriptionReject': 'false',
            'page': page,
            'order':'newest' #최신순으로 크롤링
        }
        return urllib.parse.urlencode(params)
 
    def innerHTML(s, sl=0):
        ret = ''
        for i in s.contents[sl:]:
            if i is str:
                ret += i.strip()
            else:
                ret += str(i)
        return ret
 
    def fText(s):
        if len(s): return innerHTML(s[0]).strip()
        return ''
 
    retList = []
    colSet = set()
    print("Processing: %d" % code)
    page = 1
    while 1:
        try:
            f = urllib.request.urlopen(
                "https://movie.naver.com/movie/bi/mi/pointWriteFormList.nhn?" + makeArgs(code, page))
            print(makeArgs(code, page))
            data = f.read().decode('utf-8')
        except:
            break
        soup = bs4.BeautifulSoup(re.sub("&#(?![0-9])", "", data), "html.parser")
        cs = soup.select(".score_result li")
        if not len(cs): break
        for link in cs:
            try:
                url = link.select('.score_reple em a')[0].get('onclick')
            except:
                print(page)
                print(data)
                raise ""
            m = re.search('[0-9]+', url)
            if m:
                url = m.group(0)
            else:
                url = ''
            if url in colSet: return retList
            colSet.add(url)
            cat = fText(link.select('.star_score em')) #평점
            cont = fText(link.select('.score_reple p')) #댓글내용
            cont = re.sub('<span [^>]+>.+?</span>', '', cont)
            retList.append((url, cat, cont))
        page += 1
        #최신순으로 10페이지 제한할때 아래사용, 없을시 제거
        if (page>10):
            return retList
 
    return retList
 
 
def fetch(i):
    outname = 'comments/%d.txt' % i
    try:
        if os.stat(outname).st_size > 0: return
    except:
        None
    rs = getComments(i)
    if not len(rs): return
    f = open(outname, 'w', encoding='utf-8')
    for idx, r in enumerate(rs):
        if idx: f.write(',\n')
        f.write("%s,'%s'" % (r[1], r[2].replace("'", "''").replace("\\", "\\\\"))) #수정 해야할 부분!
    f.write(';\n')
    f.close()
    time.sleep(1)
 
 #159892 코드의 영화 크롤링(ex.영화 탐정)
fetch(159892)


