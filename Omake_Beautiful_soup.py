#author: ohshirotakahito
#since: 202021/04/09

from bs4 import BeautifulSoup
from urllib import request
import numpy as np

url = 'https://vdata.nikkei.com/newsgraphics/coronavirus-japan-vaccine-status/'
response = request.urlopen(url)
soup = BeautifulSoup(response)
response.close()

soup = str(soup)

#保存するパスを作成する
save_path = ('testdata/'+'bf_' +'r.txt')

f = open(save_path, "w",encoding='utf-8')
f.write(soup)
f.close
