## script to download info from a github repo

import os
import pandas as pd
import requests
import glob

filename = os.path.join("C:/path/name/folder/", 'repo.zip')
url = 'https://github.com/author/repo/archive/master.zip'

r = requests.get(url)

with open(filename, 'wb') as f:
    f.write(r.content)
 
from zipfile import ZipFile
file_name ="C:/path/name/folder/repo.zip"

with ZipFile(file_name, 'r') as zip:
    # printing all the contents of the zip file 
    zip.printdir() 
    zip.extractall("C:/path/name/folder/")## folder in where the files will be extracted
    print('Done!')
    
data = []

for item in glob.glob('C:/path/name/folder_non_zip/*.csv'):
    data.append(pd.read_csv(item))
    
df = pd.concat(data)