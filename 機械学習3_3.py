#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
!pip install -U -q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

"""


# In[1]:


#!pip install jupyter-resource-usage


# In[ ]:


"""
id = '1oF0ljbTJxk0E9lNEbehEBAZM7FdL5aFn'  # 共有リンクで取得した id= より後の部分
downloaded = drive.CreateFile({'id': id})
downloaded.GetContentFile('dataset.zip')
"""


# In[ ]:


"""
!unzip dataset.zip
"""


# In[1]:


#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()


# In[2]:


#!pip install tensorflow==2.8
#!pip install pillow
#!pip install scikit-learn
#!pip install keras
#!pip install matplotlib

import os
from PIL import Image
import numpy as np
import  matplotlib.pylab as plt
import random
import re
from collections import defaultdict
import pathlib

import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow.keras.optimizers import Adam

from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# In[3]:


# 各種変数の定義

rootDirName = 'data/JPEG/dataset384'
height = 384
width = 384
num_classes = 5

make_data = "model_384"

# ワンホットラベルとクラス名の対応
idx_to_label = []


# In[ ]:



# 画像に対する前処理
def preprocess_image(img):
    img = tf.image.decode_image(img, channels=3)
    # img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_ACCURATE")
    img = tf.image.resize(img, [height, width]) # 画像サイズを変更
    img /= 255.0 # [0,1]に正規化

    return img

# データセットを作成する
# 返り値は(np.imgs, labels) = (画像、ラベル)
def make_dataset():

    np.imgs1=[]
    np.imgs2=[]
    np.imgs3=[]
    np.imgs4=[]
    np.imgs5=[]
    np.imgs6=[]
    np.imgs7=[]
    np.imgs8=[]
    np.imgs9=[]
    np.imgs10=[]
    np.imgs11=[]
    np.imgs12=[]
    np.imgs13=[]
    np.imgs14=[]
    np.imgs15=[]
    np.imgs16=[]
    np.imgs17=[]
    np.imgs18=[]
    np.imgs19=[]
    np.imgs20=[]
    np.imgs21=[]
    np.imgs22=[]
    np.imgs23=[]
    np.imgs24=[]
    np.imgs25=[]
    np.imgs26=[]
    np.imgs27=[]
    np.imgs28=[]
    np.imgs29=[]
    np.imgs30=[]
    np.imgs31=[]
    np.imgs32=[]
    np.imgs33=[]
    np.imgs34=[]
    np.imgs35=[]
    np.imgs36=[]
    np.imgs37=[]
    np.imgs38=[]
    np.imgs39=[]
    np.imgs40=[]
    np.imgs41=[]
    np.imgs42=[]
    np.imgs43=[]
    np.imgs44=[]
    np.imgs45=[]
    np.imgs46=[]
    np.imgs47=[]
    np.imgs48=[]
    np.imgs49=[]
    np.imgs50=[]
    np.imgs51=[]
    np.imgs52=[]
    np.imgs53=[]
    np.imgs54=[]
    np.imgs55=[]
    np.imgs56=[]
    np.imgs57=[]
    np.imgs58=[]
    np.imgs59=[]
    np.imgs60=[]
    np.imgs61=[]
    np.imgs62=[]
    np.imgs63=[]
    np.imgs64=[]
    np.imgs65=[]
    np.imgs66=[]
    np.imgs67=[]
    np.imgs68=[]
    np.imgs69=[]
    np.imgs70=[]
    np.imgs71=[]
    np.imgs72=[]
    np.imgs73=[]
    np.imgs74=[]
    np.imgs75=[]
    np.imgs76=[]
    np.imgs77=[]
    np.imgs78=[]
    np.imgs79=[]
    np.imgs80=[]
    np.imgs81=[]
    np.imgs82=[]
    np.imgs83=[]
    np.imgs84=[]
    np.imgs85=[]
    np.imgs86=[]
    np.imgs87=[]
    np.imgs88=[]
    np.imgs89=[]
    np.imgs90=[]
    np.imgs91=[]
    np.imgs92=[]
    np.imgs93=[]
    np.imgs94=[]
    np.imgs95=[]
    np.imgs96=[]
    np.imgs97=[]
    np.imgs98=[]
    np.imgs99=[]
    np.imgs100=[]
    np.imgs101=[]
    np.imgs102=[]
    np.imgs103=[]
    np.imgs104=[]
    np.imgs105=[]
    np.imgs106=[]
    np.imgs107=[]
    np.imgs108=[]
    np.imgs109=[]
    np.imgs110=[]
    np.imgs111=[]
    np.imgs112=[]
    np.imgs113=[]
    np.imgs114=[]
    np.imgs115=[]
    np.imgs116=[]
    np.imgs117=[]
    np.imgs118=[]
    np.imgs119=[]
    np.imgs120=[]
    np.imgs121=[]
    np.imgs122=[]
    np.imgs123=[]
    np.imgs124=[]
    np.imgs125=[]
    np.imgs126=[]
    np.imgs127=[]
    np.imgs128=[]
    np.imgs129=[]
    np.imgs130=[]
    np.imgs131=[]
    np.imgs132=[]
    np.imgs133=[]
    np.imgs134=[]
    np.imgs135=[]
    np.imgs136=[]
    np.imgs137=[]
    np.imgs138=[]
    np.imgs139=[]
    np.imgs140=[]
    np.imgs141=[]
    np.imgs142=[]
    np.imgs143=[]
    np.imgs144=[]
    np.imgs145=[]
    np.imgs146=[]
    np.imgs147=[]
    np.imgs148=[]
    np.imgs149=[]
    np.imgs150=[]
    np.imgs151=[]
    np.imgs152=[]
    np.imgs153=[]
    np.imgs154=[]
    np.imgs155=[]
    np.imgs156=[]
    np.imgs157=[]
    np.imgs158=[]
    np.imgs159=[]
    np.imgs160=[]
    np.imgs161=[]
    np.imgs162=[]
    np.imgs163=[]
    np.imgs164=[]
    np.imgs165=[]
    np.imgs166=[]
    np.imgs167=[]
    np.imgs168=[]
    np.imgs169=[]
    np.imgs170=[]
    np.imgs171=[]
    np.imgs172=[]
    np.imgs173=[]
    np.imgs174=[]
    np.imgs175=[]
    np.imgs176=[]
    np.imgs177=[]
    np.imgs178=[]
    np.imgs179=[]
    np.imgs180=[]
    np.imgs181=[]
    np.imgs182=[]
    np.imgs183=[]
    np.imgs184=[]
    np.imgs185=[]
    np.imgs186=[]
    np.imgs187=[]
    np.imgs188=[]
    np.imgs189=[]
    np.imgs190=[]
    np.imgs191=[]
    np.imgs192=[]
    np.imgs193=[]
    np.imgs194=[]
    np.imgs195=[]
    np.imgs196=[]
    np.imgs197=[]
    np.imgs198=[]
    np.imgs199=[]
    np.imgs200=[]
    np.imgs201=[]
    np.imgs202=[]
    np.imgs203=[]
    np.imgs204=[]
    np.imgs205=[]
    np.imgs206=[]
    np.imgs207=[]
    np.imgs208=[]
    np.imgs209=[]
    np.imgs210=[]
    np.imgs211=[]
    np.imgs212=[]
    np.imgs213=[]
    np.imgs214=[]
    np.imgs215=[]
    np.imgs216=[]
    np.imgs217=[]
    np.imgs218=[]
    np.imgs219=[]
    np.imgs220=[]
    np.imgs221=[]
    np.imgs222=[]
    np.imgs223=[]
    np.imgs224=[]
    np.imgs225=[]
    np.imgs226=[]
    np.imgs227=[]
    np.imgs228=[]
    np.imgs229=[]
    np.imgs230=[]
    np.imgs231=[]
    np.imgs232=[]
    np.imgs233=[]
    np.imgs234=[]
    np.imgs235=[]
    np.imgs236=[]
    np.imgs237=[]
    np.imgs238=[]
    np.imgs239=[]
    np.imgs240=[]
    np.imgs241=[]
    np.imgs242=[]
    np.imgs243=[]
    np.imgs244=[]
    np.imgs245=[]
    np.imgs246=[]
    np.imgs247=[]
    np.imgs248=[]
    np.imgs249=[]
    np.imgs250=[]
    np.imgs251=[]
    np.imgs252=[]
    np.imgs253=[]
    np.imgs254=[]
    np.imgs255=[]
    np.imgs256=[]
    np.imgs257=[]
    np.imgs258=[]
    np.imgs259=[]
    np.imgs260=[]
    np.imgs261=[]
    np.imgs262=[]
    np.imgs263=[]
    np.imgs264=[]
    np.imgs265=[]
    np.imgs266=[]
    np.imgs267=[]
    np.imgs268=[]
    np.imgs269=[]
    np.imgs270=[]
    np.imgs271=[]
    np.imgs272=[]
    np.imgs273=[]
    np.imgs274=[]
    np.imgs275=[]
    np.imgs276=[]
    np.imgs277=[]
    np.imgs278=[]
    np.imgs279=[]
    np.imgs280=[]
    np.imgs281=[]
    np.imgs282=[]
    np.imgs283=[]
    np.imgs284=[]
    np.imgs285=[]
    np.imgs286=[]
    np.imgs287=[]
    np.imgs288=[]
    np.imgs289=[]
    np.imgs290=[]
    np.imgs291=[]
    np.imgs292=[]
    np.imgs293=[]
    np.imgs294=[]
    np.imgs295=[]
    np.imgs296=[]
    np.imgs297=[]
    np.imgs298=[]
    np.imgs299=[]
    np.imgs300=[]
    
    imgsname = []
    labels = []
            
                
    for i in os.listdir(rootDirName):
        
        label = np.zeros(num_classes) 
        label[int(i)-1] = 1
        idx_to_label.append(i)
        
        for j in os.listdir(os.path.join(rootDirName, i)):
            
            np.imgname = os.listdir(os.path.join(rootDirName, i ,j))
            
            imgsname.append(j)
     
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[0]))
            img = preprocess_image(img) 
            np.imgs1.append(img)
        
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[1]))
            img = preprocess_image(img)
            np.imgs2.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[2]))
            img = preprocess_image(img) 
            np.imgs3.append(img)

            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[3]))
            img = preprocess_image(img) 
            np.imgs4.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[4]))
            img = preprocess_image(img) 
            np.imgs5.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[5]))
            img = preprocess_image(img)
            np.imgs6.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[6]))
            img = preprocess_image(img) 
            np.imgs7.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[7]))
            img = preprocess_image(img) 
            np.imgs8.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[8]))
            img = preprocess_image(img) 
            np.imgs9.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[9]))
            img = preprocess_image(img) 
            np.imgs10.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[10]))
            img = preprocess_image(img) 
            np.imgs11.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[11]))
            img = preprocess_image(img) 
            np.imgs12.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[12]))
            img = preprocess_image(img) 
            np.imgs13.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[13]))
            img = preprocess_image(img) 
            np.imgs14.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[14]))
            img = preprocess_image(img) 
            np.imgs15.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[15]))
            img = preprocess_image(img) 
            np.imgs16.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[16]))
            img = preprocess_image(img) 
            np.imgs17.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[17]))
            img = preprocess_image(img) 
            np.imgs18.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[18]))
            img = preprocess_image(img) 
            np.imgs19.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[19]))
            img = preprocess_image(img)
            np.imgs20.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[20]))
            img = preprocess_image(img) 
            np.imgs21.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[21]))
            img = preprocess_image(img) 
            np.imgs22.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[22]))
            img = preprocess_image(img) 
            np.imgs23.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[23]))
            img = preprocess_image(img) 
            np.imgs24.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[24]))
            img = preprocess_image(img) 
            np.imgs25.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[25]))
            img = preprocess_image(img) 
            np.imgs26.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[26]))
            img = preprocess_image(img) 
            np.imgs27.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[27]))
            img = preprocess_image(img) 
            np.imgs28.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[28]))
            img = preprocess_image(img) 
            np.imgs29.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[29]))
            img = preprocess_image(img) 
            np.imgs30.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[30]))
            img = preprocess_image(img) 
            np.imgs31.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[31]))
            img = preprocess_image(img) 
            np.imgs32.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[32]))
            img = preprocess_image(img) 
            np.imgs33.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[33]))
            img = preprocess_image(img) 
            np.imgs34.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[34]))
            img = preprocess_image(img) 
            np.imgs35.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[35]))
            img = preprocess_image(img) 
            np.imgs36.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[36]))
            img = preprocess_image(img) 
            np.imgs37.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[37]))
            img = preprocess_image(img) 
            np.imgs38.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[38]))
            img = preprocess_image(img) 
            np.imgs39.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[39]))
            img = preprocess_image(img) 
            np.imgs40.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[40]))
            img = preprocess_image(img) 
            np.imgs41.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[41]))
            img = preprocess_image(img) 
            np.imgs42.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[42]))
            img = preprocess_image(img) 
            np.imgs43.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[43]))
            img = preprocess_image(img) 
            np.imgs44.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[44]))
            img = preprocess_image(img) 
            np.imgs45.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[45]))
            img = preprocess_image(img) 
            np.imgs46.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[46]))
            img = preprocess_image(img) 
            np.imgs47.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[47]))
            img = preprocess_image(img) 
            np.imgs48.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[48]))
            img = preprocess_image(img) 
            np.imgs49.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[49]))
            img = preprocess_image(img) 
            np.imgs50.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[50]))
            img = preprocess_image(img) 
            np.imgs51.append(img)
        
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[51]))
            img = preprocess_image(img) 
            np.imgs52.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[52]))
            img = preprocess_image(img) 
            np.imgs53.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[53]))
            img = preprocess_image(img) 
            np.imgs54.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[54]))
            img = preprocess_image(img) 
            np.imgs55.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[55]))
            img = preprocess_image(img) 
            np.imgs56.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[56]))
            img = preprocess_image(img) 
            np.imgs57.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[57]))
            img = preprocess_image(img) 
            np.imgs58.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[58]))
            img = preprocess_image(img) 
            np.imgs59.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[59]))
            img = preprocess_image(img) 
            np.imgs60.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[60]))
            img = preprocess_image(img) 
            np.imgs61.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[61]))
            img = preprocess_image(img) 
            np.imgs62.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[62]))
            img = preprocess_image(img) 
            np.imgs63.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[63]))
            img = preprocess_image(img) 
            np.imgs64.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[64]))
            img = preprocess_image(img) 
            np.imgs65.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[65]))
            img = preprocess_image(img) 
            np.imgs66.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[66]))
            img = preprocess_image(img) 
            np.imgs67.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[67]))
            img = preprocess_image(img) 
            np.imgs68.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[68]))
            img = preprocess_image(img) 
            np.imgs69.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[69]))
            img = preprocess_image(img) 
            np.imgs70.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[70]))
            img = preprocess_image(img) 
            np.imgs71.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[71]))
            img = preprocess_image(img) 
            np.imgs72.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[72]))
            img = preprocess_image(img) 
            np.imgs73.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[73]))
            img = preprocess_image(img) 
            np.imgs74.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[74]))
            img = preprocess_image(img) 
            np.imgs75.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[75]))
            img = preprocess_image(img) 
            np.imgs76.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[76]))
            img = preprocess_image(img) 
            np.imgs77.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[77]))
            img = preprocess_image(img) 
            np.imgs78.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[78]))
            img = preprocess_image(img) 
            np.imgs79.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[79]))
            img = preprocess_image(img) 
            np.imgs80.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[80]))
            img = preprocess_image(img) 
            np.imgs81.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[81]))
            img = preprocess_image(img) 
            np.imgs82.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[82]))
            img = preprocess_image(img) 
            np.imgs83.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[83]))
            img = preprocess_image(img) 
            np.imgs84.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[84]))
            img = preprocess_image(img) 
            np.imgs85.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[85]))
            img = preprocess_image(img) 
            np.imgs86.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[86]))
            img = preprocess_image(img) 
            np.imgs87.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[87]))
            img = preprocess_image(img) 
            np.imgs88.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[88]))
            img = preprocess_image(img) 
            np.imgs89.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[89]))
            img = preprocess_image(img) 
            np.imgs90.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[90]))
            img = preprocess_image(img) 
            np.imgs91.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[91]))
            img = preprocess_image(img) 
            np.imgs92.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[92]))
            img = preprocess_image(img) 
            np.imgs93.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[93]))
            img = preprocess_image(img) 
            np.imgs94.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[94]))
            img = preprocess_image(img) 
            np.imgs95.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[95]))
            img = preprocess_image(img) 
            np.imgs96.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[96]))
            img = preprocess_image(img) 
            np.imgs97.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[97]))
            img = preprocess_image(img) 
            np.imgs98.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[98]))
            img = preprocess_image(img) 
            np.imgs99.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[99]))
            img = preprocess_image(img) 
            np.imgs100.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[100]))
            img = preprocess_image(img) 
            np.imgs101.append(img)
            
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[101]))
            img = preprocess_image(img) 
            np.imgs102.append(img)
        
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[102]))
            img = preprocess_image(img)
            np.imgs103.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[103]))
            img = preprocess_image(img) 
            np.imgs104.append(img)

            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[104]))
            img = preprocess_image(img) 
            np.imgs105.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[105]))
            img = preprocess_image(img) 
            np.imgs106.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[106]))
            img = preprocess_image(img)
            np.imgs107.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[107]))
            img = preprocess_image(img) 
            np.imgs108.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[108]))
            img = preprocess_image(img) 
            np.imgs109.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[109]))
            img = preprocess_image(img) 
            np.imgs110.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[110]))
            img = preprocess_image(img) 
            np.imgs111.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[111]))
            img = preprocess_image(img) 
            np.imgs112.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[112]))
            img = preprocess_image(img) 
            np.imgs113.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[113]))
            img = preprocess_image(img) 
            np.imgs114.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[114]))
            img = preprocess_image(img) 
            np.imgs115.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[115]))
            img = preprocess_image(img) 
            np.imgs116.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[116]))
            img = preprocess_image(img) 
            np.imgs117.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[117]))
            img = preprocess_image(img) 
            np.imgs118.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[118]))
            img = preprocess_image(img) 
            np.imgs119.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[119]))
            img = preprocess_image(img) 
            np.imgs120.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[120]))
            img = preprocess_image(img)
            np.imgs121.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[121]))
            img = preprocess_image(img) 
            np.imgs122.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[122]))
            img = preprocess_image(img) 
            np.imgs123.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[123]))
            img = preprocess_image(img) 
            np.imgs124.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[124]))
            img = preprocess_image(img) 
            np.imgs125.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[125]))
            img = preprocess_image(img) 
            np.imgs126.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[126]))
            img = preprocess_image(img) 
            np.imgs127.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[127]))
            img = preprocess_image(img) 
            np.imgs128.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[128]))
            img = preprocess_image(img) 
            np.imgs129.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[129]))
            img = preprocess_image(img) 
            np.imgs130.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[130]))
            img = preprocess_image(img) 
            np.imgs131.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[131]))
            img = preprocess_image(img) 
            np.imgs132.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[132]))
            img = preprocess_image(img) 
            np.imgs133.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[133]))
            img = preprocess_image(img) 
            np.imgs134.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[134]))
            img = preprocess_image(img) 
            np.imgs135.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[135]))
            img = preprocess_image(img) 
            np.imgs136.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[136]))
            img = preprocess_image(img) 
            np.imgs137.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[137]))
            img = preprocess_image(img) 
            np.imgs138.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[138]))
            img = preprocess_image(img) 
            np.imgs139.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[139]))
            img = preprocess_image(img) 
            np.imgs140.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[140]))
            img = preprocess_image(img) 
            np.imgs141.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[141]))
            img = preprocess_image(img) 
            np.imgs142.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[142]))
            img = preprocess_image(img) 
            np.imgs143.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[143]))
            img = preprocess_image(img) 
            np.imgs144.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[144]))
            img = preprocess_image(img) 
            np.imgs145.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[145]))
            img = preprocess_image(img) 
            np.imgs146.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[146]))
            img = preprocess_image(img) 
            np.imgs147.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[147]))
            img = preprocess_image(img) 
            np.imgs148.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[148]))
            img = preprocess_image(img) 
            np.imgs149.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[149]))
            img = preprocess_image(img) 
            np.imgs150.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[150]))
            img = preprocess_image(img) 
            np.imgs151.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[151]))
            img = preprocess_image(img) 
            np.imgs152.append(img)
        
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[152]))
            img = preprocess_image(img) 
            np.imgs153.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[153]))
            img = preprocess_image(img) 
            np.imgs154.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[154]))
            img = preprocess_image(img) 
            np.imgs155.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[155]))
            img = preprocess_image(img) 
            np.imgs156.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[156]))
            img = preprocess_image(img) 
            np.imgs157.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[157]))
            img = preprocess_image(img) 
            np.imgs158.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[158]))
            img = preprocess_image(img) 
            np.imgs159.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[159]))
            img = preprocess_image(img) 
            np.imgs160.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[160]))
            img = preprocess_image(img) 
            np.imgs161.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[161]))
            img = preprocess_image(img) 
            np.imgs162.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[162]))
            img = preprocess_image(img) 
            np.imgs163.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[163]))
            img = preprocess_image(img) 
            np.imgs164.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[164]))
            img = preprocess_image(img) 
            np.imgs165.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[165]))
            img = preprocess_image(img) 
            np.imgs166.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[166]))
            img = preprocess_image(img) 
            np.imgs167.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[167]))
            img = preprocess_image(img) 
            np.imgs168.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[168]))
            img = preprocess_image(img) 
            np.imgs169.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[169]))
            img = preprocess_image(img) 
            np.imgs170.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[170]))
            img = preprocess_image(img) 
            np.imgs171.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[171]))
            img = preprocess_image(img) 
            np.imgs172.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[172]))
            img = preprocess_image(img) 
            np.imgs173.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[173]))
            img = preprocess_image(img) 
            np.imgs174.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[174]))
            img = preprocess_image(img) 
            np.imgs175.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[175]))
            img = preprocess_image(img) 
            np.imgs176.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[176]))
            img = preprocess_image(img) 
            np.imgs177.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[177]))
            img = preprocess_image(img) 
            np.imgs178.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[178]))
            img = preprocess_image(img) 
            np.imgs179.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[179]))
            img = preprocess_image(img) 
            np.imgs180.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[180]))
            img = preprocess_image(img) 
            np.imgs181.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[181]))
            img = preprocess_image(img) 
            np.imgs182.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[182]))
            img = preprocess_image(img) 
            np.imgs183.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[183]))
            img = preprocess_image(img) 
            np.imgs184.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[184]))
            img = preprocess_image(img) 
            np.imgs185.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[185]))
            img = preprocess_image(img) 
            np.imgs186.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[186]))
            img = preprocess_image(img) 
            np.imgs187.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[187]))
            img = preprocess_image(img) 
            np.imgs188.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[188]))
            img = preprocess_image(img) 
            np.imgs189.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[189]))
            img = preprocess_image(img) 
            np.imgs190.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[190]))
            img = preprocess_image(img) 
            np.imgs191.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[191]))
            img = preprocess_image(img) 
            np.imgs192.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[192]))
            img = preprocess_image(img) 
            np.imgs193.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[193]))
            img = preprocess_image(img) 
            np.imgs194.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[194]))
            img = preprocess_image(img) 
            np.imgs195.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[195]))
            img = preprocess_image(img) 
            np.imgs196.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[196]))
            img = preprocess_image(img) 
            np.imgs197.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[197]))
            img = preprocess_image(img) 
            np.imgs198.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[198]))
            img = preprocess_image(img) 
            np.imgs199.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[199]))
            img = preprocess_image(img) 
            np.imgs200.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[200]))
            img = preprocess_image(img) 
            np.imgs201.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[201]))
            img = preprocess_image(img) 
            np.imgs202.append(img)
            
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[202]))
            img = preprocess_image(img) 
            np.imgs203.append(img)
        
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[203]))
            img = preprocess_image(img)
            np.imgs204.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[204]))
            img = preprocess_image(img) 
            np.imgs205.append(img)

            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[205]))
            img = preprocess_image(img) 
            np.imgs206.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[206]))
            img = preprocess_image(img) 
            np.imgs207.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[207]))
            img = preprocess_image(img)
            np.imgs208.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[208]))
            img = preprocess_image(img) 
            np.imgs209.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[209]))
            img = preprocess_image(img) 
            np.imgs210.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[210]))
            img = preprocess_image(img) 
            np.imgs211.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[211]))
            img = preprocess_image(img) 
            np.imgs212.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[212]))
            img = preprocess_image(img) 
            np.imgs213.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[213]))
            img = preprocess_image(img) 
            np.imgs214.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[214]))
            img = preprocess_image(img) 
            np.imgs215.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[215]))
            img = preprocess_image(img) 
            np.imgs216.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[216]))
            img = preprocess_image(img) 
            np.imgs217.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[217]))
            img = preprocess_image(img) 
            np.imgs218.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[218]))
            img = preprocess_image(img) 
            np.imgs219.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[219]))
            img = preprocess_image(img) 
            np.imgs220.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[220]))
            img = preprocess_image(img) 
            np.imgs221.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[221]))
            img = preprocess_image(img)
            np.imgs222.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[222]))
            img = preprocess_image(img) 
            np.imgs223.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[223]))
            img = preprocess_image(img) 
            np.imgs224.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[224]))
            img = preprocess_image(img) 
            np.imgs225.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[225]))
            img = preprocess_image(img) 
            np.imgs226.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[226]))
            img = preprocess_image(img) 
            np.imgs227.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[227]))
            img = preprocess_image(img) 
            np.imgs228.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[228]))
            img = preprocess_image(img) 
            np.imgs229.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[229]))
            img = preprocess_image(img) 
            np.imgs230.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[230]))
            img = preprocess_image(img) 
            np.imgs231.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[231]))
            img = preprocess_image(img) 
            np.imgs232.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[232]))
            img = preprocess_image(img) 
            np.imgs233.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[233]))
            img = preprocess_image(img) 
            np.imgs234.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[234]))
            img = preprocess_image(img) 
            np.imgs235.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[235]))
            img = preprocess_image(img) 
            np.imgs236.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[236]))
            img = preprocess_image(img) 
            np.imgs237.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[237]))
            img = preprocess_image(img) 
            np.imgs238.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[238]))
            img = preprocess_image(img) 
            np.imgs239.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[239]))
            img = preprocess_image(img) 
            np.imgs240.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[240]))
            img = preprocess_image(img) 
            np.imgs241.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[241]))
            img = preprocess_image(img) 
            np.imgs242.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[242]))
            img = preprocess_image(img) 
            np.imgs243.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[243]))
            img = preprocess_image(img) 
            np.imgs244.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[244]))
            img = preprocess_image(img) 
            np.imgs245.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[245]))
            img = preprocess_image(img) 
            np.imgs246.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[246]))
            img = preprocess_image(img) 
            np.imgs247.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[247]))
            img = preprocess_image(img) 
            np.imgs248.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[248]))
            img = preprocess_image(img) 
            np.imgs249.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[249]))
            img = preprocess_image(img) 
            np.imgs250.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[250]))
            img = preprocess_image(img) 
            np.imgs251.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[251]))
            img = preprocess_image(img) 
            np.imgs252.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[252]))
            img = preprocess_image(img) 
            np.imgs253.append(img)
        
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[253]))
            img = preprocess_image(img) 
            np.imgs254.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[254]))
            img = preprocess_image(img) 
            np.imgs255.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[255]))
            img = preprocess_image(img) 
            np.imgs256.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[256]))
            img = preprocess_image(img) 
            np.imgs257.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[257]))
            img = preprocess_image(img) 
            np.imgs258.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[258]))
            img = preprocess_image(img) 
            np.imgs259.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[259]))
            img = preprocess_image(img) 
            np.imgs260.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[260]))
            img = preprocess_image(img) 
            np.imgs261.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[261]))
            img = preprocess_image(img) 
            np.imgs262.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[262]))
            img = preprocess_image(img) 
            np.imgs263.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[263]))
            img = preprocess_image(img) 
            np.imgs264.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[264]))
            img = preprocess_image(img) 
            np.imgs265.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[265]))
            img = preprocess_image(img) 
            np.imgs266.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[266]))
            img = preprocess_image(img) 
            np.imgs267.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[267]))
            img = preprocess_image(img) 
            np.imgs268.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[268]))
            img = preprocess_image(img) 
            np.imgs269.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[269]))
            img = preprocess_image(img) 
            np.imgs270.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[270]))
            img = preprocess_image(img) 
            np.imgs271.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[271]))
            img = preprocess_image(img) 
            np.imgs272.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[272]))
            img = preprocess_image(img) 
            np.imgs273.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[273]))
            img = preprocess_image(img) 
            np.imgs274.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[274]))
            img = preprocess_image(img) 
            np.imgs275.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[275]))
            img = preprocess_image(img) 
            np.imgs276.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[276]))
            img = preprocess_image(img) 
            np.imgs277.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[277]))
            img = preprocess_image(img) 
            np.imgs278.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[278]))
            img = preprocess_image(img) 
            np.imgs279.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[279]))
            img = preprocess_image(img) 
            np.imgs280.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[280]))
            img = preprocess_image(img) 
            np.imgs281.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[281]))
            img = preprocess_image(img) 
            np.imgs282.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[282]))
            img = preprocess_image(img) 
            np.imgs283.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[283]))
            img = preprocess_image(img) 
            np.imgs284.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[284]))
            img = preprocess_image(img) 
            np.imgs285.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[285]))
            img = preprocess_image(img) 
            np.imgs286.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[286]))
            img = preprocess_image(img) 
            np.imgs287.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[287]))
            img = preprocess_image(img) 
            np.imgs288.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[288]))
            img = preprocess_image(img) 
            np.imgs289.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[289]))
            img = preprocess_image(img) 
            np.imgs290.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[290]))
            img = preprocess_image(img) 
            np.imgs291.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[291]))
            img = preprocess_image(img) 
            np.imgs292.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[292]))
            img = preprocess_image(img) 
            np.imgs293.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[293]))
            img = preprocess_image(img) 
            np.imgs294.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[294]))
            img = preprocess_image(img) 
            np.imgs295.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[295]))
            img = preprocess_image(img) 
            np.imgs296.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[296]))
            img = preprocess_image(img) 
            np.imgs297.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[297]))
            img = preprocess_image(img) 
            np.imgs298.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[298]))
            img = preprocess_image(img) 
            np.imgs299.append(img)
            
            img = tf.io.read_file(os.path.join(rootDirName, i, j, np.imgname[299]))
            img = preprocess_image(img) 
            np.imgs300.append(img)
            
            print(j)
            
            labels.append(label)
            
            
    return np.array(np.imgs1), np.array(np.imgs2), np.array(np.imgs3), np.array(np.imgs4), np.array(np.imgs5), np.array(np.imgs6), np.array(np.imgs7), np.array(np.imgs8), np.array(np.imgs9), np.array(np.imgs10), np.array(np.imgs11), np.array(np.imgs12), np.array(np.imgs13), np.array(np.imgs14), np.array(np.imgs15), np.array(np.imgs16), np.array(np.imgs17), np.array(np.imgs18), np.array(np.imgs19), np.array(np.imgs20), np.array(np.imgs21), np.array(np.imgs22), np.array(np.imgs23), np.array(np.imgs24), np.array(np.imgs25), np.array(np.imgs26), np.array(np.imgs27), np.array(np.imgs28), np.array(np.imgs29), np.array(np.imgs30), np.array(np.imgs31), np.array(np.imgs32), np.array(np.imgs33), np.array(np.imgs34), np.array(np.imgs35), np.array(np.imgs36), np.array(np.imgs37), np.array(np.imgs38), np.array(np.imgs39), np.array(np.imgs40), np.array(np.imgs41), np.array(np.imgs42), np.array(np.imgs43), np.array(np.imgs44), np.array(np.imgs45), np.array(np.imgs46), np.array(np.imgs47), np.array(np.imgs48), np.array(np.imgs49), np.array(np.imgs50), np.array(np.imgs51), np.array(np.imgs52), np.array(np.imgs53), np.array(np.imgs54), np.array(np.imgs55), np.array(np.imgs56), np.array(np.imgs57), np.array(np.imgs58), np.array(np.imgs59), np.array(np.imgs60), np.array(np.imgs61), np.array(np.imgs62), np.array(np.imgs63), np.array(np.imgs64), np.array(np.imgs65), np.array(np.imgs66), np.array(np.imgs67), np.array(np.imgs68), np.array(np.imgs69), np.array(np.imgs70), np.array(np.imgs71), np.array(np.imgs72), np.array(np.imgs73), np.array(np.imgs74), np.array(np.imgs75), np.array(np.imgs76), np.array(np.imgs77), np.array(np.imgs78), np.array(np.imgs79), np.array(np.imgs80), np.array(np.imgs81), np.array(np.imgs82), np.array(np.imgs83), np.array(np.imgs84), np.array(np.imgs85), np.array(np.imgs86), np.array(np.imgs87), np.array(np.imgs88), np.array(np.imgs89), np.array(np.imgs90), np.array(np.imgs91), np.array(np.imgs92), np.array(np.imgs93), np.array(np.imgs94), np.array(np.imgs95), np.array(np.imgs96), np.array(np.imgs97), np.array(np.imgs98), np.array(np.imgs99), np.array(np.imgs100), np.array(np.imgs101), np.array(np.imgs102), np.array(np.imgs103), np.array(np.imgs104), np.array(np.imgs105), np.array(np.imgs106), np.array(np.imgs107), np.array(np.imgs108), np.array(np.imgs109), np.array(np.imgs110), np.array(np.imgs111), np.array(np.imgs112), np.array(np.imgs113), np.array(np.imgs114), np.array(np.imgs115), np.array(np.imgs116), np.array(np.imgs117), np.array(np.imgs118), np.array(np.imgs119), np.array(np.imgs120), np.array(np.imgs121), np.array(np.imgs122), np.array(np.imgs123), np.array(np.imgs124), np.array(np.imgs125), np.array(np.imgs126), np.array(np.imgs127), np.array(np.imgs128), np.array(np.imgs129), np.array(np.imgs130), np.array(np.imgs131), np.array(np.imgs132), np.array(np.imgs133), np.array(np.imgs134), np.array(np.imgs135), np.array(np.imgs136), np.array(np.imgs137), np.array(np.imgs138), np.array(np.imgs139), np.array(np.imgs140), np.array(np.imgs141), np.array(np.imgs142), np.array(np.imgs143), np.array(np.imgs144), np.array(np.imgs145), np.array(np.imgs146), np.array(np.imgs147), np.array(np.imgs148), np.array(np.imgs149), np.array(np.imgs150), np.array(np.imgs151), np.array(np.imgs152), np.array(np.imgs153), np.array(np.imgs154), np.array(np.imgs155), np.array(np.imgs156), np.array(np.imgs157), np.array(np.imgs158), np.array(np.imgs159), np.array(np.imgs160), np.array(np.imgs161), np.array(np.imgs162), np.array(np.imgs163), np.array(np.imgs164), np.array(np.imgs165), np.array(np.imgs166), np.array(np.imgs167), np.array(np.imgs168), np.array(np.imgs169), np.array(np.imgs170), np.array(np.imgs171), np.array(np.imgs172), np.array(np.imgs173), np.array(np.imgs174), np.array(np.imgs175), np.array(np.imgs176), np.array(np.imgs177), np.array(np.imgs178), np.array(np.imgs179), np.array(np.imgs180), np.array(np.imgs181), np.array(np.imgs182), np.array(np.imgs183), np.array(np.imgs184), np.array(np.imgs185), np.array(np.imgs186), np.array(np.imgs187), np.array(np.imgs188), np.array(np.imgs189), np.array(np.imgs190), np.array(np.imgs191), np.array(np.imgs192), np.array(np.imgs193), np.array(np.imgs194), np.array(np.imgs195), np.array(np.imgs196), np.array(np.imgs197), np.array(np.imgs198), np.array(np.imgs199), np.array(np.imgs200), np.array(np.imgs201), np.array(np.imgs202), np.array(np.imgs203), np.array(np.imgs204), np.array(np.imgs205), np.array(np.imgs206), np.array(np.imgs207), np.array(np.imgs208), np.array(np.imgs209), np.array(np.imgs210), np.array(np.imgs211), np.array(np.imgs212), np.array(np.imgs213), np.array(np.imgs214), np.array(np.imgs215), np.array(np.imgs216), np.array(np.imgs217), np.array(np.imgs218), np.array(np.imgs219), np.array(np.imgs220), np.array(np.imgs221), np.array(np.imgs222), np.array(np.imgs223), np.array(np.imgs224), np.array(np.imgs225), np.array(np.imgs226), np.array(np.imgs227), np.array(np.imgs228), np.array(np.imgs229), np.array(np.imgs230), np.array(np.imgs231), np.array(np.imgs232), np.array(np.imgs233), np.array(np.imgs234), np.array(np.imgs235), np.array(np.imgs236), np.array(np.imgs237), np.array(np.imgs238), np.array(np.imgs239), np.array(np.imgs240), np.array(np.imgs241), np.array(np.imgs242), np.array(np.imgs243), np.array(np.imgs244), np.array(np.imgs245), np.array(np.imgs246), np.array(np.imgs247), np.array(np.imgs248), np.array(np.imgs249), np.array(np.imgs250), np.array(np.imgs251), np.array(np.imgs252), np.array(np.imgs253), np.array(np.imgs254), np.array(np.imgs255), np.array(np.imgs256), np.array(np.imgs257), np.array(np.imgs258), np.array(np.imgs259), np.array(np.imgs260), np.array(np.imgs261), np.array(np.imgs262), np.array(np.imgs263), np.array(np.imgs264), np.array(np.imgs265), np.array(np.imgs266), np.array(np.imgs267), np.array(np.imgs268), np.array(np.imgs269), np.array(np.imgs270), np.array(np.imgs271), np.array(np.imgs272), np.array(np.imgs273), np.array(np.imgs274), np.array(np.imgs275), np.array(np.imgs276), np.array(np.imgs277), np.array(np.imgs278), np.array(np.imgs279), np.array(np.imgs280), np.array(np.imgs281), np.array(np.imgs282), np.array(np.imgs283), np.array(np.imgs284), np.array(np.imgs285), np.array(np.imgs286), np.array(np.imgs287), np.array(np.imgs288), np.array(np.imgs289), np.array(np.imgs290), np.array(np.imgs291), np.array(np.imgs292), np.array(np.imgs293), np.array(np.imgs294), np.array(np.imgs295), np.array(np.imgs296), np.array(np.imgs297), np.array(np.imgs298), np.array(np.imgs299), np.array(np.imgs300), np.array(imgsname), np.array(labels)
    

X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16, X17, X18, X19, X20, X21, X22, X23, X24, X25, X26, X27, X28, X29, X30, X31, X32, X33, X34, X35, X36, X37, X38, X39, X40, X41, X42, X43, X44, X45, X46, X47, X48, X49, X50, X51, X52, X53, X54, X55, X56, X57, X58, X59, X60, X61, X62, X63, X64, X65, X66, X67, X68, X69, X70, X71, X72, X73, X74, X75, X76, X77, X78, X79, X80, X81, X82, X83, X84, X85, X86, X87, X88, X89, X90, X91, X92, X93, X94, X95, X96, X97, X98, X99, X100, X101, X102, X103, X104, X105, X106, X107, X108, X109, X110, X111, X112, X113, X114, X115, X116, X117, X118, X119, X120, X121, X122, X123, X124, X125, X126, X127, X128, X129, X130, X131, X132, X133, X134, X135, X136, X137, X138, X139, X140, X141, X142, X143, X144, X145, X146, X147, X148, X149, X150, X151, X152, X153, X154, X155, X156, X157, X158, X159, X160, X161, X162, X163, X164, X165, X166, X167, X168, X169, X170, X171, X172, X173, X174, X175, X176, X177, X178, X179, X180, X181, X182, X183, X184, X185, X186, X187, X188, X189, X190, X191, X192, X193, X194, X195, X196, X197, X198, X199, X200, X201, X202, X203, X204, X205, X206, X207, X208, X209, X210, X211, X212, X213, X214, X215, X216, X217, X218, X219, X220, X221, X222, X223, X224, X225, X226, X227, X228, X229, X230, X231, X232, X233, X234, X235, X236, X237, X238, X239, X240, X241, X242, X243, X244, X245, X246, X247, X248, X249, X250, X251, X252, X253, X254, X255, X256, X257, X258, X259, X260, X261, X262, X263, X264, X265, X266, X267, X268, X269, X270, X271, X272, X273, X274, X275, X276, X277, X278, X279, X280, X281, X282, X283, X284, X285, X286, X287, X288, X289, X290, X291, X292, X293, X294, X295, X296, X297, X298, X299, X300, imgsname, Y = make_dataset()

# 学習データとテストデータに分割

x1_train,   x1_test,   x2_train,   x2_test,   x3_train,   x3_test,   x4_train,   x4_test,   x5_train,   x5_test,   x6_train,   x6_test,   x7_train,   x7_test,   x8_train,   x8_test,   x9_train,   x9_test,  x10_train,  x10_test, x11_train,  x11_test,  x12_train,  x12_test,  x13_train,  x13_test,  x14_train,  x14_test,  x15_train,  x15_test,  x16_train,  x16_test,  x17_train,  x17_test,  x18_train,  x18_test,  x19_train,  x19_test,  x20_train,  x20_test, x21_train,  x21_test,  x22_train,  x22_test,  x23_train,  x23_test,  x24_train,  x24_test,  x25_train,  x25_test,  x26_train,  x26_test,  x27_train,  x27_test,  x28_train,  x28_test,  x29_train,  x29_test,  x30_train,  x30_test, x31_train,  x31_test,  x32_train,  x32_test,  x33_train,  x33_test,  x34_train,  x34_test,  x35_train,  x35_test,  x36_train,  x36_test,  x37_train,  x37_test,  x38_train,  x38_test,  x39_train,  x39_test,  x40_train,  x40_test, x41_train,  x41_test,  x42_train,  x42_test,  x43_train,  x43_test,  x44_train,  x44_test,  x45_train,  x45_test,  x46_train,  x46_test,  x47_train,  x47_test,  x48_train,  x48_test,  x49_train,  x49_test,  x50_train,  x50_test, x51_train,  x51_test,  x52_train,  x52_test,  x53_train,  x53_test,  x54_train,  x54_test,  x55_train,  x55_test,  x56_train,  x56_test,  x57_train,  x57_test,  x58_train,  x58_test,  x59_train,  x59_test,  x60_train,  x60_test, x61_train,  x61_test,  x62_train,  x62_test,  x63_train,  x63_test,  x64_train,  x64_test,  x65_train,  x65_test,  x66_train,  x66_test,  x67_train,  x67_test,  x68_train,  x68_test,  x69_train,  x69_test,  x70_train,  x70_test, x71_train,  x71_test,  x72_train,  x72_test,  x73_train,  x73_test,  x74_train,  x74_test,  x75_train,  x75_test,  x76_train,  x76_test,  x77_train,  x77_test,  x78_train,  x78_test,  x79_train,  x79_test,  x80_train,  x80_test, x81_train,  x81_test,  x82_train,  x82_test,  x83_train,  x83_test,  x84_train,  x84_test,  x85_train,  x85_test,  x86_train,  x86_test,  x87_train,  x87_test,  x88_train,  x88_test,  x89_train,  x89_test,  x90_train,  x90_test, x91_train,  x91_test,  x92_train,  x92_test,  x93_train,  x93_test,  x94_train,  x94_test,  x95_train,  x95_test,  x96_train,  x96_test,  x97_train,  x97_test,  x98_train,  x98_test,  x99_train,  x99_test, x100_train, x100_test, x101_train, x101_test, x102_train, x102_test, x103_train, x103_test, x104_train, x104_test, x105_train, x105_test, x106_train, x106_test, x107_train, x107_test, x108_train, x108_test, x109_train, x109_test, x110_train, x110_test, x111_train, x111_test, x112_train, x112_test, x113_train, x113_test, x114_train, x114_test, x115_train, x115_test, x116_train, x116_test, x117_train, x117_test, x118_train, x118_test, x119_train, x119_test, x120_train, x120_test, x121_train, x121_test, x122_train, x122_test, x123_train, x123_test, x124_train, x124_test, x125_train, x125_test, x126_train, x126_test, x127_train, x127_test, x128_train, x128_test, x129_train, x129_test, x130_train, x130_test, x131_train, x131_test, x132_train, x132_test, x133_train, x133_test, x134_train, x134_test, x135_train, x135_test, x136_train, x136_test, x137_train, x137_test, x138_train, x138_test, x139_train, x139_test, x140_train, x140_test, x141_train, x141_test, x142_train, x142_test, x143_train, x143_test, x144_train, x144_test, x145_train, x145_test, x146_train, x146_test, x147_train, x147_test, x148_train, x148_test, x149_train, x149_test, x150_train, x150_test, x151_train, x151_test, x152_train, x152_test, x153_train, x153_test, x154_train, x154_test, x155_train, x155_test, x156_train, x156_test, x157_train, x157_test, x158_train, x158_test, x159_train, x159_test, x160_train, x160_test, x161_train, x161_test, x162_train, x162_test, x163_train, x163_test, x164_train, x164_test, x165_train, x165_test, x166_train, x166_test, x167_train, x167_test, x168_train, x168_test, x169_train, x169_test, x170_train, x170_test, x171_train, x171_test, x172_train, x172_test, x173_train, x173_test, x174_train, x174_test, x175_train, x175_test, x176_train, x176_test, x177_train, x177_test, x178_train, x178_test, x179_train, x179_test, x180_train, x180_test, x181_train, x181_test, x182_train, x182_test, x183_train, x183_test, x184_train, x184_test, x185_train, x185_test, x186_train, x186_test, x187_train, x187_test, x188_train, x188_test, x189_train, x189_test, x190_train, x190_test, x191_train, x191_test, x192_train, x192_test, x193_train, x193_test, x194_train, x194_test, x195_train, x195_test, x196_train, x196_test, x197_train, x197_test, x198_train, x198_test, x199_train, x199_test, x200_train, x200_test, x201_train, x201_test, x202_train, x202_test, x203_train, x203_test, x204_train, x204_test, x205_train, x205_test, x206_train, x206_test, x207_train, x207_test, x208_train, x208_test, x209_train, x209_test, x210_train, x210_test, x211_train, x211_test, x212_train, x212_test, x213_train, x213_test, x214_train, x214_test, x215_train, x215_test, x216_train, x216_test, x217_train, x217_test, x218_train, x218_test, x219_train, x219_test, x220_train, x220_test, x221_train, x221_test, x222_train, x222_test, x223_train, x223_test, x224_train, x224_test, x225_train, x225_test, x226_train, x226_test, x227_train, x227_test, x228_train, x228_test, x229_train, x229_test, x230_train, x230_test, x231_train, x231_test, x232_train, x232_test, x233_train, x233_test, x234_train, x234_test, x235_train, x235_test, x236_train, x236_test, x237_train, x237_test, x238_train, x238_test, x239_train, x239_test, x240_train, x240_test, x241_train, x241_test, x242_train, x242_test, x243_train, x243_test, x244_train, x244_test, x245_train, x245_test, x246_train, x246_test, x247_train, x247_test, x248_train, x248_test, x249_train, x249_test, x250_train, x250_test, x251_train, x251_test, x252_train, x252_test, x253_train, x253_test, x254_train, x254_test, x255_train, x255_test, x256_train, x256_test, x257_train, x257_test, x258_train, x258_test, x259_train, x259_test, x260_train, x260_test, x261_train, x261_test, x262_train, x262_test, x263_train, x263_test, x264_train, x264_test, x265_train, x265_test, x266_train, x266_test, x267_train, x267_test, x268_train, x268_test, x269_train, x269_test, x270_train, x270_test, x271_train, x271_test, x272_train, x272_test, x273_train, x273_test, x274_train, x274_test, x275_train, x275_test, x276_train, x276_test, x277_train, x277_test, x278_train, x278_test, x279_train, x279_test, x280_train, x280_test, x281_train, x281_test, x282_train, x282_test, x283_train, x283_test, x284_train, x284_test, x285_train, x285_test, x286_train, x286_test, x287_train, x287_test, x288_train, x288_test, x289_train, x289_test, x290_train, x290_test, x291_train, x291_test, x292_train, x292_test, x293_train, x293_test, x294_train, x294_test, x295_train, x295_test, x296_train, x296_test, x297_train, x297_test, x298_train, x298_test, x299_train, x299_test, x300_train, x300_test, name_train, name_test, y_train, y_test = train_test_split(  X1,   X2,   X3,   X4,   X5,   X6,   X7,   X8,   X9,  X10,
                                    X11,  X12,  X13,  X14,  X15,  X16,  X17,  X18,  X19,  X20,
                                    X21,  X22,  X23,  X24,  X25,  X26,  X27,  X28,  X29,  X30,
                                    X31,  X32,  X33,  X34,  X35,  X36,  X37,  X38,  X39,  X40,
                                    X41,  X42,  X43,  X44,  X45,  X46,  X47,  X48,  X49,  X50,
                                    X51,  X52,  X53,  X54,  X55,  X56,  X57,  X58,  X59,  X60,
                                    X61,  X62,  X63,  X64,  X65,  X66,  X67,  X68,  X69,  X70,
                                    X71,  X72,  X73,  X74,  X75,  X76,  X77,  X78,  X79,  X80,
                                    X81,  X82,  X83,  X84,  X85,  X86,  X87,  X88,  X89,  X90,
                                    X91,  X92,  X93,  X94,  X95,  X96,  X97,  X98,  X99, X100,
                                   X101, X102, X103, X104, X105, X106, X107, X108, X109, X110,
                                   X111, X112, X113, X114, X115, X116, X117, X118, X119, X120,
                                   X121, X122, X123, X124, X125, X126, X127, X128, X129, X130,
                                   X131, X132, X133, X134, X135, X136, X137, X138, X139, X140,
                                   X141, X142, X143, X144, X145, X146, X147, X148, X149, X150,
                                   X151, X152, X153, X154, X155, X156, X157, X158, X159, X160,
                                   X161, X162, X163, X164, X165, X166, X167, X168, X169, X170,
                                   X171, X172, X173, X174, X175, X176, X177, X178, X179, X180,
                                   X181, X182, X183, X184, X185, X186, X187, X188, X189, X190,
                                   X191, X192, X193, X194, X195, X196, X197, X198, X199, X200,
                                   X201, X202, X203, X204, X205, X206, X207, X208, X209, X210,
                                   X211, X212, X213, X214, X215, X216, X217, X218, X219, X220,
                                   X221, X222, X223, X224, X225, X226, X227, X228, X229, X230,
                                   X231, X232, X233, X234, X235, X236, X237, X238, X239, X240,
                                   X241, X242, X243, X244, X245, X246, X247, X248, X249, X250,
                                   X251, X252, X253, X254, X255, X256, X257, X258, X259, X260,
                                   X261, X262, X263, X264, X265, X266, X267, X268, X269, X270,
                                   X271, X272, X273, X274, X275, X276, X277, X278, X279, X280,
                                   X281, X282, X283, X284, X285, X286, X287, X288, X289, X290,
                                   X291, X292, X293, X294, X295, X296, X297, X298, X299, X300, 
                                   imgsname, Y, train_size=0.8, random_state=0)


# In[6]:


#ここでそれぞれのs
print(x1_train.shape)
print(x2_train.shape)
print(x1_test.shape)
print(x2_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[7]:


#!pip install tensorflow
#!pip install keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing import image
from keras.utils.vis_utils import plot_model

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




inputs1 = keras.Input((height, width, 3), name="input1")
inputs2 = keras.Input((height, width, 3), name="input2")
inputs3 = keras.Input((height, width, 3), name="input3")
inputs4 = keras.Input((height, width, 3), name="input4")
inputs5 = keras.Input((height, width, 3), name="input5")
inputs6 = keras.Input((height, width, 3), name="input6")
inputs7 = keras.Input((height, width, 3), name="input7")
inputs8 = keras.Input((height, width, 3), name="input8")
inputs9 = keras.Input((height, width, 3), name="input9")
inputs10 = keras.Input((height, width, 3), name="input10")

inputs11 = keras.Input((height, width, 3), name="input11")
inputs12 = keras.Input((height, width, 3), name="input12")
inputs13 = keras.Input((height, width, 3), name="input13")
inputs14 = keras.Input((height, width, 3), name="input14")
inputs15 = keras.Input((height, width, 3), name="input15")
inputs16 = keras.Input((height, width, 3), name="input16")
inputs17 = keras.Input((height, width, 3), name="input17")
inputs18 = keras.Input((height, width, 3), name="input18")
inputs19 = keras.Input((height, width, 3), name="input19")
inputs20 = keras.Input((height, width, 3), name="input20")

inputs21 = keras.Input((height, width, 3), name="input21")
inputs22 = keras.Input((height, width, 3), name="input22")
inputs23 = keras.Input((height, width, 3), name="input23")
inputs24 = keras.Input((height, width, 3), name="input24")
inputs25 = keras.Input((height, width, 3), name="input25")
inputs26 = keras.Input((height, width, 3), name="input26")
inputs27 = keras.Input((height, width, 3), name="input27")
inputs28 = keras.Input((height, width, 3), name="input28")
inputs29 = keras.Input((height, width, 3), name="input29")
inputs30 = keras.Input((height, width, 3), name="input30")

inputs31 = keras.Input((height, width, 3), name="input31")
inputs32 = keras.Input((height, width, 3), name="input32")
inputs33 = keras.Input((height, width, 3), name="input33")
inputs34 = keras.Input((height, width, 3), name="input34")
inputs35 = keras.Input((height, width, 3), name="input35")
inputs36 = keras.Input((height, width, 3), name="input36")
inputs37 = keras.Input((height, width, 3), name="input37")
inputs38 = keras.Input((height, width, 3), name="input38")
inputs39 = keras.Input((height, width, 3), name="input39")
inputs40 = keras.Input((height, width, 3), name="input40")

inputs41 = keras.Input((height, width, 3), name="input41")
inputs42 = keras.Input((height, width, 3), name="input42")
inputs43 = keras.Input((height, width, 3), name="input43")
inputs44 = keras.Input((height, width, 3), name="input44")
inputs45 = keras.Input((height, width, 3), name="input45")
inputs46 = keras.Input((height, width, 3), name="input46")
inputs47 = keras.Input((height, width, 3), name="input47")
inputs48 = keras.Input((height, width, 3), name="input48")
inputs49 = keras.Input((height, width, 3), name="input49")
inputs50 = keras.Input((height, width, 3), name="input50")

inputs51 = keras.Input((height, width, 3), name="input51")
inputs52 = keras.Input((height, width, 3), name="input52")
inputs53 = keras.Input((height, width, 3), name="input53")
inputs54 = keras.Input((height, width, 3), name="input54")
inputs55 = keras.Input((height, width, 3), name="input55")
inputs56 = keras.Input((height, width, 3), name="input56")
inputs57 = keras.Input((height, width, 3), name="input57")
inputs58 = keras.Input((height, width, 3), name="input58")
inputs59 = keras.Input((height, width, 3), name="input59")
inputs60 = keras.Input((height, width, 3), name="input60")

inputs61 = keras.Input((height, width, 3), name="input61")
inputs62 = keras.Input((height, width, 3), name="input62")
inputs63 = keras.Input((height, width, 3), name="input63")
inputs64 = keras.Input((height, width, 3), name="input64")
inputs65 = keras.Input((height, width, 3), name="input65")
inputs66 = keras.Input((height, width, 3), name="input66")
inputs67 = keras.Input((height, width, 3), name="input67")
inputs68 = keras.Input((height, width, 3), name="input68")
inputs69 = keras.Input((height, width, 3), name="input69")
inputs70 = keras.Input((height, width, 3), name="input70")

inputs71 = keras.Input((height, width, 3), name="input71")
inputs72 = keras.Input((height, width, 3), name="input72")
inputs73 = keras.Input((height, width, 3), name="input73")
inputs74 = keras.Input((height, width, 3), name="input74")
inputs75 = keras.Input((height, width, 3), name="input75")
inputs76 = keras.Input((height, width, 3), name="input76")
inputs77 = keras.Input((height, width, 3), name="input77")
inputs78 = keras.Input((height, width, 3), name="input78")
inputs79 = keras.Input((height, width, 3), name="input79")
inputs80 = keras.Input((height, width, 3), name="input80")

inputs81 = keras.Input((height, width, 3), name="input81")
inputs82 = keras.Input((height, width, 3), name="input82")
inputs83 = keras.Input((height, width, 3), name="input83")
inputs84 = keras.Input((height, width, 3), name="input84")
inputs85 = keras.Input((height, width, 3), name="input85")
inputs86 = keras.Input((height, width, 3), name="input86")
inputs87 = keras.Input((height, width, 3), name="input87")
inputs88 = keras.Input((height, width, 3), name="input88")
inputs89 = keras.Input((height, width, 3), name="input89")
inputs90 = keras.Input((height, width, 3), name="input90")

inputs91 = keras.Input((height, width, 3), name="input91")
inputs92 = keras.Input((height, width, 3), name="input92")
inputs93 = keras.Input((height, width, 3), name="input93")
inputs94 = keras.Input((height, width, 3), name="input94")
inputs95 = keras.Input((height, width, 3), name="input95")
inputs96 = keras.Input((height, width, 3), name="input96")
inputs97 = keras.Input((height, width, 3), name="input97")
inputs98 = keras.Input((height, width, 3), name="input98")
inputs99 = keras.Input((height, width, 3), name="input99")
inputs100 = keras.Input((height, width, 3), name="input100")

inputs101 = keras.Input((height, width, 3), name="input101")
inputs102 = keras.Input((height, width, 3), name="input102")
inputs103 = keras.Input((height, width, 3), name="input103")
inputs104 = keras.Input((height, width, 3), name="input104")
inputs105 = keras.Input((height, width, 3), name="input105")
inputs106 = keras.Input((height, width, 3), name="input106")
inputs107 = keras.Input((height, width, 3), name="input107")
inputs108 = keras.Input((height, width, 3), name="input108")
inputs109 = keras.Input((height, width, 3), name="input109")
inputs110 = keras.Input((height, width, 3), name="input110")

inputs111 = keras.Input((height, width, 3), name="input111")
inputs112 = keras.Input((height, width, 3), name="input112")
inputs113 = keras.Input((height, width, 3), name="input113")
inputs114 = keras.Input((height, width, 3), name="input114")
inputs115 = keras.Input((height, width, 3), name="input115")
inputs116 = keras.Input((height, width, 3), name="input116")
inputs117 = keras.Input((height, width, 3), name="input117")
inputs118 = keras.Input((height, width, 3), name="input118")
inputs119 = keras.Input((height, width, 3), name="input119")
inputs120 = keras.Input((height, width, 3), name="input120")

inputs121 = keras.Input((height, width, 3), name="input121")
inputs122 = keras.Input((height, width, 3), name="input122")
inputs123 = keras.Input((height, width, 3), name="input123")
inputs124 = keras.Input((height, width, 3), name="input124")
inputs125 = keras.Input((height, width, 3), name="input125")
inputs126 = keras.Input((height, width, 3), name="input126")
inputs127 = keras.Input((height, width, 3), name="input127")
inputs128 = keras.Input((height, width, 3), name="input128")
inputs129 = keras.Input((height, width, 3), name="input129")
inputs130 = keras.Input((height, width, 3), name="input130")

inputs131 = keras.Input((height, width, 3), name="input131")
inputs132 = keras.Input((height, width, 3), name="input132")
inputs133 = keras.Input((height, width, 3), name="input133")
inputs134 = keras.Input((height, width, 3), name="input134")
inputs135 = keras.Input((height, width, 3), name="input135")
inputs136 = keras.Input((height, width, 3), name="input136")
inputs137 = keras.Input((height, width, 3), name="input137")
inputs138 = keras.Input((height, width, 3), name="input138")
inputs139 = keras.Input((height, width, 3), name="input139")
inputs140 = keras.Input((height, width, 3), name="input140")

inputs141 = keras.Input((height, width, 3), name="input141")
inputs142 = keras.Input((height, width, 3), name="input142")
inputs143 = keras.Input((height, width, 3), name="input143")
inputs144 = keras.Input((height, width, 3), name="input144")
inputs145 = keras.Input((height, width, 3), name="input145")
inputs146 = keras.Input((height, width, 3), name="input146")
inputs147 = keras.Input((height, width, 3), name="input147")
inputs148 = keras.Input((height, width, 3), name="input148")
inputs149 = keras.Input((height, width, 3), name="input149")
inputs150 = keras.Input((height, width, 3), name="input150")

inputs151 = keras.Input((height, width, 3), name="input151")
inputs152 = keras.Input((height, width, 3), name="input152")
inputs153 = keras.Input((height, width, 3), name="input153")
inputs154 = keras.Input((height, width, 3), name="input154")
inputs155 = keras.Input((height, width, 3), name="input155")
inputs156 = keras.Input((height, width, 3), name="input156")
inputs157 = keras.Input((height, width, 3), name="input157")
inputs158 = keras.Input((height, width, 3), name="input158")
inputs159 = keras.Input((height, width, 3), name="input159")
inputs160 = keras.Input((height, width, 3), name="input160")

inputs161 = keras.Input((height, width, 3), name="input161")
inputs162 = keras.Input((height, width, 3), name="input162")
inputs163 = keras.Input((height, width, 3), name="input163")
inputs164 = keras.Input((height, width, 3), name="input164")
inputs165 = keras.Input((height, width, 3), name="input165")
inputs166 = keras.Input((height, width, 3), name="input166")
inputs167 = keras.Input((height, width, 3), name="input167")
inputs168 = keras.Input((height, width, 3), name="input168")
inputs169 = keras.Input((height, width, 3), name="input169")
inputs170 = keras.Input((height, width, 3), name="input170")

inputs171 = keras.Input((height, width, 3), name="input171")
inputs172 = keras.Input((height, width, 3), name="input172")
inputs173 = keras.Input((height, width, 3), name="input173")
inputs174 = keras.Input((height, width, 3), name="input174")
inputs175 = keras.Input((height, width, 3), name="input175")
inputs176 = keras.Input((height, width, 3), name="input176")
inputs177 = keras.Input((height, width, 3), name="input177")
inputs178 = keras.Input((height, width, 3), name="input178")
inputs179 = keras.Input((height, width, 3), name="input179")
inputs180 = keras.Input((height, width, 3), name="input180")

inputs181 = keras.Input((height, width, 3), name="input181")
inputs182 = keras.Input((height, width, 3), name="input182")
inputs183 = keras.Input((height, width, 3), name="input183")
inputs184 = keras.Input((height, width, 3), name="input184")
inputs185 = keras.Input((height, width, 3), name="input185")
inputs186 = keras.Input((height, width, 3), name="input186")
inputs187 = keras.Input((height, width, 3), name="input187")
inputs188 = keras.Input((height, width, 3), name="input188")
inputs189 = keras.Input((height, width, 3), name="input189")
inputs190 = keras.Input((height, width, 3), name="input190")

inputs191 = keras.Input((height, width, 3), name="input191")
inputs192 = keras.Input((height, width, 3), name="input192")
inputs193 = keras.Input((height, width, 3), name="input193")
inputs194 = keras.Input((height, width, 3), name="input194")
inputs195 = keras.Input((height, width, 3), name="input195")
inputs196 = keras.Input((height, width, 3), name="input196")
inputs197 = keras.Input((height, width, 3), name="input197")
inputs198 = keras.Input((height, width, 3), name="input198")
inputs199 = keras.Input((height, width, 3), name="input199")
inputs200 = keras.Input((height, width, 3), name="input200")

inputs201 = keras.Input((height, width, 3), name="input201")
inputs202 = keras.Input((height, width, 3), name="input202")
inputs203 = keras.Input((height, width, 3), name="input203")
inputs204 = keras.Input((height, width, 3), name="input204")
inputs205 = keras.Input((height, width, 3), name="input205")
inputs206 = keras.Input((height, width, 3), name="input206")
inputs207 = keras.Input((height, width, 3), name="input207")
inputs208 = keras.Input((height, width, 3), name="input208")
inputs209 = keras.Input((height, width, 3), name="input209")
inputs210 = keras.Input((height, width, 3), name="input210")

inputs211 = keras.Input((height, width, 3), name="input211")
inputs212 = keras.Input((height, width, 3), name="input212")
inputs213 = keras.Input((height, width, 3), name="input213")
inputs214 = keras.Input((height, width, 3), name="input214")
inputs215 = keras.Input((height, width, 3), name="input215")
inputs216 = keras.Input((height, width, 3), name="input216")
inputs217 = keras.Input((height, width, 3), name="input217")
inputs218 = keras.Input((height, width, 3), name="input218")
inputs219 = keras.Input((height, width, 3), name="input219")
inputs220 = keras.Input((height, width, 3), name="input220")

inputs221 = keras.Input((height, width, 3), name="input221")
inputs222 = keras.Input((height, width, 3), name="input222")
inputs223 = keras.Input((height, width, 3), name="input223")
inputs224 = keras.Input((height, width, 3), name="input224")
inputs225 = keras.Input((height, width, 3), name="input225")
inputs226 = keras.Input((height, width, 3), name="input226")
inputs227 = keras.Input((height, width, 3), name="input227")
inputs228 = keras.Input((height, width, 3), name="input228")
inputs229 = keras.Input((height, width, 3), name="input229")
inputs230 = keras.Input((height, width, 3), name="input230")

inputs231 = keras.Input((height, width, 3), name="input231")
inputs232 = keras.Input((height, width, 3), name="input232")
inputs233 = keras.Input((height, width, 3), name="input233")
inputs234 = keras.Input((height, width, 3), name="input234")
inputs235 = keras.Input((height, width, 3), name="input235")
inputs236 = keras.Input((height, width, 3), name="input236")
inputs237 = keras.Input((height, width, 3), name="input237")
inputs238 = keras.Input((height, width, 3), name="input238")
inputs239 = keras.Input((height, width, 3), name="input239")
inputs240 = keras.Input((height, width, 3), name="input240")

inputs241 = keras.Input((height, width, 3), name="input241")
inputs242 = keras.Input((height, width, 3), name="input242")
inputs243 = keras.Input((height, width, 3), name="input243")
inputs244 = keras.Input((height, width, 3), name="input244")
inputs245 = keras.Input((height, width, 3), name="input245")
inputs246 = keras.Input((height, width, 3), name="input246")
inputs247 = keras.Input((height, width, 3), name="input247")
inputs248 = keras.Input((height, width, 3), name="input248")
inputs249 = keras.Input((height, width, 3), name="input249")
inputs250 = keras.Input((height, width, 3), name="input250")

inputs251 = keras.Input((height, width, 3), name="input251")
inputs252 = keras.Input((height, width, 3), name="input252")
inputs253 = keras.Input((height, width, 3), name="input253")
inputs254 = keras.Input((height, width, 3), name="input254")
inputs255 = keras.Input((height, width, 3), name="input255")
inputs256 = keras.Input((height, width, 3), name="input256")
inputs257 = keras.Input((height, width, 3), name="input257")
inputs258 = keras.Input((height, width, 3), name="input258")
inputs259 = keras.Input((height, width, 3), name="input259")
inputs260 = keras.Input((height, width, 3), name="input260")

inputs261 = keras.Input((height, width, 3), name="input261")
inputs262 = keras.Input((height, width, 3), name="input262")
inputs263 = keras.Input((height, width, 3), name="input263")
inputs264 = keras.Input((height, width, 3), name="input264")
inputs265 = keras.Input((height, width, 3), name="input265")
inputs266 = keras.Input((height, width, 3), name="input266")
inputs267 = keras.Input((height, width, 3), name="input267")
inputs268 = keras.Input((height, width, 3), name="input268")
inputs269 = keras.Input((height, width, 3), name="input269")
inputs270 = keras.Input((height, width, 3), name="input270")

inputs271 = keras.Input((height, width, 3), name="input271")
inputs272 = keras.Input((height, width, 3), name="input272")
inputs273 = keras.Input((height, width, 3), name="input273")
inputs274 = keras.Input((height, width, 3), name="input274")
inputs275 = keras.Input((height, width, 3), name="input275")
inputs276 = keras.Input((height, width, 3), name="input276")
inputs277 = keras.Input((height, width, 3), name="input277")
inputs278 = keras.Input((height, width, 3), name="input278")
inputs279 = keras.Input((height, width, 3), name="input279")
inputs280 = keras.Input((height, width, 3), name="input280")

inputs281 = keras.Input((height, width, 3), name="input281")
inputs282 = keras.Input((height, width, 3), name="input282")
inputs283 = keras.Input((height, width, 3), name="input283")
inputs284 = keras.Input((height, width, 3), name="input284")
inputs285 = keras.Input((height, width, 3), name="input285")
inputs286 = keras.Input((height, width, 3), name="input286")
inputs287 = keras.Input((height, width, 3), name="input287")
inputs288 = keras.Input((height, width, 3), name="input288")
inputs289 = keras.Input((height, width, 3), name="input289")
inputs290 = keras.Input((height, width, 3), name="input290")

inputs291 = keras.Input((height, width, 3), name="input291")
inputs292 = keras.Input((height, width, 3), name="input292")
inputs293 = keras.Input((height, width, 3), name="input293")
inputs294 = keras.Input((height, width, 3), name="input294")
inputs295 = keras.Input((height, width, 3), name="input295")
inputs296 = keras.Input((height, width, 3), name="input296")
inputs297 = keras.Input((height, width, 3), name="input297")
inputs298 = keras.Input((height, width, 3), name="input298")
inputs299 = keras.Input((height, width, 3), name="input299")
inputs300 = keras.Input((height, width, 3), name="input300")

x = layers.concatenate([  inputs1,   inputs2,   inputs3,   inputs4,   inputs5,   inputs6,   inputs7,   inputs8,   inputs9,  inputs10,
                         inputs11,  inputs12,  inputs13,  inputs14,  inputs15,  inputs16,  inputs17,  inputs18,  inputs19,  inputs20,
                         inputs21,  inputs22,  inputs23,  inputs24,  inputs25,  inputs26,  inputs27,  inputs28,  inputs29,  inputs30,
                         inputs31,  inputs32,  inputs33,  inputs34,  inputs35,  inputs36,  inputs37,  inputs38,  inputs39,  inputs40,
                         inputs41,  inputs42,  inputs43,  inputs44,  inputs45,  inputs46,  inputs47,  inputs48,  inputs49,  inputs50,
                         inputs51,  inputs52,  inputs53,  inputs54,  inputs55,  inputs56,  inputs57,  inputs58,  inputs59,  inputs60,
                         inputs61,  inputs62,  inputs63,  inputs64,  inputs65,  inputs66,  inputs67,  inputs68,  inputs69,  inputs70,
                         inputs71,  inputs72,  inputs73,  inputs74,  inputs75,  inputs76,  inputs77,  inputs78,  inputs79,  inputs80,
                         inputs81,  inputs82,  inputs83,  inputs84,  inputs85,  inputs86,  inputs87,  inputs88,  inputs89,  inputs90,
                         inputs91,  inputs92,  inputs93,  inputs94,  inputs95,  inputs96,  inputs97,  inputs98,  inputs99, inputs100,
                        inputs101, inputs102, inputs103, inputs104, inputs105, inputs106, inputs107, inputs108, inputs109, inputs110,
                        inputs111, inputs112, inputs113, inputs114, inputs115, inputs116, inputs117, inputs118, inputs119, inputs120,
                        inputs121, inputs122, inputs123, inputs124, inputs125, inputs126, inputs127, inputs128, inputs129, inputs130,
                        inputs131, inputs132, inputs133, inputs134, inputs135, inputs136, inputs137, inputs138, inputs139, inputs140,
                        inputs141, inputs142, inputs143, inputs144, inputs145, inputs146, inputs147, inputs148, inputs149, inputs150,
                        inputs151, inputs152, inputs153, inputs154, inputs155, inputs156, inputs157, inputs158, inputs159, inputs160,
                        inputs161, inputs162, inputs163, inputs164, inputs165, inputs166, inputs167, inputs168, inputs169, inputs170,
                        inputs171, inputs172, inputs173, inputs174, inputs175, inputs176, inputs177, inputs178, inputs179, inputs180,
                        inputs181, inputs182, inputs183, inputs184, inputs185, inputs186, inputs187, inputs188, inputs189, inputs190,
                        inputs191, inputs192, inputs193, inputs194, inputs195, inputs196, inputs197, inputs198, inputs199, inputs200,
                        inputs201, inputs202, inputs203, inputs204, inputs205, inputs206, inputs207, inputs208, inputs209, inputs210,
                        inputs211, inputs212, inputs213, inputs214, inputs215, inputs216, inputs217, inputs218, inputs219, inputs220,
                        inputs221, inputs222, inputs223, inputs224, inputs225, inputs226, inputs227, inputs228, inputs229, inputs230,
                        inputs231, inputs232, inputs233, inputs234, inputs235, inputs236, inputs237, inputs238, inputs239, inputs240,
                        inputs241, inputs242, inputs243, inputs244, inputs245, inputs246, inputs247, inputs248, inputs249, inputs250,
                        inputs251, inputs252, inputs253, inputs254, inputs255, inputs256, inputs257, inputs258, inputs259, inputs260,
                        inputs261, inputs262, inputs263, inputs264, inputs265, inputs266, inputs267, inputs268, inputs269, inputs270,
                        inputs271, inputs272, inputs273, inputs274, inputs275, inputs276, inputs277, inputs278, inputs279, inputs280,
                        inputs281, inputs282, inputs283, inputs284, inputs285, inputs286, inputs287, inputs288, inputs289, inputs290,
                        inputs291, inputs292, inputs293, inputs294, inputs295, inputs296, inputs297, inputs298, inputs299, inputs300 ])


x = Conv2D(filters=32,kernel_regularizer=tf.keras.regularizers.l2(0.001), kernel_size=(3, 3), strides=(1,1),activation='relu', padding='same', name="Conv1")(x)
x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1),activation='relu',padding='same', name="Conv2")(x)
x = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same',name="Pool1")(x)
x = Dropout(0.3, name="Drop1")(x)

x = Conv2D(filters=64, kernel_regularizer=tf.keras.regularizers.l2(0.001),kernel_size=(3, 3), strides=(1,1),activation='relu', padding='same', name="Conv3")(x)
x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1),activation='relu',padding='same', name="Conv4")(x)
x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name="Pool2")(x)
x = Dropout(0.5, name="Drop2")(x)

x = Flatten(name="Flat")(x)
x = Dense(units=512, kernel_regularizer=tf.keras.regularizers.l2(0.001),activation='relu',name="Dense1")(x)
x = Dropout(0.5, name="Drop3")(x)


outputs = Dense(units=5, activation='softmax',name="Dense2")(x)

#output = concatenate[x,y,z]

model = keras.Model(inputs=[inputs1, inputs2, inputs3, inputs4, inputs5, inputs6, inputs7, inputs8, inputs9, inputs10,
                            inputs11,  inputs12,  inputs13,  inputs14,  inputs15,  inputs16,  inputs17,  inputs18,  inputs19,  inputs20,
                            inputs21,  inputs22,  inputs23,  inputs24,  inputs25,  inputs26,  inputs27,  inputs28,  inputs29,  inputs30,
                            inputs31,  inputs32,  inputs33,  inputs34,  inputs35,  inputs36,  inputs37,  inputs38,  inputs39,  inputs40,
                            inputs41,  inputs42,  inputs43,  inputs44,  inputs45,  inputs46,  inputs47,  inputs48,  inputs49,  inputs50,
                            inputs51,  inputs52,  inputs53,  inputs54,  inputs55,  inputs56,  inputs57,  inputs58,  inputs59,  inputs60,
                            inputs61,  inputs62,  inputs63,  inputs64,  inputs65,  inputs66,  inputs67,  inputs68,  inputs69,  inputs70,
                            inputs71,  inputs72,  inputs73,  inputs74,  inputs75,  inputs76,  inputs77,  inputs78,  inputs79,  inputs80,
                            inputs81,  inputs82,  inputs83,  inputs84,  inputs85,  inputs86,  inputs87,  inputs88,  inputs89,  inputs90,
                            inputs91,  inputs92,  inputs93,  inputs94,  inputs95,  inputs96,  inputs97,  inputs98,  inputs99, inputs100,
                            inputs101, inputs102, inputs103, inputs104, inputs105, inputs106, inputs107, inputs108, inputs109, inputs110,
                            inputs111, inputs112, inputs113, inputs114, inputs115, inputs116, inputs117, inputs118, inputs119, inputs120,
                            inputs121, inputs122, inputs123, inputs124, inputs125, inputs126, inputs127, inputs128, inputs129, inputs130,
                            inputs131, inputs132, inputs133, inputs134, inputs135, inputs136, inputs137, inputs138, inputs139, inputs140,
                            inputs141, inputs142, inputs143, inputs144, inputs145, inputs146, inputs147, inputs148, inputs149, inputs150,
                            inputs151, inputs152, inputs153, inputs154, inputs155, inputs156, inputs157, inputs158, inputs159, inputs160,
                            inputs161, inputs162, inputs163, inputs164, inputs165, inputs166, inputs167, inputs168, inputs169, inputs170,
                            inputs171, inputs172, inputs173, inputs174, inputs175, inputs176, inputs177, inputs178, inputs179, inputs180,
                            inputs181, inputs182, inputs183, inputs184, inputs185, inputs186, inputs187, inputs188, inputs189, inputs190,
                            inputs191, inputs192, inputs193, inputs194, inputs195, inputs196, inputs197, inputs198, inputs199, inputs200,
                            inputs201, inputs202, inputs203, inputs204, inputs205, inputs206, inputs207, inputs208, inputs209, inputs210,
                            inputs211, inputs212, inputs213, inputs214, inputs215, inputs216, inputs217, inputs218, inputs219, inputs220,
                            inputs221, inputs222, inputs223, inputs224, inputs225, inputs226, inputs227, inputs228, inputs229, inputs230,
                            inputs231, inputs232, inputs233, inputs234, inputs235, inputs236, inputs237, inputs238, inputs239, inputs240,
                            inputs241, inputs242, inputs243, inputs244, inputs245, inputs246, inputs247, inputs248, inputs249, inputs250,
                            inputs251, inputs252, inputs253, inputs254, inputs255, inputs256, inputs257, inputs258, inputs259, inputs260,
                            inputs261, inputs262, inputs263, inputs264, inputs265, inputs266, inputs267, inputs268, inputs269, inputs270,
                            inputs271, inputs272, inputs273, inputs274, inputs275, inputs276, inputs277, inputs278, inputs279, inputs280,
                            inputs281, inputs282, inputs283, inputs284, inputs285, inputs286, inputs287, inputs288, inputs289, inputs290,
                            inputs291, inputs292, inputs293, inputs294, inputs295, inputs296, inputs297, inputs298, inputs299, inputs300 ],
                    outputs=outputs, name="mnist_model")
model.summary()


# In[10]:


from tensorflow.keras.optimizers import Adam

batch_size = 100
epochs = 50
lr = 0.001

# 最適化関数（Ir：学習率）
optimizer = Adam(learning_rate=lr)

# モデルコンパイル（学習設定）
model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",  #多クラス交差エントロピー
    metrics=["accuracy"] # 正答率(結果可視化用）
)

#モデル学習
history=model.fit(
    x=[  x1_train,   x2_train,   x3_train,   x4_train,   x5_train,   x6_train,   x7_train,   x8_train,   x9_train,  x10_train,
        x11_train,  x12_train,  x13_train,  x14_train,  x15_train,  x16_train,  x17_train,  x18_train,  x19_train,  x20_train,
        x21_train,  x22_train,  x23_train,  x24_train,  x25_train,  x26_train,  x27_train,  x28_train,  x29_train,  x30_train,
        x31_train,  x32_train,  x33_train,  x34_train,  x35_train,  x36_train,  x37_train,  x38_train,  x39_train,  x40_train,
        x41_train,  x42_train,  x43_train,  x44_train,  x45_train,  x46_train,  x47_train,  x48_train,  x49_train,  x50_train,
        x51_train,  x52_train,  x53_train,  x54_train,  x55_train,  x56_train,  x57_train,  x58_train,  x59_train,  x60_train,
        x61_train,  x62_train,  x63_train,  x64_train,  x65_train,  x66_train,  x67_train,  x68_train,  x69_train,  x70_train,
        x71_train,  x72_train,  x73_train,  x74_train,  x75_train,  x76_train,  x77_train,  x78_train,  x79_train,  x80_train,
        x81_train,  x82_train,  x83_train,  x84_train,  x85_train,  x86_train,  x87_train,  x88_train,  x89_train,  x90_train,
        x91_train,  x92_train,  x93_train,  x94_train,  x95_train,  x96_train,  x97_train,  x98_train,  x99_train, x100_train,
       x101_train, x102_train, x103_train, x104_train, x105_train, x106_train, x107_train, x108_train, x109_train, x110_train,
       x111_train, x112_train, x113_train, x114_train, x115_train, x116_train, x117_train, x118_train, x119_train, x120_train,
       x121_train, x122_train, x123_train, x124_train, x125_train, x126_train, x127_train, x128_train, x129_train, x130_train,
       x131_train, x132_train, x133_train, x134_train, x135_train, x136_train, x137_train, x138_train, x139_train, x140_train,
       x141_train, x142_train, x143_train, x144_train, x145_train, x146_train, x147_train, x148_train, x149_train, x150_train,
       x151_train, x152_train, x153_train, x154_train, x155_train, x156_train, x157_train, x158_train, x159_train, x160_train,
       x161_train, x162_train, x163_train, x164_train, x165_train, x166_train, x167_train, x168_train, x169_train, x170_train,
       x171_train, x172_train, x173_train, x174_train, x175_train, x176_train, x177_train, x178_train, x179_train, x180_train,
       x181_train, x182_train, x183_train, x184_train, x185_train, x186_train, x187_train, x188_train, x189_train, x190_train,
       x191_train, x192_train, x193_train, x194_train, x195_train, x196_train, x197_train, x198_train, x199_train, x200_train,
       x201_train, x202_train, x203_train, x204_train, x205_train, x206_train, x207_train, x208_train, x209_train, x210_train,
       x211_train, x212_train, x213_train, x214_train, x215_train, x216_train, x217_train, x218_train, x219_train, x220_train,
       x221_train, x222_train, x223_train, x224_train, x225_train, x226_train, x227_train, x228_train, x229_train, x230_train,
       x231_train, x232_train, x233_train, x234_train, x235_train, x236_train, x237_train, x238_train, x239_train, x240_train,
       x241_train, x242_train, x243_train, x244_train, x245_train, x246_train, x247_train, x248_train, x249_train, x250_train,
       x251_train, x252_train, x253_train, x254_train, x255_train, x256_train, x257_train, x258_train, x259_train, x260_train,
       x261_train, x262_train, x263_train, x264_train, x265_train, x266_train, x267_train, x268_train, x269_train, x270_train,
       x271_train, x272_train, x273_train, x274_train, x275_train, x276_train, x277_train, x278_train, x279_train, x280_train,
       x281_train, x282_train, x283_train, x284_train, x285_train, x286_train, x287_train, x288_train, x289_train, x290_train,
       x291_train, x292_train, x293_train, x294_train, x295_train, x296_train, x297_train, x298_train, x299_train, x300_train],
    y=y_train,  #訓練データと正解データ
    batch_size=batch_size,  #バッチサイズ
    epochs=epochs,   #エポック数(学習回数≒重みの更新回数)
    verbose=1,    #訓練の進行具合を表示
    validation_data=([  x1_test,   x2_test,   x3_test,   x4_test,   x5_test,   x6_test,   x7_test,   x8_test,   x9_test,  x10_test,
                       x11_test,  x12_test,  x13_test,  x14_test,  x15_test,  x16_test,  x17_test,  x18_test,  x19_test,  x20_test,
                       x21_test,  x22_test,  x23_test,  x24_test,  x25_test,  x26_test,  x27_test,  x28_test,  x29_test,  x30_test,
                       x31_test,  x32_test,  x33_test,  x34_test,  x35_test,  x36_test,  x37_test,  x38_test,  x39_test,  x40_test,
                       x41_test,  x42_test,  x43_test,  x44_test,  x45_test,  x46_test,  x47_test,  x48_test,  x49_test,  x50_test,
                       x51_test,  x52_test,  x53_test,  x54_test,  x55_test,  x56_test,  x57_test,  x58_test,  x59_test,  x60_test,
                       x61_test,  x62_test,  x63_test,  x64_test,  x65_test,  x66_test,  x67_test,  x68_test,  x69_test,  x70_test,
                       x71_test,  x72_test,  x73_test,  x74_test,  x75_test,  x76_test,  x77_test,  x78_test,  x79_test,  x80_test,
                       x81_test,  x82_test,  x83_test,  x84_test,  x85_test,  x86_test,  x87_test,  x88_test,  x89_test,  x90_test,
                       x91_test,  x92_test,  x93_test,  x94_test,  x95_test,  x96_test,  x97_test,  x98_test,  x99_test, x100_test,
                      x101_test, x102_test, x103_test, x104_test, x105_test, x106_test, x107_test, x108_test, x109_test, x110_test,
                      x111_test, x112_test, x113_test, x114_test, x115_test, x116_test, x117_test, x118_test, x119_test, x120_test,
                      x121_test, x122_test, x123_test, x124_test, x125_test, x126_test, x127_test, x128_test, x129_test, x130_test,
                      x131_test, x132_test, x133_test, x134_test, x135_test, x136_test, x137_test, x138_test, x139_test, x140_test,
                      x141_test, x142_test, x143_test, x144_test, x145_test, x146_test, x147_test, x148_test, x149_test, x150_test,
                      x151_test, x152_test, x153_test, x154_test, x155_test, x156_test, x157_test, x158_test, x159_test, x160_test,
                      x161_test, x162_test, x163_test, x164_test, x165_test, x166_test, x167_test, x168_test, x169_test, x170_test,
                      x171_test, x172_test, x173_test, x174_test, x175_test, x176_test, x177_test, x178_test, x179_test, x180_test,
                      x181_test, x182_test, x183_test, x184_test, x185_test, x186_test, x187_test, x188_test, x189_test, x190_test,
                      x191_test, x192_test, x193_test, x194_test, x195_test, x196_test, x197_test, x198_test, x199_test, x200_test,
                      x201_test, x202_test, x203_test, x204_test, x205_test, x206_test, x207_test, x208_test, x209_test, x210_test,
                      x211_test, x212_test, x213_test, x214_test, x215_test, x216_test, x217_test, x218_test, x219_test, x220_test,
                      x221_test, x222_test, x223_test, x224_test, x225_test, x226_test, x227_test, x228_test, x229_test, x230_test,
                      x231_test, x232_test, x233_test, x234_test, x235_test, x236_test, x237_test, x238_test, x239_test, x240_test,
                      x241_test, x242_test, x243_test, x244_test, x245_test, x246_test, x247_test, x248_test, x249_test, x250_test,
                      x251_test, x252_test, x253_test, x254_test, x255_test, x256_test, x257_test, x258_test, x259_test, x260_test,
                      x261_test, x262_test, x263_test, x264_test, x265_test, x266_test, x267_test, x268_test, x269_test, x270_test,
                      x271_test, x272_test, x273_test, x274_test, x275_test, x276_test, x277_test, x278_test, x279_test, x280_test,
                      x281_test, x282_test, x283_test, x284_test, x285_test, x286_test, x287_test, x288_test, x289_test, x290_test,
                      x291_test, x292_test, x293_test, x294_test, x295_test, x296_test, x297_test, x298_test, x299_test, x300_test ],y_test)  #テストデータ
)
        
    
    
model_dir = './model'
if os.path.exists(model_dir) == False:os.mkdir(model_dir)

model.save(model_dir+"/"+make_data+'.hdf5')
       


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.figure(figsize = (18,6))

# accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label = "accuracy", marker = "o")
plt.plot(history.history["val_accuracy"], label = "val_accuracy", marker = "o")
#plt.xticks(np.arange())
#plt.yticks(np.arange())
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(0.0, 1.1)
#plt.title("")
plt.legend(loc = "best")
plt.grid(color = 'gray', alpha=0.2)

# loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label = "loss", marker = "o")
plt.plot(history.history["val_loss"], label = "val_loss", marker = "o")
#plt.xticks(np.arange())
#plt.yticks(np.arange())
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(0,10)
#plt.title("")
plt.legend(loc="best")
plt.grid(color = 'gray', alpha = 0.2)

#plt.show()


# In[ ]:

test_num=10

# testデータ30件の予測ラベル
pred_classes=np.argmax(model.predict([x1_test[:test_num],x2_test[:test_num],x3_test[:test_num],x4_test[:test_num],x5_test[:test_num],x6_test[:test_num],x7_test[:test_num],x8_test[:test_num],x9_test[:test_num],x10_test[:test_num],
                                      x11_test[:test_num],x12_test[:test_num],x13_test[:test_num],x14_test[:test_num],x15_test[:test_num],x16_test[:test_num],x17_test[:test_num],x18_test[:test_num],x19_test[:test_num],x20_test[:test_num],
                                      x21_test[:test_num],x22_test[:test_num],x23_test[:test_num],x24_test[:test_num],x25_test[:test_num],x26_test[:test_num],x27_test[:test_num],x28_test[:test_num],x29_test[:test_num],x30_test[:test_num],
                                      x31_test[:test_num],x32_test[:test_num],x33_test[:test_num],x34_test[:test_num],x35_test[:test_num],x36_test[:test_num],x37_test[:test_num],x38_test[:test_num],x39_test[:test_num],x40_test[:test_num],
                                      x41_test[:test_num],x42_test[:test_num],x43_test[:test_num],x44_test[:test_num],x45_test[:test_num],x46_test[:test_num],x47_test[:test_num],x48_test[:test_num],x49_test[:test_num],x50_test[:test_num],
                                      x51_test[:test_num],x52_test[:test_num],x53_test[:test_num],x54_test[:test_num],x55_test[:test_num],x56_test[:test_num],x57_test[:test_num],x58_test[:test_num],x59_test[:test_num],x60_test[:test_num],
                                      x61_test[:test_num],x62_test[:test_num],x63_test[:test_num],x64_test[:test_num],x65_test[:test_num],x66_test[:test_num],x67_test[:test_num],x68_test[:test_num],x69_test[:test_num],x70_test[:test_num],
                                      x71_test[:test_num],x72_test[:test_num],x73_test[:test_num],x74_test[:test_num],x75_test[:test_num],x76_test[:test_num],x77_test[:test_num],x78_test[:test_num],x79_test[:test_num],x80_test[:test_num],
                                      x81_test[:test_num],x82_test[:test_num],x83_test[:test_num],x84_test[:test_num],x85_test[:test_num],x86_test[:test_num],x87_test[:test_num],x88_test[:test_num],x89_test[:test_num],x90_test[:test_num],
                                      x91_test[:test_num],x92_test[:test_num],x93_test[:test_num],x94_test[:test_num],x95_test[:test_num],x96_test[:test_num],x97_test[:test_num],x98_test[:test_num],x99_test[:test_num],x100_test[:test_num],
                                      x101_test[:test_num],x102_test[:test_num],x103_test[:test_num],x104_test[:test_num],x105_test[:test_num],x106_test[:test_num],x107_test[:test_num],x108_test[:test_num],x109_test[:test_num],x110_test[:test_num],
                                      x111_test[:test_num],x112_test[:test_num],x113_test[:test_num],x114_test[:test_num],x115_test[:test_num],x116_test[:test_num],x117_test[:test_num],x118_test[:test_num],x119_test[:test_num],x120_test[:test_num],
                                      x121_test[:test_num],x122_test[:test_num],x123_test[:test_num],x124_test[:test_num],x125_test[:test_num],x126_test[:test_num],x127_test[:test_num],x128_test[:test_num],x129_test[:test_num],x130_test[:test_num],
                                      x131_test[:test_num],x132_test[:test_num],x133_test[:test_num],x134_test[:test_num],x135_test[:test_num],x136_test[:test_num],x137_test[:test_num],x138_test[:test_num],x139_test[:test_num],x140_test[:test_num],
                                      x141_test[:test_num],x142_test[:test_num],x143_test[:test_num],x144_test[:test_num],x145_test[:test_num],x146_test[:test_num],x147_test[:test_num],x148_test[:test_num],x149_test[:test_num],x150_test[:test_num],
                                      x151_test[:test_num],x152_test[:test_num],x153_test[:test_num],x154_test[:test_num],x155_test[:test_num],x156_test[:test_num],x157_test[:test_num],x158_test[:test_num],x159_test[:test_num],x160_test[:test_num],
                                      x161_test[:test_num],x162_test[:test_num],x163_test[:test_num],x164_test[:test_num],x165_test[:test_num],x166_test[:test_num],x167_test[:test_num],x168_test[:test_num],x169_test[:test_num],x170_test[:test_num],
                                      x171_test[:test_num],x172_test[:test_num],x173_test[:test_num],x174_test[:test_num],x175_test[:test_num],x176_test[:test_num],x177_test[:test_num],x178_test[:test_num],x179_test[:test_num],x180_test[:test_num],
                                      x181_test[:test_num],x182_test[:test_num],x183_test[:test_num],x184_test[:test_num],x185_test[:test_num],x186_test[:test_num],x187_test[:test_num],x188_test[:test_num],x189_test[:test_num],x190_test[:test_num],
                                      x191_test[:test_num],x192_test[:test_num],x193_test[:test_num],x194_test[:test_num],x195_test[:test_num],x196_test[:test_num],x197_test[:test_num],x198_test[:test_num],x199_test[:test_num],x200_test[:test_num],
                                      x201_test[:test_num],x202_test[:test_num],x203_test[:test_num],x204_test[:test_num],x205_test[:test_num],x206_test[:test_num],x207_test[:test_num],x208_test[:test_num],x209_test[:test_num],x210_test[:test_num],
                                      x211_test[:test_num],x212_test[:test_num],x213_test[:test_num],x214_test[:test_num],x215_test[:test_num],x216_test[:test_num],x217_test[:test_num],x218_test[:test_num],x219_test[:test_num],x220_test[:test_num],
                                      x221_test[:test_num],x222_test[:test_num],x223_test[:test_num],x224_test[:test_num],x225_test[:test_num],x226_test[:test_num],x227_test[:test_num],x228_test[:test_num],x229_test[:test_num],x230_test[:test_num],
                                      x231_test[:test_num],x232_test[:test_num],x233_test[:test_num],x234_test[:test_num],x235_test[:test_num],x236_test[:test_num],x237_test[:test_num],x238_test[:test_num],x239_test[:test_num],x240_test[:test_num],
                                      x241_test[:test_num],x242_test[:test_num],x243_test[:test_num],x244_test[:test_num],x245_test[:test_num],x246_test[:test_num],x247_test[:test_num],x248_test[:test_num],x249_test[:test_num],x250_test[:test_num],
                                      x251_test[:test_num],x252_test[:test_num],x253_test[:test_num],x254_test[:test_num],x255_test[:test_num],x256_test[:test_num],x257_test[:test_num],x258_test[:test_num],x259_test[:test_num],x260_test[:test_num],
                                      x261_test[:test_num],x262_test[:test_num],x263_test[:test_num],x264_test[:test_num],x265_test[:test_num],x266_test[:test_num],x267_test[:test_num],x268_test[:test_num],x269_test[:test_num],x270_test[:test_num],
                                      x271_test[:test_num],x272_test[:test_num],x273_test[:test_num],x274_test[:test_num],x275_test[:test_num],x276_test[:test_num],x277_test[:test_num],x278_test[:test_num],x279_test[:test_num],x280_test[:test_num],
                                      x281_test[:test_num],x282_test[:test_num],x283_test[:test_num],x284_test[:test_num],x285_test[:test_num],x286_test[:test_num],x287_test[:test_num],x288_test[:test_num],x289_test[:test_num],x290_test[:test_num],
                                      x291_test[:test_num],x292_test[:test_num],x293_test[:test_num],x294_test[:test_num],x295_test[:test_num],x296_test[:test_num],x297_test[:test_num],x298_test[:test_num],x299_test[:test_num],x300_test[:test_num]]), axis=-1)

# testデータ30件の予測確率
pred_probs=model.predict([x1_test[:test_num],x2_test[:test_num],x3_test[:test_num],x4_test[:test_num],x5_test[:test_num],x6_test[:test_num],x7_test[:test_num],x8_test[:test_num],x9_test[:test_num],x10_test[:test_num],
                          x11_test[:test_num],x12_test[:test_num],x13_test[:test_num],x14_test[:test_num],x15_test[:test_num],x16_test[:test_num],x17_test[:test_num],x18_test[:test_num],x19_test[:test_num],x20_test[:test_num],
                                      x21_test[:test_num],x22_test[:test_num],x23_test[:test_num],x24_test[:test_num],x25_test[:test_num],x26_test[:test_num],x27_test[:test_num],x28_test[:test_num],x29_test[:test_num],x30_test[:test_num],
                                      x31_test[:test_num],x32_test[:test_num],x33_test[:test_num],x34_test[:test_num],x35_test[:test_num],x36_test[:test_num],x37_test[:test_num],x38_test[:test_num],x39_test[:test_num],x40_test[:test_num],
                                      x41_test[:test_num],x42_test[:test_num],x43_test[:test_num],x44_test[:test_num],x45_test[:test_num],x46_test[:test_num],x47_test[:test_num],x48_test[:test_num],x49_test[:test_num],x50_test[:test_num],
                                      x51_test[:test_num],x52_test[:test_num],x53_test[:test_num],x54_test[:test_num],x55_test[:test_num],x56_test[:test_num],x57_test[:test_num],x58_test[:test_num],x59_test[:test_num],x60_test[:test_num],
                                      x61_test[:test_num],x62_test[:test_num],x63_test[:test_num],x64_test[:test_num],x65_test[:test_num],x66_test[:test_num],x67_test[:test_num],x68_test[:test_num],x69_test[:test_num],x70_test[:test_num],
                                      x71_test[:test_num],x72_test[:test_num],x73_test[:test_num],x74_test[:test_num],x75_test[:test_num],x76_test[:test_num],x77_test[:test_num],x78_test[:test_num],x79_test[:test_num],x80_test[:test_num],
                                      x81_test[:test_num],x82_test[:test_num],x83_test[:test_num],x84_test[:test_num],x85_test[:test_num],x86_test[:test_num],x87_test[:test_num],x88_test[:test_num],x89_test[:test_num],x90_test[:test_num],
                                      x91_test[:test_num],x92_test[:test_num],x93_test[:test_num],x94_test[:test_num],x95_test[:test_num],x96_test[:test_num],x97_test[:test_num],x98_test[:test_num],x99_test[:test_num],x100_test[:test_num],
                                      x101_test[:test_num],x102_test[:test_num],x103_test[:test_num],x104_test[:test_num],x105_test[:test_num],x106_test[:test_num],x107_test[:test_num],x108_test[:test_num],x109_test[:test_num],x110_test[:test_num],
                                      x111_test[:test_num],x112_test[:test_num],x113_test[:test_num],x114_test[:test_num],x115_test[:test_num],x116_test[:test_num],x117_test[:test_num],x118_test[:test_num],x119_test[:test_num],x120_test[:test_num],
                                      x121_test[:test_num],x122_test[:test_num],x123_test[:test_num],x124_test[:test_num],x125_test[:test_num],x126_test[:test_num],x127_test[:test_num],x128_test[:test_num],x129_test[:test_num],x130_test[:test_num],
                                      x131_test[:test_num],x132_test[:test_num],x133_test[:test_num],x134_test[:test_num],x135_test[:test_num],x136_test[:test_num],x137_test[:test_num],x138_test[:test_num],x139_test[:test_num],x140_test[:test_num],
                                      x141_test[:test_num],x142_test[:test_num],x143_test[:test_num],x144_test[:test_num],x145_test[:test_num],x146_test[:test_num],x147_test[:test_num],x148_test[:test_num],x149_test[:test_num],x150_test[:test_num],
                                      x151_test[:test_num],x152_test[:test_num],x153_test[:test_num],x154_test[:test_num],x155_test[:test_num],x156_test[:test_num],x157_test[:test_num],x158_test[:test_num],x159_test[:test_num],x160_test[:test_num],
                                      x161_test[:test_num],x162_test[:test_num],x163_test[:test_num],x164_test[:test_num],x165_test[:test_num],x166_test[:test_num],x167_test[:test_num],x168_test[:test_num],x169_test[:test_num],x170_test[:test_num],
                                      x171_test[:test_num],x172_test[:test_num],x173_test[:test_num],x174_test[:test_num],x175_test[:test_num],x176_test[:test_num],x177_test[:test_num],x178_test[:test_num],x179_test[:test_num],x180_test[:test_num],
                                      x181_test[:test_num],x182_test[:test_num],x183_test[:test_num],x184_test[:test_num],x185_test[:test_num],x186_test[:test_num],x187_test[:test_num],x188_test[:test_num],x189_test[:test_num],x190_test[:test_num],
                                      x191_test[:test_num],x192_test[:test_num],x193_test[:test_num],x194_test[:test_num],x195_test[:test_num],x196_test[:test_num],x197_test[:test_num],x198_test[:test_num],x199_test[:test_num],x200_test[:test_num],
                                      x201_test[:test_num],x202_test[:test_num],x203_test[:test_num],x204_test[:test_num],x205_test[:test_num],x206_test[:test_num],x207_test[:test_num],x208_test[:test_num],x209_test[:test_num],x210_test[:test_num],
                                      x211_test[:test_num],x212_test[:test_num],x213_test[:test_num],x214_test[:test_num],x215_test[:test_num],x216_test[:test_num],x217_test[:test_num],x218_test[:test_num],x219_test[:test_num],x220_test[:test_num],
                                      x221_test[:test_num],x222_test[:test_num],x223_test[:test_num],x224_test[:test_num],x225_test[:test_num],x226_test[:test_num],x227_test[:test_num],x228_test[:test_num],x229_test[:test_num],x230_test[:test_num],
                                      x231_test[:test_num],x232_test[:test_num],x233_test[:test_num],x234_test[:test_num],x235_test[:test_num],x236_test[:test_num],x237_test[:test_num],x238_test[:test_num],x239_test[:test_num],x240_test[:test_num],
                                      x241_test[:test_num],x242_test[:test_num],x243_test[:test_num],x244_test[:test_num],x245_test[:test_num],x246_test[:test_num],x247_test[:test_num],x248_test[:test_num],x249_test[:test_num],x250_test[:test_num],
                                      x251_test[:test_num],x252_test[:test_num],x253_test[:test_num],x254_test[:test_num],x255_test[:test_num],x256_test[:test_num],x257_test[:test_num],x258_test[:test_num],x259_test[:test_num],x260_test[:test_num],
                                      x261_test[:test_num],x262_test[:test_num],x263_test[:test_num],x264_test[:test_num],x265_test[:test_num],x266_test[:test_num],x267_test[:test_num],x268_test[:test_num],x269_test[:test_num],x270_test[:test_num],
                                      x271_test[:test_num],x272_test[:test_num],x273_test[:test_num],x274_test[:test_num],x275_test[:test_num],x276_test[:test_num],x277_test[:test_num],x278_test[:test_num],x279_test[:test_num],x280_test[:test_num],
                                      x281_test[:test_num],x282_test[:test_num],x283_test[:test_num],x284_test[:test_num],x285_test[:test_num],x286_test[:test_num],x287_test[:test_num],x288_test[:test_num],x289_test[:test_num],x290_test[:test_num],
                                      x291_test[:test_num],x292_test[:test_num],x293_test[:test_num],x294_test[:test_num],x295_test[:test_num],x296_test[:test_num],x297_test[:test_num],x298_test[:test_num],x299_test[:test_num],x300_test[:test_num]]).max(axis=1)
pred_probs = ['{:.4f}'.format(i) for i in pred_probs]

# testデータ30件の画像と予測ラベル＆予測確率を出力
plt.figure(figsize=(10,10))
for i in range(test_num):
    plt.subplot(5, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # 正解なら黒、間違いなら赤で表示
    if pred_classes[i]==np.argmax(y_test[i]):
        plt.title(idx_to_label[pred_classes[i]]+':'+idx_to_label[np.argmax(y_test[i])]+'\n'+name_test[i]+'\n'+pred_probs[i])
    else:
        plt.title(idx_to_label[pred_classes[i]]+':'+idx_to_label[np.argmax(y_test[i])]+'\n'+name_test[i]+'\n'+pred_probs[i], color="red")

    plt.imshow(x1_test[i].reshape(height,width,3))


plt.show()







