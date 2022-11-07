# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Wed Aug 31 16:30:12 2022)---
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/PatientData_PDF_Online.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
pip install gspread
pip3 install gspread
conda create -n spyder-env -y
pip install 
%pip install 
''%pip install'' gspread
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/PatientData_PDF_Online.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/PatientData_PDF_Online.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')

## ---(Thu Sep  1 15:27:02 2022)---
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/PatientData_PDF_Online.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')

## ---(Thu Sep  1 15:31:54 2022)---
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/PatientData_PDF_Online.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')

## ---(Thu Sep  1 15:32:51 2022)---
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/PatientData_PDF_Online.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/PatientData_PDF_Online.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
pip install gspread
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/PatientData_PDF_Online.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
pip install pandas
pip install fpdf
pip install tensorflow
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/PatientData_PDF_Online.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
pip install matplotlib
pip install warnings
pip install random
pip install tqdm
pip install os
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
pip install seaborn
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
pip install sklearn
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
import PIL
l = []
for image in df['images']:
    try:
        img = PIL.Image.open(image)
    except:
        l.append(image)
l
"""
Created on Tue Aug 30 14:41:49 2022

@author: Micaela
"""

#https://www.hackersrealm.net/post/dogs-vs-cats-image-classification-using-python
#Dog vs cat image classification. 

# pip install tf-nightly 

#(1 = dog , 0 = cat)
#import tensorflow 
#from tensorflow import keras

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import warnings
import random 
import tqdm
import os

#https://stackoverflow.com/questions/66092421/how-to-rebuild-tensorflow-with-the-compiler-flags
#error fix, Rebuild tensorflow with appropriate compiler tags 
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.preprocessing.image import load_img
warnings.filterwarnings('ignore')

PetImages =  "C:/Users/Micaela/Desktop/BloodAnalysis/Neural_Network/PetImages"
#BloodImages = "C:/Users/Micaela/Desktop/BloodAnalysis/Neural_Network/BloodImages"

input_path = []
label = []

#A test to make sure the image path works 

#test = os.listdir(PetImages)

#for class_name in os.listdir("Neural_Network/BloodImages"):
    #for path in os.listdir("Neural_Network/BloodImages/"+class_name):
        #if class_name == 'Acanthocytes':
            #label.append(0)
        #elif class_name == 'Amibocytes':
            #label.append(1)
        #elif class_name == 'Bacterial_Rod_Formations':
            #label.append(2)
        #elif class_name == 'Echinocytes':
            #label.append(3)
        #elif class_name == 'Fungal_Forms':
            #label.append(4)
        #elif class_name == 'Hemolysis':
            #label.append(5)
        #elif class_name == 'L_Forms':
            #label.append(6)
        #elif class_name == 'Ovalocytes':
            #label.append(7)
        #elif class_name == 'Plaque':
            #label.append(8)
        #elif class_name == 'Poikilocytes':
            #label.append(9)
        #elif class_name == 'PPRBC':
            #label.append(10)
        #elif class_name == 'Protein_Linkage':
            #label.append(11)
        #elif class_name == 'Rouleau':
            #label.append(12)
        #elif class_name == 'Spicules':
            #label.append(13)
        #elif class_name == 'Target_Cells':
            #label.append(14)
        #elif class_name == 'Uric_Acid_Crystals':
            #label.append(15)
        #elif class_name == 'VAD':
            #label.append(16)
        #elif class_name == 'VCD':
            #label.append(17)

for class_name in os.listdir("Neural_Network/PetImages"):
    for path in os.listdir("Neural_Network/PetImages/"+class_name):
        if class_name == 'Cat':
            label.append(0)
        else:
            label.append(1)
        input_path.append(os.path.join("Neural_Network/PetImages", class_name, path))    

#Label 1 for dogs, 0 for cats
print(input_path[0], label[0])

df = pd.DataFrame()
df['images'] = input_path 
df['label'] = label 
df = df.sample(frac=1).reset_index(drop=True)
df.head()

#DF displays a 1 for dogs and a 0 for cats 

for i in df['images']:
    if '.jpg' not in i:
        print(i)

#Prints 
#Neural_Network/PetImages\Cat\0.jpg 0 
#Neural_Network/PetImages\Cat\Thumbs.db
#Neural_Network/PetImages\Dog\Thumbs.db

import PIL
l = []
for image in df['images']:
    try:
        img = PIL.Image.open(image)
    except:
        l.append(image)
l

df = df[df['images']!='#Neural_Network/PetImages\Cat\0.jpg 0']
df = df[df['images']!='PetImages/Cat/Thumbs.db']
df = df[df['images']!='PetImages/Cat/666.jpg']
df = df[df['images']!='PetImages/Dog/11702.jpg']
len(df)
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')

## ---(Thu Sep  1 17:17:21 2022)---
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Cat_Dog_Classification.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Scikitlearn.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/Scikitlearn.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')

## ---(Thu Sep 22 11:03:27 2022)---
runfile('C:/Users/Micaela/Desktop/BloodAnalysis/PatientData_PDF_Online.py', wdir='C:/Users/Micaela/Desktop/BloodAnalysis')