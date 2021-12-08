import json
import anfis
from anfis import membershipfunction
from anfis import mfDerivs
from anfis import anfis 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
counter=0
l="yes"
d="yes"
t="yes"
while(1):
        dataset = pd.read_csv('fuzzy_new.csv',sep=",",  header=None ,  usecols=[0,1,2])
	XTEST = dataset.iloc[:, 0:3].values
                              #print(XTEST)
        filename = 'anfis_rasp.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
	YTE=anfis.predict(loaded_model,XTEST)
        YTEST=round(YTE)
	if(YTEST==1):
		print("clean")
	 	OUT="CLEAN"
	elif(YTEST==2):
		print("mild")
		OUT="MILD"
	elif(YTEST==3):
		print("smelly")
   	        OUT="SMELLY"
	elif(YTEST==4):
   		print("dirty")
   		OUT="DIRTY"
	elif(YTEST==5):
   		print("pungent")
   		OUT="PUNGENT"
        counter=counter+1 
        print(counter);

