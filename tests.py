import anfis
import membership
#import membershipfunction, mfDerivs
import numpy 
import matplotlib.pyplot as plt
import pandas as pd
f2=pd.read_csv("Labelled_Datasetnew.csv", sep=",",  header=None ,  usecols=[3,4,5,6,7,8,9,10,11,12])  
f2.columns=["x","y","a","b","c","d","e","f","g","i"]     
           
for index,line1 in f2.iterrows():
    if((line1['a']+line1['d'] == 2) and (line1['b'] > 2)):
        line1['cl']= 2
    elif((line1['a']+line1['d'] == 1) and line1['b'] > 3):  
        line1['cl']= 3
    else:
        line1['cl']=line1['b']
    
    if((line1['e']+line1['f']+line1['g'] == 3) and (line1['c'] > 2)):
        line1['ex']= 2
    elif((line1['e']+line1['f']+line1['g'] == 2) and line1['c'] > 3):  
        line1['ex']= 3
    elif((line1['e']+line1['f']+line1['g'] == 1) and line1['c'] > 4):  
        line1['ex']= 4
    else:
        line1['ex']=line1['c']
    if(line1['cl']<line1['ex']):
        line1['cl']=line1['ex']
    print(line1['cl'])
    with open('fuzzy.csv', 'a') as f:
        f.write(str(line1["x"])+","+str(line1["y"])+","+str(line1["cl"])+","+str(int(line1["i"])))
        f.write("\n")
# Importing the dataset
dataset = pd.read_csv('fuzzy.csv',sep=",",  header=None ,  usecols=[0,1,2,3])
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, 3].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=6)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.5, random_state=6)

mf = [[['gbellmf',{'a':6,'b':9,'c':4.5}],['gbellmf',{'a':5.5,'b':7,'c':12.5}],['gbellmf',{'a':4.5,'b':7,'c':19.5}],['gbellmf',{'a':6.,'b':8.,'c':27.}],['gbellmf',{'a':8,'b':14,'c':38}]],
[['gaussmf',{'mean':150.0,'sigma':10.}],['gaussmf',{'mean':250.0,'sigma':10.}],['gaussmf',{'mean':350.0,'sigma':10.}]],
[['gaussmf',{'mean':0.5,'sigma':1.}],['gaussmf',{'mean':1.5,'sigma':1.}],['gaussmf',{'mean':2.5,'sigma':1.}],['gaussmf',{'mean':3.5,'sigma':1.}],['gaussmf',{'mean':4.5,'sigma':1.}]]

]

mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X_train, y_train, mfc)
anf.trainHybridJangOffLine(epochs=2)
Ynew=anfis.predict(anf,X_test)
y1=[]
for i in Ynew:
    if (i<6 and i>=1):
        y1.append(int(i))
    else: 
        y1.append(3)
    
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y1)
score=accuracy_score(y_test, y1)
print(cm)
print(score)

