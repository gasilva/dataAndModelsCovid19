import os
import io
import pandas as pd
from datetime import datetime,timedelta
import os.path
import time
import sys
from yabox import DE
from tqdm import tqdm
import dataFit_SEAIRD_AdjustIC as adjustIC

def create_fun(country,e0,a0,r0,d0,date,version):
    def fun(point):
        s0, deltaDate, i0, wcases, wrec = point
        cleanRecovered=True
        startNCases=100
        predict_range=200

        Date = datetime.strptime(date, "%m/%d/%y")
        end_date = Date + timedelta(days=+int(deltaDate))
        startdate=end_date.strftime('X%m/X%d/%y').replace('X0','X').replace('X','')
        
        learner=adjustIC.Learner(country, adjustIC.lossOdeint, startdate, predict_range,\
            s0, e0, a0, i0, r0, d0, version, startNCases, wcases, wrec, 
            cleanRecovered)
        gtot=learner.train()
        return gtot
    return fun

def opt(country,e0,a0,r0,d0,date,version):
    bounds = [(1e6,9e6),(-5,15),(0,600),\
              (0.1,0.7),(0.1,0.4)]
    maxiterations=300
    f=create_fun(country,e0,a0,r0,d0,date,version)
    de = DE(f, bounds, maxiters=maxiterations)
    i=0
    with tqdm(total=maxiterations*500) as pbar:
        for step in de.geniterator():
            try:
                idx = step.best_idx
                norm_vector = step.population[idx]
                best_params = de.denormalize([norm_vector])
            except:
                print("error in function evaluation")
            pbar.update(i)
            i+=1
    p=best_params[0]
    return p

date="2/25/20"
s0=12000000
e0=0
a0=0
i0=265
r0=0
d0=0
#weigth for fitting data
wcases=0.6
wrec=0.1
#weightDeaths = 1 - weigthCases - weigthRecov
#countries=["Italy","China","France"]
countries=["Brazil"]

optimal=[]
version=240
y=[]
for country in countries:
    
    strFile='./data/optimum'+str(version)+'.pkl'
    dfresult=pd.DataFrame([[1e6,1e6,1e6,1e6]], columns=['g1','g2','g3','Total'])
    dfresult.to_pickle(strFile)    
    
    if country=="China":
        date="1/25/20"
        s0=600e3
        e0=1e-4
        i0=800
        r0=0 #-250e3
        d0=0
        #start fitting when the number of cases >= start
        # startNCases=0
        #how many days is the prediction
        # predict_range=150
        #weigth for fitting data
        wcases=0.15
        wrec=0.1
        #weightDeaths = 1 - weigthCases - weigthRecov
    
    if country=="Italy":
        date="2/24/20"
        s0=2.1e6 #3e6*4*2*2*0.7*1.2*1.1
        e0=1e-4
        i0=200
        r0=0
        d0=50
        #start fitting when the number of cases >= start
        # startNCases=100
        #how many days is the prediction
        # predict_range=150
        #weigth for fitting data
        wcases=0.1
        wrec=0.1
        #weightDeaths = 1 - weigthCases - weigthRecov

    if country=="France":
        date="3/3/20"
        s0=1e6 #1.5e6*1.5*120/80*1.05
        e0=1e-4
        i0=0
        r0=0
        d0=0
        #start fitting when the number of cases >= start
        # startNCases=100
        #how many days is the prediction
        # predict_range=150
        #weigth for fitting data
        wcases=0.1
        wrec=0.1
        #weightDeaths = 1 - weigthCases - weigthRecov

    if country=="Brazil":
        date="3/3/20"
        s0=3.0e6*2.5 #500e3*1.7
        e0=1e-4
        i0=100
        r0=0 #5e3 #5000 #14000
        d0=0
        #start fitting when the number of cases >= start
        #startNCases=150
        #how many days is the prediction
        #predict_range=200
        #weigth for fitting data
        weigthCases=0.4
        weigthRecov=0.10
        #weightDeaths = 1 - weigthCases - weigthRecov
        cleanRecovered=True

    optimal.append(opt(country,e0,a0,r0,d0,date,version))  
    version+=1

for i in range(0,len(countries)):    
    with io.open('./results/resultOpt'+countries[i]+str(version)+'.txt', 'w', encoding='utf8') as f:
        f.write("country = {}\n".format(countries[i]))
        f.write("S0 = {}\n".format(optimal[i][0]))
        f.write("Delta Date Days = {}\n".format(optimal[i][1]))   
        f.write("I0 = {}\n".format(optimal[i][2]))   
        f.write("wCases = {}\n".format(optimal[i][3]))   
        f.write("wRec = {}\n".format(optimal[i][4]))    
    
    stdoutOrigin=sys.stdout 
    sys.stdout = open('./results/log'+countries[i]+str(i)+'.txt', "w")    
    print(optimal[i])
    sys.stdout.close()
    sys.stdout=stdoutOrigin