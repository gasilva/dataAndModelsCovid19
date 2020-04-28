import os
import io
import pandas as pd
from scipy.optimize import brute
from datetime import datetime,timedelta
import ray
import os.path
import time
import sys


def fun(point, country,e0,a0,r0,d0,date,version):
    s0, deltaDate, i0, wcases, wrec = point
    download=False
    Date = datetime.strptime(date, "%m/%d/%y")
    end_date = Date + timedelta(days=+int(deltaDate))
    dateStr=end_date.strftime('X%m/X%d/%y').replace('X0','X').replace('X','')
    command  = "python dataFit_SEAIRD_AdjustIC.py"
    command  +=' --countries {}'.format(country)
    command  +=' --start-date {}'.format(dateStr)
    command  +=' --download-data {}'.format(download)
    command  +=' --S_0 {}'.format(int(s0+0.5))
    command  +=' --E_0 {}'.format(int(e0+0.5))
    command  +=' --A_0 {}'.format(int(a0+0.5))
    command  +=' --I_0 {}'.format(int(i0+0.5))
    command  +=' --R_0 {}'.format(int(r0+0.5))
    command  +=' --D_0 {}'.format(int(d0+0.5))
    command  +=' --VER {}'.format(int(version))
    command  +=' --WCASES {}'.format(wcases)
    command  +=' --WREC {}'.format(wrec)
    print(command)  
    os.system(command)
    
    file_path='./data/optimum'+str(version)+'.pkl'
    
    time_to_wait=60
    time_counter=0
    while not os.path.exists(file_path):
        time.sleep(1)
        time_counter += 1
        if time_counter > time_to_wait:break
    
    df= pd.read_pickle(file_path)
    print("infected {}, recovered {}, death {}, total {}".\
    format(df.g1.values[0],df.g2.values[0],df.g3.values[0],df.Total.values[0]))
    return df.Total

@ray.remote
def opt(country,e0,a0,r0,d0,date,version):
    rranges = [slice(1e5,2.5e6,5e5),slice(0,1,1),slice(200,600,200),\
               slice(0.1,0.6,0.1),slice(0.1,0.2,0.05)]
    optimal = brute(fun,        
        ranges=rranges,
        args=(country,e0,a0,r0,d0,date,version), full_output=True, \
            disp=True, finish=None)
    return optimal

ray.shutdown()
ray.init(num_cpus=5)

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
countries=["Italy","China","France"]

optimal=[]
version=240
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

    optimal.append(opt.remote(country,e0,a0,r0,d0,date,version))   
    version+=1

optimal = ray.get(optimal)

for i in range(0,len(countries)):    
    with io.open('./results/resultOpt'+countries[i]+str(version)+'.txt', 'w', encoding='utf8') as f:
        f.write("country = {}\n".format(countries[i]))
        f.write("S0 = {}\n".format(optimal[i][0][0]))
        f.write("Delta Date Days = {}\n".format(optimal[i][0][1]))   
        f.write("I0 = {}\n".format(optimal[i][0][2]))   
        f.write("wCases = {}\n".format(optimal[i][0][3]))   
        f.write("wRec = {}\n".format(optimal[i][0][4]))   
        f.write("Function Minimum = {}\n".format(optimal[i][1]))  
    
    stdoutOrigin=sys.stdout 
    sys.stdout = open('./results/log'+countries[i]+str(i)+'.txt', "w")    
    print(optimal[i])
    sys.stdout.close()
    sys.stdout=stdoutOrigin