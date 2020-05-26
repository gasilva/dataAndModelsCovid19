import os
import io
import pandas as pd
from datetime import datetime,timedelta
import os.path
import time
import sys
from yabox import DE
from tqdm import tqdm
from modules.dataFit_SEAIRD_v2AdjustIC import Learner,lossOdeint
import subprocess as sub
from itertools import (takewhile,repeat)
import ray

def line_count(filename):
    return int(sub.check_output(['wc', '-l', filename]).split()[0])

def rawincount(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum( buf.count(b'\n') for buf in bufgen )

def append_new_line(file_name, text_to_append):
    #Append given text as a new line at the end of file
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(9999999)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

def create_fun(country,e0,a0,date,version):
    def fun(point):
        s0, deltaDate, i0, wcases, wrec, r0, d0 = point
        cleanRecovered=False
        startNCases=100
        predict_range=200
        s0=int(round(s0))
        i0=int(round(i0))
        deltaDate=int(deltaDate)
        r0=int(round(r0))
        d0=int(round(d0))

        if version==0:
            versionStr=""
        else:
            versionStr=str(version)

        Date = datetime.strptime(date, "%m/%d/%y")
        end_date = Date + timedelta(days=+int(deltaDate))
        startdate=end_date.strftime('X%m/X%d/%y').replace('X0','X').replace('X','')

        strFile='./results/history_'+country+versionStr+'.csv'
        if os.path.isfile(strFile):
            totLines=rawincount(strFile)
        else:
            totLines=-1
        
        learner=Learner(country, lossOdeint, startdate, predict_range,\
            s0, e0, a0, i0, r0, d0, version, startNCases, wcases, wrec, 
            cleanRecovered)
        optimal, gtot=learner.train()

        beta, beta2, sigma, sigma2, sigma3, gamma, b, mu = optimal
        print(f"s0={s0}, date={startdate}, i0={i0}, wrec={wrec}, wcases={wcases}")
        print(f"country={country}, beta={beta:.8f}, beta2={beta2:.8f}, 1/sigma={1/sigma:.8f},"+\
            f" 1/sigma2={1/sigma2:.8f},1/sigma3={1/sigma3:.8f}, gamma={gamma:.8f}, b={b:.8f},"+\
            f" mu={mu:.8f}, r_0:{(beta/gamma):.8f}")
        print("f(x)={}".format(gtot))
        print(country+" is done!")

        strSave='{}, {}, {}, '.format(totLines+1,country, gtot)
        strSave=strSave+', '.join(map(str,point))
        strSave=strSave+', '+', '.join(map(str,optimal))
        append_new_line('./results/history_'+country+versionStr+'.csv', strSave)  

        return gtot
    return fun

#register function for parallel processing
@ray.remote
def opt(country,e0,a0,date,version):
    bounds = [(0.5e6,15e6),(0,5),(0,1500),\
              (0.1,0.5),(0.08,0.12),(0,1500),(0,1500)]
    maxiterations=300
    f=create_fun(country,e0,a0,date,version)
    de = DE(f, bounds, maxiters=maxiterations)
    i=0
    with tqdm(total=maxiterations*250) as pbar:
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

ray.shutdown()
ray.init(num_cpus=3)

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
countries=["Italy","France","Brazil"]
# countries=["Italy","China","France","US","Germany","Sweden","United Kingdowm", "Spain", "Belgium"]
#countries=["Brazil"]

optimal=[]
version=60
versionx=version
y=[]

for country in countries:  

    if version==0:
        versionStr=""
    else:
        versionStr=str(version)
    
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
        date="3/02/20"
        s0=3.0e6*6.5 #3.0e6*3.5 not clean #3.0e6*2.5 clean
        e0=1e-4
        i0=500 #not clean
        r0=0 #14000 #14000 not clean #0 clean
        d0=0
        #start fitting when the number of cases >= start
        startNCases=150
        #how many days is the prediction
        predict_range=200
        #weigth for fitting data
        wcases=0.47 #not clean
        wrec=0.1 #not clean
        #weightDeaths = 1 - weigthCases - weigthRecov
        cleanRecovered=False

    strFile='./results/history_'+country+versionStr+'.csv'
    if os.path.isfile(strFile):
        os.remove(strFile)
    optimal.append(opt.remote(country,e0,a0,date,version))  
    version+=1
optimal = ray.get(optimal)

for i in range(0,len(countries)):    

    if versionx==0:
        versionStr=""
    else:
        versionStr=str(versionx)
    
    with io.open('./results/resultOpt'+countries[i]+versionStr+'.txt', 'w', encoding='utf8') as f:
        f.write("country = {}\n".format(countries[i]))
        f.write("S0 = {}\n".format(optimal[i][0]))
        f.write("Delta Date Days = {}\n".format(optimal[i][1]))   
        f.write("I0 = {}\n".format(optimal[i][2]))   
        f.write("wCases = {}\n".format(optimal[i][3]))   
        f.write("wRec = {}\n".format(optimal[i][4]))    
        f.write("R0 = {}\n".format(optimal[i][5])) 
        f.write("D0 = {}\n".format(optimal[i][6])) 
    
    stdoutOrigin=sys.stdout 
    sys.stdout = open('./results/log'+countries[i]+versionStr+'.txt', "w")    
    print(optimal[i])
    sys.stdout.close()
    sys.stdout=stdoutOrigin

    versionx+=1