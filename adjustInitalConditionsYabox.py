import os
import io
import pandas as pd
from datetime import datetime,timedelta
import os.path
import time
import sys
from yabox import DE
from modules.dataFit_SEAIRD_v2bOnlyAdjustIC import Learner,lossOdeint,\
    download_data,load_json,sumCases_province

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

def create_fun(country,e0,a0,date,version,startNCases):
    def fun(point):
        s0, deltaDate, i0, wcases, wrec, r0, d0 = point
        cleanRecovered=False
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
        optimal, gtot = learner.train()

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
def opt(country,e0,a0,date,version,startNCases):
    if country=="China":
        dayInf=0
    else:
        dayInf=-10
    bounds=[(0.5e6,20e6),(dayInf,10),(0,1500),\
              (0.01,0.7),(0.01,0.12),(-50e3,50e3),(0,1500)]
    maxiterations=1000
    f=create_fun(country,e0,a0,date,version,startNCases)
    de = DE(f, bounds, maxiters=maxiterations)
    for step in de.geniterator():
        try:
            idx = step.best_idx
            norm_vector = step.population[idx]
            best_params = de.denormalize([norm_vector])
        except:
            print("error in function evaluation")
    p=best_params[0]
    return p

#download new data
data_d = load_json("./data_url.json")
download_data(data_d)
sumCases_province('data/time_series_19-covid-Confirmed.csv', 'data/time_series_19-covid-Confirmed-country.csv')
sumCases_province('data/time_series_19-covid-Recovered.csv', 'data/time_series_19-covid-Recovered-country.csv')
sumCases_province('data/time_series_19-covid-Deaths.csv', 'data/time_series_19-covid-Deaths-country.csv')

#common parameters initialization
e0=0
a0=0
#selected countries to be analyzed
countries=["Italy","France","Brazil", "China"]
# countries=["Italy","China","France","US","Germany","Sweden","United Kingdowm", "Spain", "Belgium"]
#countries=["Brazil"]

#one processor per country
ray.shutdown()
ray.init(num_cpus=len(countries))

#initialize vars
optimal=[]
version=200
versionx=version
y=[]

#main loop
for country in countries:  

    #version of optimization
    if version==0:
        versionStr=""
    else:
        versionStr=str(version)
    
    #initialize parameters for each country in the list
    if country=="China":
        date="1/25/20"
        #start fitting when the number of cases >= start
        startNCases=0
    if country=="Italy":
        date="2/24/20"
        #start fitting when the number of cases >= start
        startNCases=100
    if country=="France":
        date="3/3/20"
        s0=1e6 #1.5e6*1.5*120/80*1.05
        #start fitting when the number of cases >= start
        startNCases=100
    if country=="Brazil":
        date="3/02/20"
        #start fitting when the number of cases >= start
        startNCases=150

    #remove previous history file
    strFile='./results/history_'+country+versionStr+'.csv'
    if os.path.isfile(strFile):
        os.remove(strFile)

    #optimize    
    optimal.append(opt.remote(country,e0,a0,date,version,startNCases))  
    version+=1

#finalize ray workers
optimal = ray.get(optimal)

#save final results of IC
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