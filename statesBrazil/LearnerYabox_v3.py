# Import the necessary packages and modules
import sys
import csv
import math
import array
import operator
import argparse
import sys
import json
import ssl
import os
import urllib.request
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from csv import reader
from csv import writer
import dateutil.parser
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.optimize import basinhopping
from yabox import DE
from tqdm import tqdm
import sigmoid as sg

from environs import Env
env = Env()
env.str("CUDA_DEVICE_ORDER",'PCI_BUS_ID')
env.str("CUDA_VISIBLE_DEVICES",'0,1,2,3,4,5')
env.int("NUMBA_ENABLE_CUDASIM",1)
env.bool("OMPI_MCA_opal_cuda_support",True)

#parallel computation
import ray
ray.shutdown()
ray.init(num_gpus=96,num_cpus=6)

import unicodedata

#register function for parallel processing
@ray.remote(memory=8*1024*1024*1024,num_gpus=10,num_cpus=0)
class Learner(object):
    def __init__(self, state, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, \
    startNCases, weigthCases, weigthRecov, weigthDeath, end_date, ratio, cleanRecovered, version, \
                 underNotif=True, Deaths=False, propWeigth=True, savedata=True):
        self.state = state
        self.start_date = start_date
        self.predict_range = predict_range
        self.s_0 = s_0
        self.e_0 = e_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.d_0 = d_0
        self.a_0 = a_0
        self.startNCases = startNCases
        self.weigthCases = weigthCases
        self.weigthRecov = weigthRecov
        self.weigthDeath = weigthDeath
        self.cleanRecovered=cleanRecovered
        self.version=version
        self.savedata = savedata
        self.under = underNotif
        self.end_date = end_date
        self.Deaths = Deaths
        self.propWeigth = propWeigth
        self.ratio=0 #ratio
        self.sigmoidTime=0.85

    def strip_accents(self,text):
        try:
            text = unicode(text, 'utf-8')
        except NameError: # unicode is a default on python 3 
            pass

        text = unicodedata.normalize('NFD', text)\
               .encode('ascii', 'ignore')\
               .decode("utf-8")
        return str(text)

    def load_confirmed(self):
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv('./data/confirmados.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
        y=[]
        x=[]
        start=datetime.strptime(self.start_date, "%Y-%m-%d")+timedelta(days=10)
        start2=start.strftime("%Y-%m-%d")
        for i in range(0,len(df.date)):
            y.append(df[self.state].values[i])
            x.append(df.date.values[i])
        df2=pd.DataFrame(data=y,index=x,columns=[""])
        df2 =df2.apply (pd.to_numeric, errors='coerce')
        df2[start2:] = df2[start2:].replace({0:np.nan})
        df2 = df2.dropna()
        df2.index = pd.DatetimeIndex(df2.index)
        #interpolate missing data
        df2 = df2.reindex(pd.date_range(df2.index.min(), df2.index.max()), fill_value=np.nan)
        df2 = df2.interpolate(method='akima', axis=0).ffill().bfill()
        #string type for dates and integer for data
        df2 = df2.astype(int)
        df2.index = df2.index.astype(str)
        df2=df2[self.start_date:]
        return df2

    def load_dead(self):
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv('./data/mortes.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
        y=[]
        x=[]
        start=datetime.strptime(self.start_date, "%Y-%m-%d")+timedelta(days=10)
        start2=start.strftime("%Y-%m-%d")
        for i in range(0,len(df.date)):
            y.append(df[self.state].values[i])
            x.append(df.date.values[i])
        df2=pd.DataFrame(data=y,index=x,columns=[""])
        df2 =df2.apply (pd.to_numeric, errors='coerce')
        df2[start2:] = df2[start2:].replace({0:np.nan})
        df2 = df2.dropna()
        df2.index = pd.DatetimeIndex(df2.index)
        #interpolate missing data
        df2 = df2.reindex(pd.date_range(df2.index.min(), df2.index.max()), fill_value=np.nan)
        df2 = df2.interpolate(method='akima', axis=0).ffill().bfill()
        #string type for dates and integer for data
        df2 = df2.astype(int)
        df2.index = df2.index.astype(str)
        df2=df2[self.start_date:]
        return df2
    
    def extend_index(self):
        #values = index.values
        #current = datetime.strptime(index[-1], '%Y-%m-%d')
        '''
        while len(values) < new_size:
            print(current)
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%Y-%m-%d'))
        '''
        endData = datetime.strptime(self.data.index[-1], '%Y-%m-%d')
        end = datetime.strftime(endData  + timedelta(days=self.predict_range), '%Y-%m-%d')
        values = pd.date_range(start=self.data.index[0],end=end)
        return list(values)
    
    def create_lossOdeint(self):
            
        def lossOdeint(point):
            size = len(self.data)+1
            sigma=[]
            beta0, sigma01, sigma02, startT, startT2, sigma0,  a, b, betaR, nu, mu = point
            p=0.4

            def SEAIRD(y,t):
                gamma=a+b
                sigma=sg.sigmoid2(t-startT,t-startT2,sigma0,sigma01,sigma02,t-int(size*self.sigmoidTime+0.5))
                beta=beta0
                S = y[0]
                E = y[1]
                A = y[2]
                I = y[3]
                R = y[4]
                D = y[5]
                y0=(-(A*betaR+I)*beta*S) #S
                y1=(A*betaR+I)*beta*S-sigma*E-mu*E #E
                y2=sigma*E*(1-p)-gamma*A #A
                y3=sigma*E*p-gamma*I-mu*I #I
                y5=a*I-nu*D+mu*(E+I+R)#D
                y4=(-(y0+y1+y2+y3+y5)) #R
                if y5<0:
                    y4=y4-y5
                    y5=0
                return [y0,y1,y2,y3,y4,y5]

            y0=[self.s_0,self.e_0,self.a_0,self.i_0,self.r_0,self.d_0]
            tspan=np.arange(0, size+200, 1)
            res, info =odeint(SEAIRD,y0,tspan,atol=1e-4, rtol=1e-6, full_output=True, 
                              mxstep=500000,hmin=1e-12)   
            res = np.where(res < 0, 0, res)
            res = np.where(res >= 1e10, 1e10, res)
#             res = res.astype(int)

            # calculate fitting error
            ix= np.where(self.data.values >= self.startNCases)
            
            #cases
            l1 = np.mean((res[ix[0],3] - (self.data.values[ix]))**2)
            
            #deaths
            l2 = (res[ix[0],5] - self.death.values[ix])**2
            sizeD=len(l2)
            l2Final=np.mean(l2[sizeD-8:sizeD])
            l2=np.mean(l2)

            #calculate derivatives
            #and the error of the derivative between prediction and the data

            #for deaths
            dDeath=np.diff(res[1:size,5])           
            dDeathData=np.diff(self.death.values.T[:])
            dErrorD=np.mean(((dDeath-dDeathData)**2)[-8:]) 

            #for infected
            dInf=np.diff(res[1:size,3])
            dInfData=np.diff(self.data.values.T[:])          
            dErrorI=np.mean(((dInf-dInfData)**2)[-8:])

            if self.Deaths:
                #penalty function for negative derivative at end of deaths
                NegDeathData=np.diff(res[:,5])
                dNeg=np.mean(NegDeathData[-5:])
                correctGtot=max(0,np.sign(dNeg))*(dNeg)**2
                del NegDeathData
            else:
                correctGtot=0
                dNeg=0
            
            if self.propWeigth:
                wt=self.weigthCases+self.weigthDeath
            else:
                wt=1
                
            wCases=self.weigthCases/wt
            wDeath=self.weigthDeath/wt                                                                          
                
            #objective function
            gtot=wCases*(l1+0.05*dErrorI) + wDeath*(8*l2+dErrorD+4*l2Final)

            del l1, l2, correctGtot, dNeg, dErrorI, dErrorD,dInfData, dInf, dDeathData, dDeath
            
            return gtot
        return lossOdeint

    #predict final extended values
    def predict(self, point):

        beta0, sigma01, sigma02, startT, startT2, sigma0,  a, b, betaR, nu, mu  = point
        new_index = self.extend_index()
        size = len(new_index)
        sizeData = len(self.data)+1
        p=0.4
            
        def SEAIRD(y,t):
            gamma=a+b
            sigma=sg.sigmoid2(t-startT,t-startT2,sigma0,sigma01,sigma02,t-int(sizeData*self.sigmoidTime+0.5))
            beta=beta0
            S = y[0]
            E = y[1]
            A = y[2]
            I = y[3]
            R = y[4]
            D = y[5]
            y0=(-(A*betaR+I)*beta*S) #S
            y1=(A*betaR+I)*beta*S-sigma*E-mu*E #E
            y2=sigma*E*(1-p)-gamma*A #A
            y3=sigma*E*p-gamma*I-mu*I #I
            y5=a*I-nu*D+mu*(E+I+R) #D
            y4=(-(y0+y1+y2+y3+y5)) #R
            if y5<0:
                y4=y4-y5
                y5=0
            return [y0,y1,y2,y3,y4,y5]

        y0=[self.s_0,self.e_0,self.a_0,self.i_0,self.r_0,self.d_0]
        tspan=np.arange(0, size, 1)
        res, info =odeint(SEAIRD,y0,tspan,atol=1e-4, rtol=1e-6, full_output=True, 
                              mxstep=500000,hmin=1e-12)           
        res = np.where(res < 0, 0, res)
        res = np.where(res >= 1e10, 1e10, res)
#         res=res.astype(int)

        return new_index, res[:,0], res[:,1],res[:,2],res[:,3],res[:,4], res[:,5]


    #run optimizer and plotting
    def train(self):
        
        self.death= self.load_dead()
        self.recovered = self.load_confirmed()*self.ratio
        self.data = self.load_confirmed()-self.recovered-self.death
        
        size=len(self.data)+1
          
        bounds =[(1e-16, .9),(1e-16, .9),(1e-16, .9),
            (5,int(size*self.sigmoidTime+0.5)-5),(int(size*self.sigmoidTime+0.5)+5,size-5),
            (1e-16, .9),
            (1e-16, 10),(1e-16, 10),
            (0,10),(1e-16,10),(1e-16,10)
            ]            

        maxiterations=4500
        f=self.create_lossOdeint()
        de = DE(f, bounds, maxiters=maxiterations) #,popsize=100)
        i=0
        with tqdm(total=maxiterations*1750*maxiterations/3500) as pbar:
            for step in de.geniterator():
                idx = step.best_idx
                norm_vector = step.population[idx]
                best_params = de.denormalize([norm_vector])
                pbar.update(i)
                i+=1
        p=best_params[0]

        today = datetime.today()
        endDate = today + timedelta(days=-2)
        self.end_date= datetime.strftime(endDate, '%Y-%m-%d') 
        
        self.death= self.load_dead()
        self.recovered = self.load_confirmed()*self.ratio
        self.data = self.load_confirmed()-self.recovered-self.death
        
        new_index, y0, y1, y2, y3, y4, y5 = self.predict(p)

        #prepare dataframe to export
        dataFr = [y0, y1, y2, y3, y4, y5]
        dataFr2 = np.array(dataFr).T
        df = pd.DataFrame(data=dataFr2)
        df.columns  = ['Susceptible','Exposed','Asymptomatic','Infected','Recovered','Deaths']
        df.index = pd.date_range(start=new_index[0], 
            end=new_index[-1])
        df.index.name = 'date'
        
        del dataFr, dataFr2, idx, norm_vector, best_params, dead,\
            new_index, extended_actual, extended_death, y0, y1, y2, y3, y4, y5
        
        if self.savedata:
            #save simulation results for comparison and use in another codes/routines
            df.to_pickle('./data/SEAIRD_'+self.state+self.version+'.pkl')
            df.to_csv('./results/data/SEAIRD_'+self.state+self.version+'.csv', sep=",")
        return p