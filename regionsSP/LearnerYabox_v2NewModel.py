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
env.int("CUDA_VISIBLE_DEVICES",1)
env.int("NUMBA_ENABLE_CUDASIM",1)
env.bool("OMPI_MCA_opal_cuda_support",True)

#parallel computation
import ray
ray.shutdown()
ray.init(num_cpus=6,num_gpus=4) #,log_to_driver=False,ignore_reinit_error=True)  #,memory=230*1024*1024*1024)

import unicodedata

#register function for parallel processing
@ray.remote(memory=10*1024*1024*1024)
class Learner(object):
    def __init__(self, districtRegion, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, \
    startNCases, weigthCases, weigthRecov, weigthDeath, end_date, cleanRecovered, version, \
                 underNotif=True, Deaths=True, propWeigth=True, savedata=True):
        self.districtRegion = districtRegion
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
        df = pd.read_csv('./data/DRS_confirmados.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
        y=[]
        x=[]
        for i in range(0,len(df.date)):
            y.append(df[self.districtRegion].values[i])
            x.append(df.date.values[i])
        df2=pd.DataFrame(data=y,index=x,columns=[""])
        df2 =df2.apply (pd.to_numeric, errors='coerce')
        df2 = df2.dropna()
        df2.index = pd.DatetimeIndex(df2.index)
        df2 = df2.reindex(pd.date_range(df2.index.min(), df2.index.max()), fill_value=np.nan)
        df2 = df2.interpolate(method='akima', axis=0).ffill().bfill()
        df2 = df2.astype(int)
        df2.index = df2.index.astype(str)
        df2=df2[self.start_date:]
        return df2

    def load_dead(self):
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv('./data/DRS_mortes.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
        y=[]
        x=[]
        for i in range(0,len(df.date)):
            y.append(df[self.districtRegion].values[i])
            x.append(df.date.values[i])
        df2=pd.DataFrame(data=y,index=x,columns=[""])
        df2 =df2.apply (pd.to_numeric, errors='coerce')
        df2 = df2.dropna()
        df2.index = pd.DatetimeIndex(df2.index)
        df2 = df2.reindex(pd.date_range(df2.index.min(), df2.index.max()), fill_value=np.nan)
        df2 = df2.interpolate(method='akima', axis=0).ffill().bfill()
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
            size = len(self.data)
            beta0, beta01, startT, beta2, sigma,  a, b, c, d, mu = point
            gamma=a+b
            gamma2=c+d
            
            def SEAIRD(y,t):
                beta=sg.sigmoid(t-startT,beta0,beta01)
                S = y[0]
                E = y[1]
                A = y[2]
                I = y[3]
                R = y[4]
                p=0.4
                y0=(-(beta2*A+beta*I)*S-mu*S) #S
                y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
                y2=sigma*E*(1-p)-mu*A-gamma2*A #A
                y3=sigma*E*p-gamma*I-mu*I #I
                y4=(b*I+d*A-mu*R)#R
                y5=(-(y0+y1+y2+y3+y4)) #D
                return [y0,y1,y2,y3,y4,y5]

            y0=[self.s_0,self.e_0,self.a_0,self.i_0,self.r_0,self.d_0]
            size = len(self.data)+1
            tspan=np.arange(0, size+100, 1)
            res=odeint(SEAIRD,y0,tspan,mxstep=5000000) #,hmax=0.01)
            res = np.where(res >= 1e10, 1e10, res)

            # calculate fitting error by using numpy.where
            ix= np.where(self.data.values >= self.startNCases)
            l1 = np.mean((res[ix[0],3] - (self.data.values[ix]))**2)
            l2 = np.mean((res[ix[0],5] - self.death.values[ix])**2)
            l3 = np.mean((res[ix[0],4] - (self.recovered.values[ix]))**2)

            if self.Deaths:
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

                #penalty function for negative derivative at end of deaths
                NegDeathData=np.diff(res[:,5])
                dNeg=np.mean(NegDeathData[-5:]) 
                correctGtot=max(abs(dNeg),0)**2
                del NegDeathData, dInfData, dInf, dDeathData, dDeath
            else:
                correctGtot=0
                dNeg=0
                dErrorI=0
                dErrorD=0
            
            if self.propWeigth:
                wt=self.weigthCases+self.weigthDeath+self.weigthRecov
                self.weigthCases=self.weigthCases/wt
                self.weigthDeath=self.weigthDeath/wt
                self.weigthRecov=self.weigthRecov/wt
                
            #objective function
            gtot=self.weigthCases*(l1+0.05*dErrorI) + self.weigthDeath*(l2+0.2*dErrorD) + self.weigthRecov*l3

            #final objective function
            gtot=abs(10*min(np.sign(dNeg),0)*correctGtot)+abs(gtot)
            
            del l1, l2, l3, correctGtot, dNeg, dErrorI, dErrorD
            
            return gtot
        return lossOdeint

    def create_lossSub(self,p):

        beta0, beta01, startT, beta2, sigma, a, b, c, d, mu = p        
        
        def lossSub(point):
            sub, subRec, subDth = point
            gamma=a+b
            gamma2=c+d
            
            def SEAIRD(y,t):
                beta=sg.sigmoid(t-startT,beta0,beta01)
                S = y[0]
                E = y[1]
                A = y[2]
                I = y[3]
                R = y[4]
                p=0.4
                y0=(-(beta2*A+beta*I)*S-mu*S) #S
                y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
                y2=sigma*E*(1-p)-mu*A-gamma2*A #A
                y3=sigma*E*p-gamma*I-mu*I #I
                y4=(b*I+d*A-mu*R)#R
                y5=(-(y0+y1+y2+y3+y4)) #D
                return [y0,y1,y2,y3,y4,y5]

            y0=[self.s_0,self.e_0,self.a_0,self.i_0,self.r_0,self.d_0]
            size = len(self.data)+1
            tspan=np.arange(0, size, 1)
            res=odeint(SEAIRD,y0,tspan,mxstep=5000000) #,hmax=0.01)
            res = np.where(res >= 1e10, 1e10, res)
            res[:,3]=sub*res[:,3]
            res[:,4]=subRec*res[:,4]
            res[:,5]=subDth*res[:,5]

            # calculate fitting error by using numpy.where
            ix= np.where(self.data.values >= self.startNCases)
            l1 = np.mean((res[ix[0],3] - (self.data.values[ix]))**2)
            l2 = np.mean((res[ix[0],5] - self.death.values[ix])**2)
            l3 = np.mean((res[ix[0],4] - (self.recovered.values[ix]))**2)

            if self.Deaths:
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

                #penalty function for negative derivative at end of deaths
                NegDeathData=np.diff(res[:,5])
                dNeg=np.mean(NegDeathData[-5:]) 
                correctGtot=max(abs(dNeg),0)**2
                del NegDeathData, dInfData, dInf, dDeathData, dDeath
            else:
                correctGtot=0
                dNeg=0
                dErrorI=0
                dErrorD=0

            if self.propWeigth:
                wt=self.weigthCases+self.weigthDeath+self.weigthRecov
                weigthCases=self.weigthCases/wt
                weigthDeath=self.weigthDeath/wt
                weigthRecov=self.weigthRecov/wt
                
            #objective function
            gtot=self.weigthCases*(l1+0.05*dErrorI) + self.weigthDeath*(l2+0.2*dErrorD) + self.weigthRecov*l3

            #final objective function
            gtot=abs(10*min(np.sign(dNeg),0)*correctGtot)+abs(gtot)
            
            del l1, l2, l3, correctGtot, dNeg, dErrorI, dErrorD
            return gtot 
        return lossSub     
    
    

    #predict final extended values
    def predict(self, p):

        beta0, beta01, startT, beta2, sigma, a, b, c, d, mu, sub, subRec, subDth  = p
        new_index = self.extend_index()
        size = len(new_index)
        gamma=a+b
        gamma2=c+d

        def SEAIRD(y,t):
            beta=sg.sigmoid(t-startT,beta0,beta01)
            S = y[0]
            E = y[1]
            A = y[2]
            I = y[3]
            R = y[4]
            p=0.4
            y0=(-(beta2*A+beta*I)*S-mu*S) #S
            y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
            y2=sigma*E*(1-p)-mu*A-gamma2*A #A
            y3=sigma*E*p-gamma*I-mu*I #I
            y4=(b*I+d*A-mu*R)#R
            y5=(-(y0+y1+y2+y3+y4)) #D
            return [y0,y1,y2,y3,y4,y5]

        y0=[self.s_0,self.e_0,self.a_0,self.i_0,self.r_0,self.d_0]
        tspan=np.arange(0, size, 1)
        res=odeint(SEAIRD,y0,tspan,mxstep=5000000) #,hmax=0.01)
        res = np.where(res >= 1e10, 1e10, res)
        res[:,3]=sub*res[:,3]
        res[:,4]=subRec*res[:,4]
        res[:,5]=subDth*res[:,5]

        return new_index, res[:,0], res[:,1],res[:,2],res[:,3],res[:,4], res[:,5]

    #run optimizer and plotting
    def train(self):

        self.death= self.load_dead()
        self.data = self.load_confirmed()*(1-0.15)-self.death
        self.recovered = self.load_confirmed()*0.15
        
        size=len(self.data)
#         bounds=[(1e-12, .2),(1e-12, .2),(5,size-5),(1e-12, .2),
#             (1/120, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4)]
        bounds=[(1e-16, .2),(1e-16, .2),(5,size-5),(1e-16, .2),
            (1/120, .4),(1e-16, .4),(1e-16, .4),(1e-16, .4),(1e-16, .4),(1e-16, .4)]

        maxiterations=3500
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
        
        if self.under:
            f=self.create_lossSub(p)
            bnds = ((.1,4),(.1,4),(.1,4))
            x0 = [0.9, 0.9, 0.9]
            minimizer_kwargs = { "method": "L-BFGS-B","bounds":bnds }
            optimal = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs,disp=True,niter=100) 
            p2=optimal.x
        else:
            p2=[1,1,1]
        
        p = np.concatenate((p, p2))
        beta0, beta01, startT, beta2, sigma, a, b, c, d, mu, sub, subRec, subDth  = p
        print("districtRegion {}".format(self.districtRegion))
        print("under notifications cases {:.2f}".format(p2[0]))
        print("under notifications recovered {:.2f}".format(p2[1]))
        print("under notifications deaths {:.2f}".format(p2[2]))

        today = datetime.today()
        endDate = today + timedelta(days=-2)
        self.end_date= datetime.strftime(endDate, '%Y-%m-%d') 
        self.death= self.load_dead()
        self.data = self.load_confirmed()*(1-0.15)-self.death
        self.recovered = self.load_confirmed()*0.15
        
        new_index, y0, y1, y2, y3, y4, y5 = self.predict(p)

        #prepare dataframe to export
        dataFr = [y0, y1, y2, y3, y4, y5]
        dataFr2 = np.array(dataFr).T
        df = pd.DataFrame(data=dataFr2)
        df.columns  = ['Susceptible','Exposed','Asymptomatic','Infected','Recovered','Deaths']
        df.index = pd.date_range(start=new_index[0], 
            end=new_index[-1])
        df.index.name = 'date'
        
        if self.savedata:
            dR = self.strip_accents(self.districtRegion)
            #save simulation results for comparison and use in another codes/routines
            df.to_pickle('./data/SEAIRD_sigmaOpt_'+dR+'.pkl')
            df.to_csv('./results/data/SEAIRD_sigmaOpt_'+dR+'.csv', sep=",")

        del dataFr, dataFr2, idx, norm_vector, best_params, df,\
            new_index, y0, y1, y2, y3, y4, y5            
            
        return p