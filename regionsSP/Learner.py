#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import the necessary packages and modules
import sys
import math
import array
import operator
import sys
import json
import ssl
import os
import urllib.request
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import dateutil.parser
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

#parallel computation
import ray
ray.shutdown()
ray.init(num_cpus=3)

#register function for parallel processing
@ray.remote
class Learner(object):
    def __init__(self, districtRegion, loss, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, startNCases, ratio, weigthCases, weigthRecov, cleanRecovered, version, savedata=True):
        self.districtRegion = districtRegion
        self.loss = loss
        self.start_date = start_date
        self.predict_range = predict_range
        self.s_0 = s_0
        self.e_0 = e_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.d_0 = d_0
        self.a_0 = a_0
        self.startNCases = startNCases
        self.ratio = ratio
        self.weigthCases = weigthCases
        self.weigthRecov = weigthRecov
        self.cleanRecovered=cleanRecovered
        self.version=version
        self.savedata = savedata

    def load_confirmed(self, districtRegion):
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv('./data/DRS_confirmados.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
        y=[]
        x=[]
        for i in range(0,len(df.date)):
            y.append(df[districtRegion].values[i])
            x.append(df.date.values[i])
        df2=pd.DataFrame(data=y,index=x,columns=[""])
        df2=df2[self.start_date:]
        return df2

    def load_dead(self, districtRegion):
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv('./data/DRS_mortes.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
        y=[]
        x=[]
        for i in range(0,len(df.date)):
            y.append(df[districtRegion].values[i])
            x.append(df.date.values[i])
        df2=pd.DataFrame(data=y,index=x,columns=[""])
        df2=df2[self.start_date:]
        return df2
    
    def extend_index(self, index, new_size):
        #values = index.values
        #current = datetime.strptime(index[-1], '%Y-%m-%d')
        '''
        while len(values) < new_size:
            print(current)
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%Y-%m-%d'))
        '''
        start = datetime.strptime(index[0], '%Y-%m-%d')
        end = datetime.strftime(start  + timedelta(days=new_size), '%Y-%m-%d')
        start = datetime.strftime(start, '%Y-%m-%d')
        values = pd.date_range(start=start, 
            end=end)
        return list(values)

    #predict final extended values
    def predict(self, beta, beta2, sigma, sigma2, sigma3, gamma, b, mu, data, \
                    death, districtRegion, s_0, e_0, a_0, i_0, r_0, d_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SEAIRD(y,t):
            S = y[0]
            E = y[1]
            A = y[2]
            I = y[3]
            R = y[4]
            D = y[5]
            p=0.2
            # beta2=beta
            y0=-(beta2*A+beta*I)*S+mu*S #S
            y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
            y2=sigma*E*(1-p)-gamma*A-mu*A #A
            y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I#I
            y4=b*I+gamma*A+sigma2*I-mu*R #R
            y5=-(y0+y1+y2+y3+y4) #D
            return [y0,y1,y2,y3,y4,y5]

        y0=[s_0,e_0,a_0,i_0,r_0,d_0]
        tspan=np.arange(0, size, 1)
        res=odeint(SEAIRD,y0,tspan,hmax=0.01)

        #data not extended
        extended_actual = data.values
        extended_death = death.values

        #extending data does not work
        # x=[None]*(size - len(data.values))
        # extended_actual = np.concatenate((data.values, x))
        # extended_death = np.concatenate((death.values, x))

        return new_index, extended_actual, extended_death, res[:,0], res[:,1],res[:,2],res[:,3],res[:,4], res[:,5]

    #run optimizer and plotting
    def train(self):
        self.data = self.load_confirmed(self.districtRegion)
        self.death = self.load_dead(self.districtRegion)
        
        print_info = False
        
        if print_info:
            print("\n running model for "+self.districtRegion)
        optimal = minimize(self.loss,        
            [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
            args=(self.data, self.death, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0, \
                self.startNCases, self.ratio, self.weigthCases, self.weigthRecov),
            method='L-BFGS-B',
            bounds=[(1e-12, 50), (1e-12, 50), (1./160.,0.2),  (1./160.,0.2), (1./160.,0.2),\
                 (1e-16, 0.4), (1e-12, 0.2), (1e-12, 0.2)])
            #beta, beta2, sigma, sigma2, sigma3, gamma, b
        if print_info:
            print("\n", optimal)
        beta, beta2, sigma, sigma2, sigma3, gamma, b, mu = optimal.x
        new_index, extended_actual, extended_death, y0, y1, y2, y3, y4, y5 \
                = self.predict(beta, beta2, sigma, sigma2, sigma3, gamma, b, mu, self.data, \
                self.death, self.districtRegion, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0)

        dataFr = [y0, y1, y2, y3, y4, y5]
        dataFr2 = np.array(dataFr).T
        df = pd.DataFrame(data=dataFr2)
        df.columns  = ['Susceptible','Exposed','Asymptomatic','Infected','Recovered','Deaths']
        
        df.index = pd.date_range(start=new_index[0], 
            end=new_index[-1])
        df.index.name = 'date'
        
        if self.savedata:
            #save simulation results for comparison and use in another codes/routines
            df.to_pickle('./data/SEAIRD_sigmaOpt_'+self.districtRegion+'.pkl')
            df.to_csv('./results/data/SEAIRD_sigmaOpt_'+self.districtRegion+'.csv', sep=",")
        else:
            return optimal.fun