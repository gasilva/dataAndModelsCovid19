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
import sigmoid as sg

#parallel computation
import ray
ray.shutdown()
ray.init()

#register function for parallel processing
@ray.remote
class Learner(object):
    def __init__(self, state, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, startNCases, ratio, weigthCases, weigthRecov, cleanRecovered, version, savedata=True):
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
        self.ratio = ratio
        self.weigthCases = weigthCases
        self.weigthRecov = weigthRecov
        self.cleanRecovered=cleanRecovered
        self.version=version
        self.savedata = savedata

    def load_confirmed(self, state):
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv('./data/confirmados.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
        y=[]
        x=[]
        for i in range(0,len(df.date)):
            y.append(df[state].values[i])
            x.append(df.date.values[i])
        df2=pd.DataFrame(data=y,index=x,columns=[""])
        df2=df2[self.start_date:]
        return df2

    def load_dead(self, state):
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv('./data/mortes.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
        y=[]
        x=[]
        for i in range(0,len(df.date)):
            y.append(df[state].values[i])
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

    def create_lossOdeint(self,data, \
                death, s_0, e_0, a_0, i_0, r_0, d_0, startNCases, \
                     ratioRecovered,weigthCases, weigthRecov):

        def lossOdeint(point):
            size = len(data)
            beta0, beta01, startT, beta2, sigma, sigma2, sigma3,  gamma, b, gamma2, d, mu = point
            def SEAIRD(y,t):
                rx=sg.sigmoid(t-startT)
                beta=beta0*rx+beta01*(1-rx)
                S = y[0]
                E = y[1]
                A = y[2]
                I = y[3]
                R = y[4]
                p=0.2
                y0=-(beta2*A+beta*I)*S-mu*S #S
                y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
                y2=sigma*E*(1-p)-mu*A-gamma2*A #A
                y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I #I
                y4=b*I+d*A+sigma2*I-mu*R #R
                y5=(-(y0+y1+y2+y3+y4)) #D
                return [y0,y1,y2,y3,y4,y5]

            y0=[s_0,e_0,a_0,i_0,r_0,d_0]
            size = len(data)+1
            tspan=np.arange(0, size+100, 1)
            res=odeint(SEAIRD,y0,tspan) #,hmax=0.01)

            # calculate fitting error by using numpy.where
            ix= np.where(data.values >= startNCases)
            l1 = np.mean((res[ix[0],3] - data.values[ix])**2)
            l2 = np.mean((res[ix[0],5] - death.values[ix])**2)
            l3 = np.mean((res[ix[0],4] - data.values[ix]*ratioRecovered)**2)

            #weight for cases
            u = weigthCases
            #weight for recovered
            w = weigthRecov 
            #weight for deaths
            v = max(0,1. - u - w)

            #calculate derivatives
            #and the error of the derivative between prediction and the data

            #for deaths
            dDeath=np.diff(res[1:size,5])
            dDeathData=np.diff(death.values.T[0][:])
            dErrorX=(dDeath-dDeathData)**2
            dErrorD=np.mean(dErrorX[-8:]) 

            #for infected
            dInf=np.diff(res[1:size,3])
            dInfData=np.diff(data.values.T[0][:])
            dErrorY=(dInf-dInfData)**2
            dErrorI=np.mean(dErrorY[-8:])

            #objective function
            gtot=u*(l1+0.05*dErrorI) + v*(l2+0.2*dErrorD) + w*l3

            #penalty function for negative derivative at end of deaths
            NegDeathData=np.diff(res[:,5])
            dNeg=np.mean(NegDeathData[-5:]) 
            correctGtot=max(abs(dNeg),0)**2

            #final objective function
            gtot=correctGtot-10*min(np.sign(dNeg),0)*correctGtot+gtot

            return gtot
        return lossOdeint

    #predict final extended values
#predict final extended values
    def predict(self, beta0, beta01, startT, beta2, sigma, sigma2, sigma3,  gamma, b, gamma2, d, mu, data, \
                    death, state, s_0, e_0, a_0, i_0, r_0, d_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)

        def SEAIRD(y,t):
            rx=sg.sigmoid(t-startT)
            beta=beta0*rx+beta01*(1-rx)
            S = y[0]
            E = y[1]
            A = y[2]
            I = y[3]
            R = y[4]
            p=0.2
            y0=-(beta2*A+beta*I)*S-mu*S #S
            y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
            y2=sigma*E*(1-p)-mu*A-gamma2*A #A
            y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I #I
            y4=b*I+d*A+sigma2*I-mu*R #R
            y5=(-(y0+y1+y2+y3+y4)) #D
            return [y0,y1,y2,y3,y4,y5]

        y0=[s_0,e_0,a_0,i_0,r_0,d_0]
        tspan=np.arange(0, size, 1)
        res=odeint(SEAIRD,y0,tspan)

        #data not extended
        extended_actual = data.values
        extended_death = death.values

        return new_index, extended_actual, extended_death, res[:,0], res[:,1],res[:,2],res[:,3],res[:,4], res[:,5]

    #run optimizer and plotting
    def train(self):
        dead=self.load_dead(self.state)
        self.data = self.load_confirmed(self.state)*(1-self.ratio)-dead
        self.death = dead
        
        print_info = False
        f=self.create_lossOdeint(self.data, \
            self.death, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0, self.startNCases, \
                 self.ratio, self.weigthCases, self.weigthRecov)
        
        size=len(self.data)

        if print_info:
            print("\n running model for "+self.state)
        optimal = minimize(f,        
            [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
            method='L-BFGS-B',
            bounds=[(1e-12, .2),(1e-12, .2),(5,size-5),(1e-12, .2),(1/120 ,0.4),(1/120, .4),
        (1/120, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4)], options={'gtol': 1e-8, 'disp': True})

        if print_info:
            print("\n", optimal)
    
        beta0, beta01, startT, beta2, sigma, sigma2, \
                           sigma3, gamma, b, gamma2, d, mu = optimal.x
        new_index, extended_actual, extended_death, y0, y1, y2, y3, y4, y5 \
                = self.predict(beta0, beta01, startT, beta2, sigma, sigma2, sigma3, gamma, b, gamma2, d, mu, \
                    self.data, self.death, self.state, self.s_0, \
                    self.e_0, self.a_0, self.i_0, self.r_0, self.d_0)

        dataFr = [y0, y1, y2, y3, y4, y5]
        dataFr2 = np.array(dataFr).T
        df = pd.DataFrame(data=dataFr2)
        df.columns  = ['Susceptible','Exposed','Asymptomatic','Infected','Recovered','Deaths']
        
        df.index = pd.date_range(start=new_index[0], 
            end=new_index[-1])
        df.index.name = 'date'
        
        if self.savedata:
            #save simulation results for comparison and use in another codes/routines
            df.to_pickle('./data/SEAIRD_'+self.state+self.version+'.pkl')
            df.to_csv('./results/data/SEAIRD_'+self.state+self.version+'.csv', sep=",")
        else:
            return optimal.fun, optimal.x