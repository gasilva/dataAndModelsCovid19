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
from yabox import DE
from tqdm import tqdm

#parallel computation
import ray
ray.shutdown()
ray.init(num_cpus=20)

#register function for parallel processing
@ray.remote
class Learner(object):
    def __init__(self, state, loss, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, startNCases, ratio, weigthCases, weigthRecov, cleanRecovered, version, savedata=True):
        self.state = state
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
            beta, beta2, sigma, sigma2, sigma3, gamma, b, mu = point
            def SEAIRD(y,t):
                S = y[0]
                E = y[1]
                A = y[2]
                I = y[3]
                R = y[4]
                p=0.2
                # beta2=beta
                y0=-(beta2*A+beta*I)*S-mu*S #S
                y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
                y2=sigma*E*(1-p)-gamma*A-mu*A #A
                y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I #I
                y4=b*I+gamma*A+sigma2*I-mu*R #R
                y5=(-(y0+y1+y2+y3+y4)) #D
                return [y0,y1,y2,y3,y4,y5]

            y0=[s_0,e_0,a_0,i_0,r_0,d_0]
            tspan=np.arange(0, size, 1)
            res=odeint(SEAIRD,y0,tspan) 
            #,hmax=0.01)

            tot=0
            l1=0
            l2=0
            l3=0
            for i in range(0,len(data.values)):
                if data.values[i]>startNCases:
                    l1 = l1+(res[i,3] - data.values[i])**2
                    l2 = l2+(res[i,5] - death.values[i])**2
                    newRecovered=min(1e6,data.values[i]*ratioRecovered)
                    l3 = l3+(res[i,4] - newRecovered)**2
                    tot+=1
            l1=np.sqrt(l1/max(1,tot))
            l2=np.sqrt(l2/max(1,tot))
            l3=np.sqrt(l3/max(1,tot))

            #weight for cases
            u = weigthCases
            #weight for recovered
            w = weigthRecov 
            #weight for deaths
            v = max(0,1. - u - w)
            return u*l1 + v*l2 + w*l3 
        return lossOdeint
    

    #predict final extended values
    def predict(self, beta, beta2, sigma, sigma2, sigma3, gamma, b, mu, data, \
                    death, state, s_0, e_0, a_0, i_0, r_0, d_0):
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
        res=odeint(SEAIRD,y0,tspan)

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
        self.data = self.load_confirmed(self.state)
        self.death = self.load_dead(self.state)

        bounds=[(1e-12, .2),(1e-12, .2),(1/60 ,0.4),(1/60, .4),
        (1/60, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4)]

        maxiterations=1500
        f=self.create_lossOdeint(self.data, \
            self.death, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0, self.startNCases, \
                 self.ratio, self.weigthCases, self.weigthRecov)
        de = DE(f, bounds, maxiters=maxiterations)
        i=0
        with tqdm(total=maxiterations*1000) as pbar:
            for step in de.geniterator():
                idx = step.best_idx
                norm_vector = step.population[idx]
                best_params = de.denormalize([norm_vector])
                pbar.update(i)
                i+=1
        p=best_params[0]

        #parameter list for optimization
        #beta, beta2, sigma, sigma2, sigma3, gamma, b, mu

        beta, beta2, sigma, sigma2, sigma3, gamma, b, mu = p

        new_index, extended_actual, extended_death, y0, y1, y2, y3, y4, y5 \
                = self.predict(beta, beta2, sigma, sigma2, sigma3, gamma, b, mu, \
                    self.data, self.death, self.state, self.s_0, \
                    self.e_0, self.a_0, self.i_0, self.r_0, self.d_0)

        #prepare dataframe to export
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
            return optimal.fun