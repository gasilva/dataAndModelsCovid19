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
import sigmoidOnly as sg

#parallel computation
import ray
ray.shutdown()
ray.init(num_cpus=1,num_gpus=5,memory=230*1024*1024*1024)

#register function for parallel processing
@ray.remote(memory=10*1024*1024*1024)
class Learner(object):
    def __init__(self, country, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, \
    startNCases, weigthCases, weigthRecov, cleanRecovered, version, underNotif=False, savedata=True):
        self.country = country
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
        self.cleanRecovered=cleanRecovered
        self.version=version
        self.savedata = savedata

    def load_confirmed(self, country):
        df = pd.read_csv('data/time_series_19-covid-Confirmed-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:] 

    def load_recovered(self, country):
        df = pd.read_csv('data/time_series_19-covid-Recovered-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]

    def load_dead(self, country):
        df = pd.read_csv('data/time_series_19-covid-Deaths-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]
    
    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values
    
    def create_lossOdeint(self,data,\
                    death, recovered, s_0, e_0, a_0, i_0, r_0, d_0, startNCases, \
                    weigthCases, weigthRecov):

        def lossOdeint(point):
            size = len(self.data)
            beta0, beta01, startT, beta2, sigma, sigma2, sigma3,  gamma, b, gamma2, d, mu, sub, subRec = point
            def SEAIRD(y,t):
                rx=sg.sigmoid(t-startT,beta0,beta01)
                beta=beta0*rx+beta01*(1-rx)
                S = y[0]
                E = y[1]
                A = y[2]
                I = y[3]
                R = y[4]
                p=0.4
                y0=(-(beta2*A+beta*I)*S-mu*S) #S
                y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
                y2=sigma*E*(1-p)-mu*A-gamma2*A #A
                y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I #I
                y4=(b*I+d*A+sigma2*I-mu*R)#R
                y5=(-(y0+y1+y2+y3+y4)) #D
                return [y0,y1,y2,y3,y4,y5]

            y0=[s_0,e_0,a_0,i_0,r_0,d_0]
            size = len(data)+1
            tspan=np.arange(0, size+200, 1)
            res=odeint(SEAIRD,y0,tspan,mxstep=5000000) #,hmax=0.01)
            res = np.where(res >= 1e10, 1e10, res)
            res[:,3]=sub*res[:,3]
            res[:,4]=subRec*res[:,4]

            # calculate fitting error by using numpy.where
            ix= np.where(data.values >= startNCases)
            l1 = np.mean((res[ix[0],3] - (data.values[ix]))**2)
            l2 = np.mean((res[ix[0],5] - death.values[ix])**2)
            l3 = np.mean((res[ix[0],4] - (recovered.values[ix]))**2)

            #calculate derivatives
            #and the error of the derivative between prediction and the data

            #for deaths
            dDeath=np.diff(res[1:size,5])           
            dDeathData=np.diff(self.death.values.T[:])
            dErrorD=np.mean(((dDeath-dDeathData)**2)[-8:]) 

            #for infected
            dInf=np.diff(res[1:size,3])
            dInfData=np.diff(data.values.T[:])          
            dErrorI=np.mean(((dInf-dInfData)**2)[-8:])

            #objective function
            gtot=weigthCases*(l1+0.05*dErrorI) + max(0,1. - weigthCases - weigthRecov)*(l2+0.2*dErrorD) + weigthRecov*l3

            #penalty function for negative derivative at end of deaths
            NegDeathData=np.diff(res[:,5])
            dNeg=np.mean(NegDeathData[-5:]) 
            correctGtot=max(abs(dNeg),0)**2

            #final objective function
            gtot=abs(10*min(np.sign(dNeg),0)*correctGtot)+abs(gtot)

            del NegDeathData, dInf, dInfData, dDeath, dDeathData, l1, l2, l3, correctGtot, dNeg, dErrorI, dErrorD
            return gtot
        return lossOdeint
    

    #predict final extended values
    def predict(self, beta0, beta01, startT, beta2, sigma, sigma2, sigma3,  gamma, b, gamma2, d, mu, sub, subRec, data, \
                    death, recovered, country, s_0, e_0, a_0, i_0, r_0, d_0, predict_range):

        new_index = self.extend_index(data.index, 300)
        size = len(new_index)

        def SEAIRD(y,t):
            rx=sg.sigmoid(t-startT,beta0,beta01)
            beta=beta0*rx+beta01*(1-rx)
            S = y[0]
            E = y[1]
            A = y[2]
            I = y[3]
            R = y[4]
            p=0.4
            y0=(-(beta2*A+beta*I)*S-mu*S) #S
            y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
            y2=sigma*E*(1-p)-mu*A-gamma2*A #A
            y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I #I
            y4=(b*I+d*A+sigma2*I-mu*R)#R
            y5=(-(y0+y1+y2+y3+y4)) #D
            return [y0,y1,y2,y3,y4,y5]

        y0=[s_0,e_0,a_0,i_0,r_0,d_0]
        tspan=np.arange(0, size, 1)
        res=odeint(SEAIRD,y0,tspan,mxstep=5000000) #,hmax=0.01)
        res = np.where(res >= 1e10, 1e10, res)
        res[:,3]=sub*res[:,3]
        res[:,4]=subRec*res[:,4]
            
        #data not extended
        extended_actual = np.concatenate((data.values, \
                            [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, \
                            [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, \
                            [None] * (size - len(death.values))))

        return new_index, extended_actual, extended_death, extended_recovered,\
               res[:,0], res[:,1],res[:,2],res[:,3],res[:,4], res[:,5]

    #run optimizer and plotting
    def train(self):

        self.death= self.load_dead(self.country)
        self.recovered = self.load_recovered(self.country)
        self.data = self.load_confirmed(self.country)-self.recovered-self.death

        size=len(self.data)
        bounds=[(1e-12, .2),(1e-12, .2),(5,size-5),(1e-12, .2),(1/120 ,0.4),(1/120, .4),
            (1/120, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(.5,1.1),(.5,1.1)]

        maxiterations=10500
        f=self.create_lossOdeint(self.data, \
            self.death, self.recovered, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0, self.startNCases, \
                 self.weigthCases, self.weigthRecov)
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

        #parameter list for optimization
        beta0, beta01, startT, beta2, sigma, sigma2, sigma3, gamma, b, gamma2, d, mu, sub, subRec  = p

        new_index, extended_actual, extended_death, extended_recovered, y0, y1, y2, y3, y4, y5 \
                = self.predict(beta0, beta01, startT, beta2, sigma, sigma2, sigma3, gamma, b, gamma2, d, mu,sub, subRec,\
                    self.data, self.death, self.recovered, self.country, self.s_0, \
                    self.e_0, self.a_0, self.i_0, self.r_0, self.d_0, self.predict_range)

        #prepare dataframe to export
        df = pd.DataFrame({
                    'Susceptible': y0,
                    'Exposed': y1,
                    'Asymptomatic': y2,
                    'Infected data': extended_actual,
                    'Infected': y3,
                    'Recovered': extended_recovered,
                    'Predicted Recovered': y4,
                    'Death data': extended_death,
                    'Predicted Deaths': y5},
                    index=new_index)

        if self.savedata:
            #save simulation results for comparison and use in another codes/routines
            df.to_pickle('./data/SEAIRDv5_Yabox_'+self.country+'.pkl')
            df.to_csv('./results/data/SEAIRDv5_Yabox_'+self.country+'.csv', sep=",")

        del idx, norm_vector, best_params, df,\
            new_index, extended_actual, extended_death, extended_recovered, y0, y1, y2, y3, y4, y5            
            
        return p