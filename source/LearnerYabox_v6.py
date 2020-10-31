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
import yabox as yb
from yabox.algorithms import DE
print('Yabox version: ', yb.__version__)
import sigmoidOnly as sg
from ipywidgets import FloatProgress
from IPython.display import display
from tqdm import tqdm
from ipywidgets import FloatProgress, HBox, VBox, HTML

#parallel computation
import ray
ray.shutdown()
ray.init(num_cpus=1,num_gpus=5,memory=230*1024*1024*1024)

#register function for parallel processing
@ray.remote(memory=10*1024*1024*1024)
class Learner(object):
    def __init__(self, country, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, \
    startNCases, weigthCases, weigthRecov, weigthDeath, end_date, cleanRecovered, version, \
                 underNotif=True, Deaths=False, propWeigth=True, savedata=True):
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
        self.weigthDeath = weigthDeath
        self.cleanRecovered=cleanRecovered
        self.version=version
        self.savedata = savedata
        self.under = underNotif
        self.end_date = end_date

    def load_confirmed(self):
        df = pd.read_csv('data/time_series_19-covid-Confirmed-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        return country_df.iloc[0].loc[self.start_date:self.end_date] 

    def load_recovered(self):
        df = pd.read_csv('data/time_series_19-covid-Recovered-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        return country_df.iloc[0].loc[self.start_date:self.end_date]

    def load_dead(self):
        df = pd.read_csv('data/time_series_19-covid-Deaths-country.csv')
        country_df = df[df['Country/Region'] == self.country]
        return country_df.iloc[0].loc[self.start_date:self.end_date]
    
    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values
    
    def create_lossOdeint(self):

        def lossOdeint(point):
            size = len(self.data)
            beta0, beta01, startT, beta2, sigma, sigma2, sigma3,  a, b, c, d, mu = point
            gamma=a+b
            gamma2=c+d
            
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

            y0=[self.s_0,self.e_0,self.a_0,self.i_0,self.r_0,self.d_0]
            size = len(self.data)+1
            tspan=np.arange(0, size+200, 1)
            res=odeint(SEAIRD,y0,tspan,mxstep=5000000) #,hmax=0.01)
            res = np.where(res >= 1e10, 1e10, res)

            # calculate fitting error by using numpy.where
            ix= np.where(self.data.values >= self.startNCases)
            l1 = np.mean((res[ix[0],3] - (self.data.values[ix]))**2)
            l2 = np.mean((res[ix[0],5] - self.death.values[ix])**2)
            l3 = np.mean((res[ix[0],4] - (self.recovered.values[ix]))**2)

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
            
            weigthDeath=1
            weigthRecov=1
            weigthCases=1
            
            #objective function
            gtot=weigthCases*(l1+0.05*dErrorI) + weigthDeath*(l2+0.2*dErrorD) + weigthRecov*l3
            
            #penalty function for negative derivative at end of deaths
            NegDeathData=np.diff(res[:,5])
            dNeg=np.mean(NegDeathData[-5:]) 
            correctGtot=max(abs(dNeg),0)**2

            #final objective function
            gtot=abs(10*min(np.sign(dNeg),0)*correctGtot)+abs(gtot)

            del l1, l2, l3, NegDeathData, dNeg, correctGtot, dErrorI, dInfData, dInf, dErrorD, dDeathData, dDeath
            return gtot
        return lossOdeint

    def create_lossSub(self,p):

        beta0, beta01, startT, beta2, sigma, sigma2, sigma3, a, b, c, d, mu = p        
        
        def lossSub(point):
            sub, subRec, subDth = point
            gamma=a+b
            gamma2=c+d
            
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

            y0=[self.s_0,self.e_0,self.a_0,self.i_0,self.r_0,self.d_0]
            size = len(self.data)+1
            tspan=np.arange(0, size+200, 1)
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
            
            weigthDeath=1
            weigthRecov=1
            weigthCases=1
            
            #objective function
            gtot=weigthCases*(l1+0.05*dErrorI) + weigthDeath*(l2+0.2*dErrorD) + weigthRecov*l3
            
            #penalty function for negative derivative at end of deaths
            NegDeathData=np.diff(res[:,5])
            dNeg=np.mean(NegDeathData[-5:]) 
            correctGtot=max(abs(dNeg),0)**2

            #final objective function
            gtot=abs(10*min(np.sign(dNeg),0)*correctGtot)+abs(gtot)

            del l1, l2, l3, NegDeathData, dNeg, correctGtot, dErrorI, dInfData, dInf, dErrorD, dDeathData, dDeath
            return gtot 
        return lossSub    
    
    

    #predict final extended values
    def predict(self, p):

        beta0, beta01, startT, beta2, sigma, sigma2, sigma3, a, b, c, d, mu, sub, subRec, subDth  = p
        new_index = self.extend_index(self.data.index, 300)
        size = len(new_index)
        gamma=a+b
        gamma2=c+d

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

        y0=[self.s_0,self.e_0,self.a_0,self.i_0,self.r_0,self.d_0]
        tspan=np.arange(0, size, 1)
        res=odeint(SEAIRD,y0,tspan,mxstep=5000000) #,hmax=0.01)
        res = np.where(res >= 1e10, 1e10, res)
        res[:,3]=sub*res[:,3]
        res[:,4]=subRec*res[:,4]
        res[:,5]=subDth*res[:,5]

        #data not extended
        extended_actual = np.concatenate((self.data.values, \
                            [None] * (size - len(self.data.values))))
        extended_recovered = np.concatenate((self.recovered.values, \
                            [None] * (size - len(self.recovered.values))))
        extended_death = np.concatenate((self.death.values, \
                            [None] * (size - len(self.death.values))))

        del p,tspan,gamma,gamma2,size

        return new_index, extended_actual, extended_death, extended_recovered,\
               res[:,0], res[:,1],res[:,2],res[:,3],res[:,4], res[:,5]

    #run optimizer and plotting
    def train(self):

        self.death= self.load_dead()
        self.recovered = self.load_recovered()
        self.data = self.load_confirmed()-self.recovered-self.death
        
        size=len(self.data)
        bounds=[(1e-12, .2),(1e-12, .2),(5,size-5),(1e-12, .2),(1/120 ,0.4),(1/120, .4),
            (1/120, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4)]

        maxiterations=5500
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
        beta0, beta01, startT, beta2, sigma, sigma2, sigma3, a, b, c, d, mu, sub, subRec, subDth  = p
        print("country {}".format(self.country))
        print("under notifications cases {:.2f}".format(p2[0]))
        print("under notifications recovered {:.2f}".format(p2[1]))
        print("under notifications deaths {:.2f}".format(p2[2]))

        today = datetime.today()
        endDate = today + timedelta(days=-2)
        self.end_date= datetime.strftime(endDate, '%-m/%-d/%y') 
        self.death= self.load_dead()
        self.recovered = self.load_recovered()
        self.data = self.load_confirmed()-self.recovered-self.death

        new_index, extended_actual, extended_death, extended_recovered, y0, y1, y2, y3, y4, y5 \
                = self.predict(p)

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