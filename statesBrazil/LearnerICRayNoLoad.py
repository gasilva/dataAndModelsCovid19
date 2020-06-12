#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import the necessary packages and modules
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import sigmoid as sg
import gc
import ray

#register function for parallel processing
@ray.remote
class Learner(object):
    def __init__(self, state, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, startNCases, ratio, weigthCases, weigthRecov, cleanRecovered, version, data, death, savedata=True):
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
        self.data = data
        self.death = death

    def append_new_line(self,file_name, text_to_append):
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
            del data
            file_object.write(text_to_append)

    def create_lossOdeint(self, data, death, s_0, e_0, a_0, i_0, r_0, d_0, startNCases, \
                     ratioRecovered,weigthCases, weigthRecov):

        def lossOdeint(point):
            size = len(self.data)
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

            #calculate derivatives
            #and the error of the derivative between prediction and the data

            #for deaths
            dDeath=np.diff(res[1:size,5])
            dDeathData=np.diff(self.death.values.T[0][:])
            dErrorD=np.mean(((dDeath-dDeathData)**2)[-8:]) 

            #for infected
            dInf=np.diff(res[1:size,3])
            dInfData=np.diff(data.values.T[0][:])
            dErrorI=np.mean(((dInf-dInfData)**2)[-8:])

            #objective function
            gtot=weigthCases*(l1+0.05*dErrorI) + max(0,1. - weigthCases - weigthRecov)*(l2+0.2*dErrorD) + weigthRecov*l3

            #penalty function for negative derivative at end of deaths
            NegDeathData=np.diff(res[:,5])
            dNeg=np.mean(NegDeathData[-5:]) 
            correctGtot=max(abs(dNeg),0)**2

            #final objective function
            gtot=10*min(np.sign(dNeg),0)*correctGtot+gtot

            del NegDeathData, dInf, dInfData, dDeath, dDeathData, l1, l2, l3, correctGtot, dNeg, dErrorI, dErrorD
            return gtot
        return lossOdeint

    #run optimizer and plotting
    @ray.method(num_return_vals=1)
    def train(self):
        
        f=self.create_lossOdeint(self.data, \
            self.death, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0, self.startNCases, \
                 self.ratio, self.weigthCases, self.weigthRecov)

        optimal = minimize(f,        
            [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
            method='L-BFGS-B',
            bounds=[(1e-12, .2),(1e-12, .2),(5,len(self.data)-5),(1e-12, .2),(1/120 ,0.4),(1/120, .4),
        (1/120, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4)], options={'gtol': 1e-12, 'disp': False})
        
        point = self.s_0, self.start_date, self.i_0, self.weigthCases, self.weigthRecov
        
        strSave='{}, {}, '.format(self.state, abs(optimal.fun))
        strSave=strSave+', '.join(map(str,point))
        strSave=strSave+', '+', '.join(map(str,optimal.x))
        self.append_new_line('./results/history_'+self.state+str(self.version)+'.csv', strSave) 
        
        del self, f, strSave, point
        gc.collect()

        return optimal.fun