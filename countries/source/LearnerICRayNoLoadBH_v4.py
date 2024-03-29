#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import the necessary packages and modules
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import basinhopping
import sigmoidOnly as sg
import gc
import ray

#register function for parallel processing
@ray.remote(memory=5*1024*1024*1024)
class Learner(object):
    def __init__(self, country, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, startNCases, \
                 cleanRecovered, version, data, death, recovered, savedata=True):
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
        self.cleanRecovered= cleanRecovered
        self.version=version
        self.savedata = savedata
        self.data = data
        self.death = death
        self.recovered = recovered

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

    def create_lossOdeint(self):

        def lossOdeint(point):
            size = len(self.data)
            beta0, beta01, startT, beta2, sigma, sigma2, sigma3,  a, b, c, d, mu = point
            gamma = a + b
            gamma2 = c + d
            def SEAIRD(y,t):
                rx=sg.sigmoid(t-startT,beta0,beta01)
                beta=beta0*rx+beta01*(1-rx)
                S = y[0]
                E = y[1]
                A = y[2]
                I = y[3]
                R = y[4]
                p=0.4
                y0=-(beta2*A+beta*I)*S-mu*S #S
                y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
                y2=sigma*E*(1-p)-mu*A-gamma2*A #A
                y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I #I
                y4=b*I+d*A+sigma2*I-mu*R #R
                y5=(-(y0+y1+y2+y3+y4)) #D
                return [y0,y1,y2,y3,y4,y5]

            y0=[self.s_0,self.e_0,self.a_0,self.i_0,self.r_0,self.d_0]
            size = len(self.data)+1
            tspan=np.arange(0, size+200, 1)
            res=odeint(SEAIRD,y0,tspan,mxstep=5000000) #,hmax=0.01)
            res = np.where(res >= 1e10, 1e10, res)
            
            # calculate fitting error by using numpy.where
            ix= np.where(self.data.values >= self.startNCases)
            l1 = np.mean((res[ix[0],3] - self.data.values[ix])**2)
            l2 = np.mean((res[ix[0],5] - self.death.values[ix])**2)
            l3 = np.mean((res[ix[0],4] - self.recovered.values[ix])**2)

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
            
            #objective function
            gtot=(l1+0.05*dErrorI) + (l2+0.2*dErrorD) + l3
            
            #penalty function for negative derivative at end of deaths
            NegDeathData=np.diff(res[:,5])
            dNeg=np.mean(NegDeathData[-5:]) 
            correctGtot=max(abs(dNeg),0)**2

            #final objective function
            gtot=abs(10*min(np.sign(dNeg),0)*correctGtot)+abs(gtot)

            del l1, l2, l3, NegDeathData, dNeg, correctGtot, dErrorI, dInfData, dInf, dErrorD, dDeathData, dDeath
            return gtot
        return lossOdeint

    #run optimizer and plotting
    def train(self):
        
        f=self.create_lossOdeint()
        bnds = ((1e-12, .2),(1e-12, .2),(5,len(self.data)-5),(1e-12, .2),(1/120 ,0.4),(1/120, .4),
        (1/120, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4))# your bounds
        x0 = [1e-3, 1e-3, 0, 1e-3, 1/120, 1/120, 1/120, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
        minimizer_kwargs = { "method": "L-BFGS-B","bounds":bnds }
        optimal = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs,niter=10,disp=True)  
        
        self.weigthCases=1
        self.weigthRecov=1
        self.weigthDeath=1

        point = self.s_0, self.start_date, self.i_0, self.d_0, self.r_0, self.startNCases, self.weigthCases, \
                    self.weigthRecov, self.weigthDeath
        
        strSave='{}, {}, '.format(self.country, abs(optimal.fun))
        strSave=strSave+', '.join(map(str,point))
        self.append_new_line('./results/history_'+self.country+str(self.version)+'.csv', strSave) 
        
        del self, f, strSave, point
        gc.collect()

        return optimal.fun