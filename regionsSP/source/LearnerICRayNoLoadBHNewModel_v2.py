#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import the necessary packages and modules
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import basinhopping
import sigmoid as sg
import gc
import ray

# ray.shutdown()
# ray.init(num_gpus=96,num_cpus=17,
#          ignore_reinit_error=True)

#register function for parallel processing
@ray.remote(num_cpus=1,num_gpus=4)
class Learner(object):
    def __init__(self, districtRegion, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, startNCases, ratio, weigthCases, weigthRecov, cleanRecovered, version, data, death, underNotif=True, Deaths=False, propWeigth=True, savedata=True):
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
        self.ratio = ratio
        self.weigthCases = weigthCases
        self.weigthRecov = weigthRecov
        self.weigthDeath = 1-weigthRecov-weigthCases
        self.cleanRecovered=cleanRecovered
        self.version=version
        self.data = data
        self.death = death
        self.Deaths = Deaths
        self.propWeigth = propWeigth
        self.underNotif = underNotif
        self.savedata = savedata

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
            beta0, beta01, startT, sigma,  a, b, d, mu, p = point
            gamma=a+b
            gamma2=d
            
            def SEAIRD(y,t):
                beta=sg.sigmoid(t-startT,beta0,beta01)
                S = y[0]
                E = y[1]
                A = y[2]
                I = y[3]
                R = y[4]
#                 p=0.4
                y0=(-(A+I)*beta*S-mu*S) #S
                y1=(A+I)*beta*S-sigma*E-mu*E #E
                y2=sigma*E*(1-p)-gamma2*A #A
                y3=sigma*E*p-gamma*I-mu*I #I
                y4=(b*I+d*A)#R
                y5=(-(y0+y1+y2+y3+y4)) #D
                return [y0,y1,y2,y3,y4,y5]

            y0=[self.s_0,self.e_0,self.a_0,self.i_0,self.r_0,self.d_0]
            size = len(self.data)+1
            tspan=np.arange(0, size, 1)
            res=odeint(SEAIRD,y0,tspan,atol=1e-4, rtol=1e-6)            
            res = np.where(res >= 1e10, 1e10, res)

            # calculate fitting error by using numpy.where
            ix= np.where(self.data.values >= self.startNCases)
            l1 = np.mean((res[ix[0],3] - self.data.values[ix])**2)
            l2 = np.mean((res[ix[0],5] - self.death.values[ix])**2)
            
            deltaRecSq=(np.divide(res[ix[0],4],res[ix[0],3])-self.ratio)**2
            l3 = np.mean(deltaRecSq)
#             l3 = np.mean((res[ix[0],4] - (self.recovered.values[ix]))**2)


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
                NegDeathData=np.diff(res[:,3])
                dNeg=np.mean(NegDeathData[-5:]) 
                correctGtot=(dNeg)**2
                del NegDeathData
            else:
                correctGtot=0
                dNeg=0
            
            if self.propWeigth:
                wt=self.weigthCases+self.weigthDeath+self.weigthRecov
            else:
                wt=1
                
            wCases=self.weigthCases/wt
            wDeath=self.weigthDeath/wt
            wRecov=self.weigthRecov/wt
                
            #objective function
            gtot=wCases*(l1*4+0.05*dErrorI) + wDeath*(l2*4+0.2*dErrorD) + wRecov*l3

            #final objective function
            gtot=10*correctGtot*np.sign(dNeg)+gtot

            del l1, l2, l3, correctGtot, dNeg, dErrorI, dErrorD,dInfData, dInf, dDeathData, dDeath
            
            return gtot
        return lossOdeint

    #run optimizer and plotting
    def train(self):
        
        f=self.create_lossOdeint()
        
        bnds = ((1e-12, .2),(1e-12, .2),(5,len(self.data)-5),
        (1/160, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(0,1))# your bounds

        x0 = [1e-3, 1e-3, 0, 1e-3, 1/50, 1e-3, 1e-3, 1e-3,0.4]
        minimizer_kwargs = { "method": "L-BFGS-B","bounds":bnds }
        optimal = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs,niter=10,disp=True)        
        point = self.s_0, self.start_date, self.i_0, self.d_0, self.startNCases, self.weigthCases, self.weigthRecov
        
        strSave='{}, {}, '.format(self.districtRegion, abs(optimal.fun))
        strSave=strSave+', '.join(map(str,point))
        self.append_new_line('./results/history_'+self.districtRegion+str(self.version)+'.csv', strSave) 
        
        del self, f, strSave, point
        gc.collect()

        return optimal.fun