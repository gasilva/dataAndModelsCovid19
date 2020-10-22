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
    def __init__(self, country, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0, startNCases, weigthCases, weigthRecov, weigthDeath, cleanRecovered, version, data, death, recovered, savedata=True):
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

    def create_lossOdeint(self, data, death, recovered, s_0, e_0, a_0, i_0, r_0, d_0, startNCases, \
                     weigthCases, weigthRecov,weigthDeath):

        def lossOdeint(point):
            size = len(self.data)
            beta0, beta01, startT, beta2, sigma, sigma2, sigma3,  a, b, c, d, mu, sub, subRec, subDth = point
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

            y0=[s_0,e_0,a_0,i_0,r_0,d_0]
            size = len(data)+1
            tspan=np.arange(0, size+200, 1)
            res=odeint(SEAIRD,y0,tspan,mxstep=5000000) #,hmax=0.01)
            res = np.where(res >= 1e10, 1e10, res)
            res[:,3]=sub*res[:,3]
            res[:,4]=subRec*res[:,4]
            res[:,5]=subDth*res[:,5]
            
            # calculate fitting error by using numpy.where
            ix= np.where(data.values >= startNCases)
            l1 = np.mean((res[ix[0],3] - data.values[ix])**2)
            l2 = np.mean((res[ix[0],5] - death.values[ix])**2)
            l3 = np.mean((res[ix[0],4] - recovered.values[ix])**2)

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
            
            wt=weigthCases+weigthRecov+weigthDeath
            wc=weigthCases/wt
            wr=weigthRecov/wt
            wd=weigthDeath/wt
            
            #objective function
            gtot=wc*(l1+0.05*dErrorI) + wd*(l2+0.2*dErrorD) + wr*l3

#             #penalty function for negative derivative at end of deaths
#             NegDeathData=np.diff(res[:,5])
#             dNeg=np.mean(NegDeathData[-5:]) 
#             correctGtot=max(abs(dNeg),0)**2

#             #final objective function
#             gtot=abs(10*min(np.sign(dNeg),0)*correctGtot)+abs(gtot)

            del dInf, dInfData, dDeath, dDeathData, l1, l2, l3, dErrorI, dErrorD
            return gtot
        return lossOdeint

    #run optimizer and plotting
    def train(self):
        
        f=self.create_lossOdeint(self.data, \
            self.death, self.recovered, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0, self.startNCases, \
                  self.weigthCases, self.weigthRecov,self.weigthDeath)

        bnds = ((1e-12, .2),(1e-12, .2),(5,len(self.data)-5),(1e-12, .2),(1/120 ,0.4),(1/120, .4),
        (1/120, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1.,1.),(1.,1.),(1.,1.))# your bounds
        x0 = [1e-3, 1e-3, 0, 1e-3, 1/120, 1/120, 1/120, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3,1e-3, 1e-3, 1e-3]
        minimizer_kwargs = { "method": "L-BFGS-B","bounds":bnds }
        optimal = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs,niter=10,disp=False)  
        p = optimal.x

        bnds = ((p[0], p[0]),(p[1], p[1]),(p[2],p[2]),(p[3], p[3]),(p[4] ,p[4]),(p[5], p[5]),
        (p[6], p[6]),(p[7], p[7]),(p[8], p[8]),(p[9], p[9]),(p[10], p[10]),(p[11], p[11]),(.5,2),(.5,2),(.5,2))
        x0 = [p[0], p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10],p[11],1.5, 0.9, 0.9]
        minimizer_kwargs = { "method": "L-BFGS-B","bounds":bnds }
        optimal = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs,niter=3,disp=True) 
        
        point = self.s_0, self.start_date, self.i_0, self.d_0, self.r_0, self.startNCases, self.weigthCases, \
                    self.weigthRecov, self.weigthDeath
        
        strSave='{}, {}, '.format(self.country, abs(optimal.fun))
        strSave=strSave+', '.join(map(str,point))
        self.append_new_line('./results/history_'+self.country+str(self.version)+'.csv', strSave) 
        
        del self, f, strSave, point
        gc.collect()

        return optimal.fun