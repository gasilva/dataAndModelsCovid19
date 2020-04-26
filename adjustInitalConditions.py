import sys
import argparse
import json
import os
import urllib.request
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from csv import reader
from csv import writer
from datetime import datetime,timedelta
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.optimize import minimize

def fun(s0,countries,date,e0,a0,i0,r0,d0):
    for country in countries:
        command  = "python dataFit_SEAIRD_sgimaOptGlobalOptimumRay.py "
        # command  = "python dataFit_SEAIRD_sgimaOptGlobalOptimum.py "
    
        try:
            command  +=' --countries {co} '.format(co=country)
            command  +=' --start-date {d} '.format(d=date)
            command  +=' --S_0 {} '.format(int(s0))
            command  +=' --E_0 {} '.format(int(e0))
            command  +=' --A_0 {} '.format(int(a0))
            command  +=' --I_0 {} '.format(int(i0))
            command  +=' --R_0 {} '.format(int(r0))
            command  +=' --D_0 {} '.format(int(d0))
            print(command)  
            os.system(command )
        except Exception as e: 
            print(e)
    df= pd.read_pickle('./data/optimum.pkl')
    
    print('infected, recovered, death, total',df.g1,df.g2,df.g3,df.Total)
    return df.Total

date="2/25/2020"
s0=12000000
e0=0
a0=0
i0=265
r0=0
d0=0
countries=["Italy"]

df=pd.DataFrame([[1e6,1e6,1e6,1e6]], columns=['g1','g2','g3','Total'])
df.to_pickle('./data/optimum.pkl')

optimal = minimize(fun,        
    [3000000],
    args=(countries,date,e0,a0,i0,r0,d0),
    method='L-BFGS-B',
    bounds=[(100, 30e6)],options={'disp': True})

S0_optimum = optimal.x

print(S0_optimum)