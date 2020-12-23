# Import the necessary packages and modules
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import csv
import math
import pandas as pd
import array
import operator
import argparse
import sys
import json
import ssl
import os
from datetime import datetime,timedelta

def savePlot(strFile):
    if os.path.isfile(strFile):
        os.remove(strFile) 
    plt.savefig(strFile,dpi=600)

def loadDataFrame(filename):
    df= pd.read_pickle(filename)
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    df.columns = [c.lower().replace('(', '') for c in df.columns]
    df.columns = [c.lower().replace(')', '') for c in df.columns]
    return df

country="Brazil"
df_SEIR = loadDataFrame('./data/SEIR_'+country+'.pkl')
df_SEIRD = loadDataFrame('./data/SEIRD_'+country+'.pkl')
df_SEIRDopt = loadDataFrame('./data/SEIRD_sigmaOpt_'+country+'.pkl')
df_SEAIRD = loadDataFrame('./data/SEAIRD_'+country+'.pkl')
df_SEAIRDopt = loadDataFrame('./data/SEAIRD_sigmaOpt_'+country+'.pkl')
df_SIRD = loadDataFrame('./data/SIRD_'+country+'.pkl')
df_SEAIRDoptGlobal = loadDataFrame('./data/SEAIRD_sigmaOpt_Global'+country+'.pkl')

# look at columns names of dataframe chosen
# print("SEAIRDoptGlobal")
# for col in df_SEAIRDoptGlobal.columns: 
#      print(col) 
# sys.exit()

plt.rc('font', size=14)
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title("Compare Models Infected for "+country)
plt.xticks(np.arange(0, 150, 15))
df_SEIRD.infected_data.plot(ax=ax,style="bo",label="data")
df_SIRD.infected.plot(ax=ax,label="SIRD")
df_SEIR.infected.plot(ax=ax,label="SEIR")
df_SEIRD.infected.plot(ax=ax,label="SEIRD")
df_SEIRDopt.infected.plot(ax=ax,label="SEIRDopt")
df_SEAIRD.infected.plot(ax=ax,label="SEAIRD")
df_SEAIRDopt.infected.plot(ax=ax,label="SEAIRDopt")
df_SEAIRDoptGlobal.infected.plot(ax=ax,label="SEAIRDoptGlobal")
plt.legend()

plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
xy=(1.04, 0.1), xycoords='axes fraction',
xytext=(0, 0), textcoords='offset points',
ha='right',rotation=90)

savePlot("./results/compareModelInfected"+country+".png")

plt.show() 
plt.close()

plt.rc('font', size=14)
fig, ax = plt.subplots(figsize=(15, 10))
plt.xticks(np.arange(0, 150, 15))
ax.set_title("Compare Models Deaths for "+country)
df_SEIRD.death_data.plot(ax=ax,style="bo",label="data")
df_SIRD.estimated_deaths.plot(ax=ax,label="SEIRD")
df_SEIR.recovered.plot(ax=ax,label="SEIR")
df_SEIRD.predicted_deaths.plot(ax=ax,label="SEIRD")
df_SEIRDopt.predicted_deaths.plot(ax=ax,label="SEIRDopt")
df_SEAIRD.infected.plot(ax=ax,label="SEAIRD")
df_SEAIRDopt.predicted_deaths.plot(ax=ax,label="SEAIRDopt")
df_SEAIRDoptGlobal.predicted_deaths.plot(ax=ax,label="SEAIRDoptGlobal")
plt.legend()

plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
xy=(1.04, 0.1), xycoords='axes fraction',
xytext=(0, 0), textcoords='offset points',
ha='right',rotation=90)

savePlot("./results/compareModelDeaths"+country+".png")

plt.show() 
plt.close()

plt.rc('font', size=14)
fig, ax = plt.subplots(figsize=(15, 10))
plt.xticks(np.arange(0, 150, 15))
ax.set_title("Compare Models Recovered for "+country)
df_SEAIRDoptGlobal.recovered.plot(ax=ax,style="bo",label="data")
df_SIRD.recovered.plot(ax=ax,label="SEIRD")
df_SEIR.recovered.plot(ax=ax,label="SEIR")
df_SEIRD.predicted_recovered_alive.plot(ax=ax,label="SEIRD")
df_SEIRDopt.predicted_recovered_alive.plot(ax=ax,label="SEIRDopt")
df_SEAIRD.predicted_recovered_alive.plot(ax=ax,label="SEAIRD")
df_SEAIRDopt.predicted_recovered_alive.plot(ax=ax,label="SEAIRDopt")
df_SEAIRDoptGlobal.predicted_recovered.plot(ax=ax,label="SEAIRDoptGlobal")
plt.legend()

plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
xy=(1.04, 0.1), xycoords='axes fraction',
xytext=(0, 0), textcoords='offset points',
ha='right',rotation=90)

savePlot("./results/compareModelRecovered"+country+".png")

plt.show() 
plt.close()
