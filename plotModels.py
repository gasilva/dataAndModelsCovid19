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
        os.remove(strFile)   # Opt.: os.system("del "+strFile)
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

# print(df_SIRD)

# print(df_SEIR.index)
# print(df_SEIRD)
# print(df_SEIRDopt)
# print(df_SEAIRD)
# print(df_SEAIRDopt)


# sys.exit()

# susceptible
# exposed
# asymptomatic
# infected_data
# infected
# recovered_alive
# predicted_recovered_alive
# death_data
# predicted_deaths

# SIRD SEIR
# susceptible
# exposed
# infected_data
# infected
# recovered_data
# recovered
# death_data
# predicted_deaths

# for col in df_SEIRD.columns: 
#     print(col) 

# sys.exit()

df = pd.DataFrame({
            'Infected data': df_SEIRD.infected_data,
            'SIRD': df_SIRD.infected,
            'SEIR': df_SEIR.infected,
            'SEIRD': df_SEIRD.infected,
            'SEIRD Opt': df_SEIRDopt.infected,
            'SEAIRD Opt': df_SEAIRDopt.infected},
            index=df_SEIRD.index
            )

plt.rc('font', size=14)
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title("Compare Models Infected for "+country)
# ax.set_ylim((0, max(y0+5e3)))
df.plot(ax=ax)

plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
xy=(1.04, 0.1), xycoords='axes fraction',
xytext=(0, 0), textcoords='offset points',
ha='right',rotation=90)

savePlot("./results/compareModelInfected"+country+".png")

plt.show() 
plt.close()

df = pd.DataFrame({
            'Death data': df_SEIRD.death_data,
            'SIRD': df_SIRD.infected,
            'SEIR': df_SEIR.infected,
            'SEIRD': df_SEIRD.predicted_deaths,
            'SEIRD Opt': df_SEIRDopt.predicted_deaths,
            'SEAIRD Opt': df_SEAIRDopt.predicted_deaths},
            index=df_SEIRD.index
            )

plt.rc('font', size=14)
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title("Compare Models Deaths for "+country)
df.plot(ax=ax)

plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
xy=(1.04, 0.1), xycoords='axes fraction',
xytext=(0, 0), textcoords='offset points',
ha='right',rotation=90)

savePlot("./results/compareModelDeaths"+country+".png")

plt.show() 
plt.close()

df = pd.DataFrame({
            'Recovered data': df_SEAIRDopt.recovered_alive,
            'SIRD': df_SIRD.recovered,
            'SEIR': df_SEIR.recovered,
            'SEIRD': df_SEIRD.predicted_recovered_alive,
            'SEIRD Opt': df_SEIRDopt.predicted_recovered_alive,
            'SEAIRD Opt': df_SEAIRDopt.predicted_recovered_alive},
            index=df_SEIRD.index
            )

plt.rc('font', size=14)
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_title("Compare Models Recovered for "+country)
df.plot(ax=ax)

plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
xy=(1.04, 0.1), xycoords='axes fraction',
xytext=(0, 0), textcoords='offset points',
ha='right',rotation=90)

savePlot("./results/compareModelRecovered"+country+".png")

plt.show() 
plt.close()
