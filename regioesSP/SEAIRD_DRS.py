# Import the necessary packages and modules
import sys
import csv
import math
import array
import operator
import argparse
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
from scipy.optimize import fsolve
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def logGrowth(growth,finalDay):
    x =[]
    y = []
    x=np.linspace(0,finalDay,finalDay+1)
    for i in range(0,len(x)):
        if i==0:
            y.append(100)
        else:
            y.append(y[i-1]*growth)
    return x,y

def predictionsPlot(df,startCases):
    cases=df.infected[df.infected > startCases]    
    time=np.linspace(0,len(cases)-1,len(cases))   
    return time,cases

def savePlot(strFile):
    if os.path.isfile(strFile):
        os.remove(strFile)   # Opt.: os.system("del "+strFile)
    plt.savefig(strFile,dpi=600)

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/max(1e-12,a)))

def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))

def getCases(df,state):
    cases=[]
    time=[]
    for i in range(len(df[state].values)):
        if df[state].values[i]>=100:
                cases.append(df[state].values[i])
    time=np.linspace(0,len(cases)-1,len(cases))  
    return time,cases

def loadDataFrame(filename):
    df= pd.read_pickle(filename)
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    df.columns = [c.lower().replace('(', '') for c in df.columns]
    df.columns = [c.lower().replace(')', '') for c in df.columns]
    return df

def parse_arguments(state):
    parser = argparse.ArgumentParser()

    state1="DRS 01 - Grande São Paulo"
    date="2020-03-15"
    s0=280.0e3
    e0=1e-4
    i0=1
    r0=0
    k0=0   
    a0=0

    parser.add_argument(
        '--states',
        dest='states',
        type=str,
        default=state1)
    
    parser.add_argument(
        '--download-data',
        dest='download_data',
        default=False
    )

    parser.add_argument(
        '--start-date',
        dest='start_date',
        type=str,
        default=date)
    
    parser.add_argument(
        '--prediction-days',
        dest='predict_range',
        type=int,
        default=150)

    parser.add_argument(
        '--S_0',
        dest='s_0',
        type=int,
        default=s0)

    parser.add_argument(
        '--E_0',
        dest='e_0',
        type=int,
        default=e0)

    parser.add_argument(
        '--A_0',
        dest='a_0',
        type=int,
        default=a0)

    parser.add_argument(
        '--I_0',
        dest='i_0',
        type=int,
        default=i0)

    parser.add_argument(
        '--R_0',
        dest='r_0',
        type=int,
        default=r0)

    parser.add_argument(
        '--D_0',
        dest='d_0',
        type=int,
        default=k0)

    args = parser.parse_args()

    state_list = []
    if args.states != "":
        try:
            states_raw = args.states
            state_list = states_raw.split(",")
        except Exception:
            sys.exit("QUIT: states parameter is not on CSV format")
    else:
        sys.exit("QUIT: You must pass a state list on CSV format.")

    return (state_list, args.download_data, args.start_date, args.predict_range, args.s_0, args.e_0, args.a_0, args.i_0, args.r_0, args.d_0)

def download_data(url_dictionary):
    #Lets download the files
    for url_title in url_dictionary.keys():
        urllib.request.urlretrieve(url_dictionary[url_title], "./data/" + url_title)

def load_json(json_file_str):
    # Loads  JSON into a dictionary or quits the program if it cannot.
    try:
        with open(json_file_str, "r") as json_file:
            json_variable = json.load(json_file)
            return json_variable
    except Exception:
        sys.exit("Cannot open JSON file: " + json_file_str)


class Learner(object):
    def __init__(self, state, loss, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0):
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

    def load_confirmed(self, state):
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv('./data/DRS_confirmados.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
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
        df = pd.read_csv('./data/DRS_mortes.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
        y=[]
        x=[]
        for i in range(0,len(df.date)):
            y.append(df[state].values[i])
            x.append(df.date.values[i])
        df2=pd.DataFrame(data=y,index=x,columns=[""])
        df2=df2[self.start_date:]
        return df2
    
    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%Y-%m-%d')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%Y-%m-%d'))
        return values

    #predict final extended values
    def predict(self, beta, sigma, sigma2, gamma, b, data, death, state, s_0, e_0, a_0, i_0, r_0, d_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SEAIRD(y,t):
        # def SEAIRD(t,y):
            S = y[0]
            E = y[1]
            A = y[2]
            I = y[3]
            R = y[4]
            D = y[5]
            p=0.2
            # sigma=1./22.
            # sigma2=1./55.
            y0=-beta*(A+I)*S #S
            y1=beta*S*(A+I)-sigma*E #E
            y2=sigma*E*(1-p)-gamma*A #A
            y3=sigma*E*p-gamma*I-sigma2*I#I
            y4=b*I+gamma*A+b/gamma*sigma2*I #R
            y5=max(0,1.-(y0+y1+y2+y3+y4)) #D
            return [y0,y1,y2,y3,y4,y5]

        y0=[s_0,e_0,a_0,i_0,r_0,d_0]
        tspan=np.arange(0, size, 1)
        res=odeint(SEAIRD,y0,tspan)
        # solution = solve_ivp(SEAIRD, [0, size], [s_0,e_0,a_0,i_0,r_0,d_0], t_eval=np.arange(0, size, 1), vectorized=True)
        extended_actual = data.values
        extended_death = death.values

        # x=[None]*(size - len(data.values))
        # extended_actual = np.concatenate((data.values, x))
        # extended_death = np.concatenate((death.values, x))

        # return new_index, extended_actual, extended_death, solution.y[0],solution.y[1],solution.y[2],solution.y[3],solution.y[4],solution.y[5], a, b
        return new_index, extended_actual, extended_death, res[:,0],res[:,1],res[:,2],res[:,3],res[:,4],res[:,5], a, b


    #run optimizer and plotting
    def train(self):
        self.data = self.load_confirmed(self.state)
        self.death = self.load_dead(self.state)

        optimal = minimize(lossOdeint,        
            [0.001, 0.001, 0.001, 0.001, 0.001],
            args=(self.data, self.death, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0),
            method='L-BFGS-B',
            bounds=[(1e-12, 5), (1./80.,0.2),  (1./100.,0.2), (1e-12, 0.6), (1e-12, 0.6)])
            #beta, sigma, sigma2, gamma, b

        # sigma=1/22
        # sigma2=1/55

        print(optimal)
        beta, sigma, sigma2, gamma, b = optimal.x
        new_index, extended_actual, extended_death, y0, y1, y2, y3, y4, y5, \
                a, b = self.predict(beta, sigma, sigma2, gamma, b, self.data, \
                self.death, self.state, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0)

        dataFr = [y0, y1, y2, y3, y4, y5]
        dataFr2 = np.array(dataFr).T
        df = pd.DataFrame(data=dataFr2)
        df.columns  = ['Susceptible','Exposed','Asymptomatic','Infected','Recovered','Deaths']
        df.index = pd.date_range(start=datetime.strptime(new_index[0],'%Y-%m-%d'), 
            end=datetime.strptime(new_index[len(new_index)-1],'%Y-%m-%d'))

        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("SEAIR-D Model for "+self.state)
        ax.plot(new_index[range(0,150)],y0,'g-',label="Susceptible")
        ax.plot(new_index[range(0,150)],y1,'r-',label="Exposed")
        ax.plot(new_index[range(0,150)],y2,'b-',label="Asymptomatic")
        plt.xticks(np.arange(0, 150, 20))
        ax.plot(new_index[range(0,150)],y3,'y-',label="Infected")
        # ax.plot(new_index[range(0,150)],y4,'c-',label="Recovered")
        ax.plot(new_index[range(0,150)],y5,'m-',label="Deaths")
        ax.plot(new_index[range(0,len(extended_actual))],extended_actual,'o',label="Infected data")
        ax.plot(new_index[range(0,len(extended_death))],extended_death,'x',label="Death data")
        ax.legend()
        print(f"state={self.state}, beta={beta:.8f}, 1/sigma={1/sigma:.8f}, 1/sigma2={1/sigma2:.8f},gamma={gamma:.8f}, b={b:.8f}, r_0:{(beta/gamma):.8f}")
        
        #plot margin annotation
        plt.annotate('Dr. Guilherme A. L. da Silva, www.ats4i.com', fontsize=10, 
        xy=(1.04, 0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='right',rotation=90)
        plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', fontsize=10, 
        xy=(1.045,0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='left',rotation=90)

        #save simulation results for comparison and use in another codes/routines
        df.to_pickle('./data/SEAIRD_sigmaOpt_'+self.state+'.pkl')
        state=self.state
        strFile ="./results/modelSEAIRDOpt"+state+".png"
        savePlot(strFile)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("Zoom SEAIR-D Model for "+self.state)
        plt.xticks(np.arange(0, 150, 20))
        ax.set_ylim(0,max(y3)+5e3)
        ax.plot(new_index[range(0,150)],y3,'y-',label="Infected")
        # ax.plot(new_index[range(0,150)],y4,'c-',label="Recovered")
        ax.plot(new_index[range(0,150)],y5,'m-',label="Deaths")
        ax.plot(new_index[range(0,len(extended_actual))],extended_actual,'o',label="Infected data")
        ax.plot(new_index[range(0,len(extended_death))],extended_death,'x',label="Death data")
        ax.legend()
       
        plt.annotate('Dr. Guilherme A. L. da Silva, www.ats4i.com', fontsize=10, 
        xy=(1.04, 0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='right',rotation=90)
        plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', fontsize=10, 
        xy=(1.045,0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='left',rotation=90)

        state=self.state
        strFile ="./results/ZoomModelSEAIRDOpt"+state+".png"
        savePlot(strFile)

        plt.show()
        plt.close()

#objective function Odeint solver
def lossOdeint(point, data, death, s_0, e_0, a_0, i_0, r_0, d_0):
    size = len(data)
    beta, sigma, sigma2, gamma, b = point
    def SEAIRD(y,t):
    # def SEAIRD(t,y):
        S = y[0]
        E = y[1]
        A = y[2]
        I = y[3]
        R = y[4]
        D = y[5]
        p=0.2
        # sigma=1./22.
        # sigma2=1./55.
        y0=-beta*(A+I)*S #S
        y1=beta*S*(A+I)-sigma*E #E
        y2=sigma*E*(1-p)-gamma*A #A
        y3=sigma*E*p-gamma*I-sigma2*I #I
        y4=b*I+gamma*A+b/gamma*sigma2*I #R
        y5=max(0,1.-(y0+y1+y2+y3+y4)) #D
        return [y0,y1,y2,y3,y4,y5]

    y0=[s_0,e_0,a_0,i_0,r_0,d_0]
    tspan=np.arange(0, size, 1)
    res=odeint(SEAIRD,y0,tspan)

    l1=0
    l2=0
    l3=0
    tot=0

    for i in range(0,len(data.values)):
        if data.values[i]>30:
            l1 = l1+(res[i,3] - data.values[i])**2
            l2 = l2+(res[i,5] - death.values[i])**2
            l3 = l3+(res[i,4] - death.values[i]*1.5)**2
            tot+=1
    l1=np.sqrt(l1/tot)
    l2=np.sqrt(l2/tot)
    l3=np.sqrt(l3/tot)
    # solution = solve_ivp(SEAIRD, [0, size], [s_0,e_0,a_0,i_0,r_0,d_0], t_eval=np.arange(0, size, 1), vectorized=True)
    # l1 = np.sqrt(np.mean((solution.y[3] - data.values)**2))
    # l2 = np.sqrt(np.mean((solution.y[5] - death.values)**2))
    #weight for cases
    u = 0.4  #Brazil US 0.1
    w = 0.0
    #weight for deaths
    v = max(0,1. - u - w)
    return u*l1 + v*l2 + w*l3

#main program SIRD model

def main(state):
    
    START_DATE = {
  'Japan': '1/22/20',
  'Italy': '1/31/20',
  'Republic of Korea': '1/22/20',
  'Iran (Islamic Republic of)': '2/19/20'}

    states, download, startdate, predict_range, s_0, e_0, a_0, i_0, r_0, d_0 = parse_arguments(state)

    # if download:
    #     data_d = load_json("./data_url.json")
    #     download_data(data_d) 

    for state in states:
        #learner = Learner(state, loss, startdate, predict_range, s_0, i_0, r_0, d_0)
        learner = Learner(state, lossOdeint, startdate, predict_range, s_0, e_0, a_0, i_0, r_0, d_0)
        #try:
        learner.train()
        #except BaseException:
        #    print('WARNING: Problem processing ' + str(state) +
        #        '. Be sure it exists in the data exactly as you entry it.' +
        #        ' Also check date format if you passed it as parameter.')

#Initial parameters
#Choose here your options

#option
#opt=0 all plots
#opt=1 corona log plot
#opt=2 logistic model prediction
#opt=3 bar plot with growth rate
#opt=4 log plot + bar plot
#opt=5 SEIR-D Model
opt=0

#load new confirmed cases
# data_d = load_json("./data_url.json")
# download_data(data_d)

#plot version - changes the file name png
version="1"

#choose state for curve fitting
#choose state for growth curve
#one of states above
#state="DRS 01 - Grande São Paulo"
state = sys.argv[2]

#choose state for SEIRD model
# "Brazil"
# "China"
# "Italy"
# "France"
# "United Kingdom"
# "US"
# states above are already adjusted
#stateSEAIRD="DRS 01 - Grande São Paulo"
stateSEAIRD = sys.argv[2]

# For other states you can run at command line
# but be sure to define S_0, I_0, R_0, d_0
# the sucess of fitting will depend on these paramenters
#
# usage: dataAndModelsCovid19.py [-h] [--states state_CSV] [--download-data]
#                  [--start-date START_DATE] [--prediction-days PREDICT_RANGE]
#                  [--S_0 S_0] [--I_0 I_0] [--R_0 R_0]

# optional arguments:
#   -h, --help            show this help message and exit
#   --states state_CSV
#                         states on CSV format. It must exact match the data
#                         names or you will get out of bonds error.
#   --download-data       Download fresh data and then run
#   --start-date START_DATE
#                         Start date on MM/DD/YY format ... I know ...It
#                         defaults to first data available 1/22/20
#   --prediction-days PREDICT_RANGE
#                         Days to predict with the model. Defaults to 150
#   --S_0 S_0             S_0. Defaults to 100000
#   --I_0 I_0             I_0. Defaults to 2
#   --R_0 R_0             R_0. Defaults to 0

#initial vars
a = 0.0
b = 0.0
c = 0.0 
date = []

#load CSV file
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
df = pd.read_csv('./data/DRS_confirmados.csv',delimiter=',',parse_dates=True, date_parser=dateparse)

#prepare data for plotting
state1="DRS 01 - Grande São Paulo"
[time1,cases1]=getCases(df,state1)
state2="DRS 07 - Campinas"
[time2,cases2]=getCases(df,state2)
state3="DRS 04 - Baixada Santista"
[time3,cases3]=getCases(df,state3)
state4="DRS 13 - Ribeirão Preto"
[time4,cases4]=getCases(df,state4)
state5="DRS 03 - Araraquara"
[time5,cases5]=getCases(df,state5)

if opt==1 or opt==0 or opt==4:

    model='SEAIRD_sigmaOpt'

    df = loadDataFrame('./data/SEAIRD_sigmaOpt_'+state+'.pkl')
    # for col in df.columns: 
        # print(col) 
    time6, cases6 = predictionsPlot(df,180)
    time6 = time6[0:60]
    cases6 = cases6[0:60]

    #model
    growth = 1.2
    x,y = logGrowth(growth,40)
    growth1 = 1.1
    x1,y1 = logGrowth(growth1,40)

    # Plot the data
    plt.rcParams['figure.figsize'] = [9, 7]
    plt.rc('font', size=14)
    plt.plot(time2, cases2,'r+-',label=state2) 
    plt.plot(time4, cases4,'mv-',label=state4) 
    plt.plot(time5, cases5,'cx-',label=state5) 
    plt.plot(time3, cases3,'go-',label=state3) 
    plt.plot(time6, cases6,'--',c='0.6',label=state3+" "+model) 
    plt.plot(time1, cases1,'b-',label=state1) 
    plt.plot(x, y,'y--',label='{:.1f}'.format((growth-1)*100)+'% per day',alpha=0.3)
    plt.plot(x1, y1,'y-.',label='{:.1f}'.format((growth1-1)*100)+'% per day',alpha=0.3) 
    plt.rc('font', size=11)

    plt.annotate(state3+" {:.1f} K".format(cases3[len(cases3)-1]/1000), # this is the text
        (time3[len(cases3)-1],cases3[len(cases3)-1]), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,10), # distance from text to points (x,y)
        ha='left') # horizontal alignment can be left, right or center

    idx=int(np.argmax(cases6))
    plt.annotate("{:.1f} K".format(max(cases6)/1000), # this is the text
        (time6[idx],cases6[idx]), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(5,-15), # distance from text to points (x,y)
        ha='right') # horizontal alignment can be left, right or center

    plt.annotate(state2+" {:.1f} K".format(cases2[len(cases2)-1]/1000), # this is the text
        (time2[len(cases2)-1],cases2[len(cases2)-1]), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,10), # distance from text to points (x,y)
        ha='center') # horizontal alignment can be left, right or center
    
    plt.annotate(state1+" {:.1f} K".format(cases1[len(cases1)-1]/1000), # this is the text
        (time1[len(cases1)-1],cases1[len(cases1)-1]), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,10), # distance from text to points (x,y)
        ha='center') # horizontal alignment can be left, right or center

    style = dict(size=10, color='gray')

    plt.annotate('Dr. Guilherme A. L. da Silva, www.ats4i.com', fontsize=10, 
            xy=(1.05, 0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90)
    # plt.annotate('Source: https://github.com/CSSEGISandData/COVID-19.git', fontsize=10, 
    #         xy=(1.06,0.1), xycoords='axes fraction',
    #         xytext=(0, 0), textcoords='offset points',
    #         ha='left',rotation=90)

  
    plt.xlabel('Days after 100th case')
    plt.ylabel('Official registered cases')
    plt.yscale('log')
    plt.title("Corona virus growth")
    plt.legend()

    #save figs
    savePlot('./results/coronaPythonEN_'+version+'.png')

    # Show the plot
    plt.show() 
    plt.close()

if opt==2 or opt==0:

    if opt==2:
        #model
        #33% per day
        x =[]
        y = []
        x=np.linspace(0,30,31)
        for i in range(0,len(x)):
            if i==0:
                y.append(100)
            else:
                y.append(y[i-1]*1.33)

        #50% per day
        x1 =[]
        y1 = []
        x1=np.linspace(0,30,31)
        for i in range(0,len(x1)):
            if i==0:
                y1.append(100)
            else:
                y1.append(y1[i-1]*1.25)

    #model fitting

    if state==state1:
        casesFit=cases1
        timeFit=time1
        maxCases=27e4
        maxTime=80
        guessExp=2

    if state==state2:
        casesFit=cases2
        timeFit=time2
        maxCases=13e4
        maxTime=80
        guessExp=2

    if state==state3:
        casesFit=cases3
        timeFit=time3
        maxCases=10e3
        maxTime=50
        guessExp=0.5

    if state==state4:
        casesFit=cases4
        timeFit=time4
        maxCases=12e4
        maxTime=80
        guessExp=2

    #logistic curve
    fit = curve_fit(logistic_model,timeFit,casesFit,p0=[5,60,8000])
    print ("Infection speed=",fit[0][0])
    print ("Day with the maximum infections occurred=",int(fit[0][1]))
    print ("Total number of recorded infected people at the infection’s end=",int(fit[0][2]))

    errors = [np.sqrt(fit[1][i][i]) for i in [0,1]]
    print("Errors = ",errors)
 
    #exponential curve
    exp_fit = curve_fit(exponential_model,timeFit,casesFit,p0=[guessExp,guessExp,guessExp])

    #plot
    pred_x = np.linspace(0,maxTime,maxTime+1)
    plt.rcParams['figure.figsize'] = [7, 7]
    plt.rc('font', size=14)

    # Real data
    plt.scatter(timeFit,casesFit,label="Real cases "+state,color="red")
    # Predicted logistic curve
    plt.plot(pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in pred_x], label="Logistic model" )
    # Predicted exponential curve
    plt.plot(pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in pred_x], label="Exponential model" )
    plt.legend()
    plt.xlabel("Days since 100th case")
    plt.ylabel("Total number of infected people in "+state)
    plt.ylim((min(y)*0.9,maxCases))

    plt.annotate('Dr. Guilherme A. L. da Silva, www.ats4i.com', fontsize=10, 
            xy=(1.05, -0.12), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90)
    plt.annotate('Source: https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d', fontsize=8, 
            xy=(1.06,-0.12), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='left',rotation=90)

    #save figs
    strFile ='./results/coronaPythonModelEN'+state+'.png'
    savePlot(strFile)

    plt.show() 
    plt.close()

if opt==3 or opt==0 or opt==4:

    plt.rcParams['figure.figsize'] = [9, 7]
    plt.rc('font', size=14)
    
    if state==state1:
        casesGrowth=cases1
        timeGrowth=time1
        maxCases=27e4
        maxTime=80
        guessExp=2

    if state==state2:
        casesGrowth=cases2
        timeGrowth=time2
        maxCases=13e4
        maxTime=80
        guessExp=2

    if state==state3:
        casesGrowth=cases3
        timeGrowth=time3
        maxCases=30e3
        maxTime=50
        guessExp=0.5

    if state==state4:
        casesGrowth=cases4
        timeGrowth=time4
        maxCases=12e4
        maxTime=80
        guessExp=2

    #growth rate
    growth=[]
    for i in range(0,len(casesGrowth)-1):
        growth.append(100*float(casesGrowth[i+1])/float(casesGrowth[i])-100)

    #Setup dummy data
    N = 10
    ind = timeGrowth[1:]
    bars = growth
    colors = cm.rainbow(np.asfarray(growth,float) / float(max(np.asfarray(growth,float))))
    plot = plt.scatter(growth, growth, c = growth, cmap = 'rainbow')
    plt.clf()
    plt.colorbar(plot)

    #Plot bars
    plt.bar(ind, bars, color=colors)
    plt.xlabel('Days since 100th case')

    # Make the y-axis label and tick labels match the line color.
    plt.ylabel(state+' growth official cases per day [%]') 

    #Plot a line
    plt.axhline(y=33,color='r',linestyle='--')

    plt.annotate("doubling each 3 days", # this is the text
        (13,33), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,5), # distance from text to points (x,y)
        ha='center') # horizontal alignment can be left, right or center

    # Text on the top of each barplot
    for i in range(1,len(ind)):
        plt.text(x = ind[i]-0.5 , y = growth[i]+0.5, s = " {:.1f}%".format(growth[i]), size = 7)

    plt.annotate('Dr. Guilherme A. L. da Silva, www.ats4i.com', fontsize=10, 
            xy=(1.24, 0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90)
    # plt.annotate('Source: https://github.com/CSSEGISandData/COVID-19.git', fontsize=10, 
    #         xy=(1.25,0.1), xycoords='axes fraction',
    #         xytext=(0, 0), textcoords='offset points',
    #         ha='left',rotation=90)

    #save figs
    strFile ='./results/coronaPythonGrowthEN_'+state+'.png'
    savePlot(strFile)

    plt.show() 
    plt.close()

    #growth rate
    growth=[]
    for i in range(0,len(casesGrowth)-1):
        growth.append(float(casesGrowth[i+1])-float(casesGrowth[i]))


    #Setup dummy data
    N = 10
    ind = timeGrowth[1:]
    bars = growth

    colors = cm.rainbow(np.asfarray(growth,float) / float(max(np.asfarray(growth,float))))
    plot = plt.scatter(growth, growth, c = growth, cmap = 'rainbow')
    plt.clf()
    plt.colorbar(plot)

    #Plot bars
    plt.bar(ind, bars, color=colors)
    plt.xlabel('Days since 100th case')

    # Make the y-axis label and tick labels match the line color.
    plt.ylabel(state+' growth official cases per day [number]') 

    # Plot a line
    plt.axhline(y=300,color='r',linestyle='--')

    plt.annotate("Expected per day", # this is the text
        (5,310), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,5), # distance from text to points (x,y)
        ha='center') # horizontal alignment can be left, right or center

    # Text on the top of each barplot
    for i in range(1,len(ind)):
        plt.text(x = ind[i]-0.5 , y = growth[i]+5, s = " {:.0f}".format(growth[i]), size = 7)

    plt.annotate('Dr. Guilherme A. L. da Silva, www.ats4i.com', fontsize=10, 
            xy=(1.24, 0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90)
    # plt.annotate('Source: https://github.com/CSSEGISandData/COVID-19.git', fontsize=10, 
    #         xy=(1.25,0.1), xycoords='axes fraction',
    #         xytext=(0, 0), textcoords='offset points',
    #         ha='left',rotation=90)

    #save figs
    strFile ='./results/coronaPythonGrowthDeltaCasesEN_'+state+'.png'
    savePlot(strFile)

    plt.show() 
    plt.close()

if opt==5 or opt==0:

    #https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html

    if __name__ == '__main__':
        main(stateSEAIRD)
