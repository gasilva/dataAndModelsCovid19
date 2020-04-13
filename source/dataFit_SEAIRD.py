

# Import the necessary packages and modules
from datetime import datetime
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
import urllib.request
from csv import reader
from csv import writer
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/max(1e-12,a)))

def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))

def getCases(df,country):
    cases1=[]
    time1=[]
    tamLi = np.shape(df)[0]
    tamCol = np.shape(df)[1]
    jx=0
    for i in range(0,tamCol):
        tamCountry=0
        j1=0
        if df[i][1]==country and not df[i][0]=="ignore":
            for j in range(4,tamLi):
                if float(df[i][j])>=100.0 and j1>=jx:
                    cases1.append(float(df[i][j]))
                    time1.append(float(j1))
                    jx=j1
                    j1+=1
    return [time1, cases1]

def parse_arguments(country):
    parser = argparse.ArgumentParser()

    country1=country

    if country1=="Brazil":
        date="3/3/20"
        s0=350e3
        e0=1e-4
        i0=100
        r0=200
        k0=300

    if country1=="China":
        date="1/22/20"
        s0=210e3
        e0=1e-4
        i0=800
        r0=0 #-250e3
        k0=0

    if country1=="Italy":
        date="2/14/20"
        s0=250e3
        e0=1e-4
        i0=50
        r0=0
        k0=0

    if country1=="France":
        date="2/25/20"
        s0=190e3
        e0=1e-4
        i0=265
        r0=0
        k0=0

    if country1=="United Kingdom":
        date="2/25/20"
        s0=138000
        e0=1e-4
        i0=22
        r0=0 #-50
        k0=150

    if country1=="US":
        date="2/25/20"
        s0=900000
        e0=1e-4
        i0=00
        r0=0
        k0=600
    
    a0=0

    parser.add_argument(
        '--countries',
        dest='countries',
        type=str,
        default=country1)
    
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

    country_list = []
    if args.countries != "":
        try:
            countries_raw = args.countries
            country_list = countries_raw.split(",")
        except Exception:
            sys.exit("QUIT: countries parameter is not on CSV format")
    else:
        sys.exit("QUIT: You must pass a country list on CSV format.")

    return (country_list, args.download_data, args.start_date, args.predict_range, args.s_0, args.e_0, args.a_0, args.i_0, args.r_0, args.d_0)

def sumCases_province(input_file, output_file):
    with open(input_file, "r") as read_obj, open(output_file,'w',newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
               
        lines=[]
        for line in csv_reader:
            lines.append(line)    

        i=0
        ix=0
        for i in range(0,len(lines[:])-1):
            if lines[i][1]==lines[i+1][1]:
                if ix==0:
                    ix=i
                lines[ix][4:] = np.asfarray(lines[ix][4:],float)+np.asfarray(lines[i+1][4:] ,float)
            else:
                if not ix==0:
                    lines[ix][0]=""
                    csv_writer.writerow(lines[ix])
                    ix=0
                else:
                    csv_writer.writerow(lines[i])
            i+=1     

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
    def __init__(self, country, loss, start_date, predict_range,s_0, e_0, a_0, i_0, r_0, d_0):
        self.country = country
        self.loss = loss
        self.start_date = start_date
        self.predict_range = predict_range
        self.s_0 = s_0
        self.e_0 = e_0
        self.i_0 = i_0
        self.r_0 = r_0
        self.d_0 = d_0
        self.a_0 = a_0

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

    #predict final extended values
    def predict(self, beta, sigma, sigma2, gamma, b, data, recovered, death, healed, country, s_0, e_0, a_0, i_0, r_0, d_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SEAIRD(y,t):
            S = y[0]
            E = y[1]
            A = y[2]
            I = y[3]
            R = y[4]
            D = y[5]
            sigma=1/22.0
            sigma2=1/55.0
            p=0.15
            y0=-beta*(A+I)*S #S
            y1=beta*S*(A+I)-(sigma)*E #E
            y2=sigma*E*(1-p)-gamma*A #A
            y3=sigma*E*p-gamma*I-sigma2*I #I
            y4=b*I+gamma*A+b/gamma*sigma2*I #R
            y5=max(0,1.-(y0+y1+y2+y3+y4)) #D
            return [y0,y1,y2,y3,y4,y5]
        y0=[s_0,e_0,a_0,i_0,r_0,d_0]
        tspan=np.arange(0, size, 1)
        res=odeint(SEAIRD,y0,tspan)

        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        extended_healed = np.concatenate((healed.values, [None] * (size - len(healed.values))))
        return new_index, extended_actual, extended_recovered, extended_death, res[:,0], res[:,1],res[:,2],res[:,3],res[:,4], res[:,5], extended_healed, a, b

    #run optimizer and plotting
    def train(self):
        self.death = self.load_dead(self.country)
        self.healed = self.load_recovered(self.country)
        self.recovered = self.healed + self.death
        self.data = self.load_confirmed(self.country) - self.recovered

        optimal = minimize(lossOdeint,        
            [0.001, 0.001, 0.001, 0.001, 0.001],
            args=(self.data, self.recovered, self.death, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0),
            method='L-BFGS-B',
            bounds=[(1e-12, 5), (1e-12,0.2),  (1e-12,0.2), (1e-12, 0.6), (1e-12, 0.6)])
            #beta, sigma, gamma

        sigma=1/22.0
        sigma2=1/55.0

        print(optimal)
        beta, xsigma, xsigma2, gamma, b = optimal.x
        new_index, extended_actual, extended_recovered, extended_death, y0, y1, y2, y3, y4, y5, \
                extended_healed, a, b = self.predict(beta, sigma, sigma2, gamma, b, self.data, self.recovered, \
                self.death, self.healed, self.country, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, self.d_0)

        df = pd.DataFrame({
                    'Susceptible': y0,
                    'Exposed': y1,
                    'Asymptomatic': y2,
                    'Infected data': extended_actual,
                    'Infected': y3,
                    'Recovered (Alive)': extended_healed,
                    'Predicted Recovered (Alive)': y4,
                    'Death data': extended_death,
                    'Predicted Deaths': y5},
                    index=new_index)

        #plt.rcParams['figure.figsize'] = [7, 7]
        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("SEAIR-D Model for "+self.country)
        ax.set_ylim((0, max(y0+5e3)))
        df.plot(ax=ax)
        print(f"country={self.country}, beta={beta:.8f}, 1/sigma={1/sigma:.8f}, 1/sigma2={1/sigma2:.8f},gamma={gamma:.8f}, b={b:.8f}, r_0:{(beta/gamma):.8f}")
        
        plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
        xy=(1.04, 0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='right',rotation=90)
        plt.annotate('Source: https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html', fontsize=10, 
        xy=(1.06,0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='left',rotation=90)
        plt.annotate('Original SEIR-D with delay model, São Paulo, Brazil', fontsize=10, 
        xy=(1.045,0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='left',rotation=90)

        df.to_pickle('./data/SEAIRD_'+self.country+'.pkl')

        country=self.country
        strFile ="./results/modelSEAIRD"+country+".png"
        savePlot(strFile)

        plt.show()
        plt.close()

#objective function Odeint solver
def lossOdeint(point, data, recovered, death, s_0, e_0, a_0, i_0, r_0, d_0):
    size = len(data)
    beta, sigma, sigma2, gamma, b = point
    def SEAIRD(y,t):
        S = y[0]
        E = y[1]
        A = y[2]
        I = y[3]
        R = y[4]
        D = y[5]
        sigma=1/22.0
        sigma2=1/55.0
        p=0.15
        y0=-beta*(A+I)*S #S
        y1=beta*S*(A+I)-(sigma)*E #E
        y2=sigma*E*(1-p)-gamma*A #A
        y3=sigma*E*p-gamma*I-sigma2*I #I
        y4=b*I+gamma*A+b/gamma*sigma2*I #R
        y5=max(0,1.-(y0+y1+y2+y3+y4)) #D
        return [y0,y1,y2,y3,y4,y5]
    y0=[s_0,e_0,a_0,i_0,r_0,d_0]
    tspan=np.arange(0, size, 1)
    res=odeint(SEAIRD,y0,tspan)
    l1 = np.sqrt(np.mean((res[:,3] - data)**2))
    l2 = np.sqrt(np.mean((res[:,4]- recovered)**2))
    l3 = np.sqrt(np.mean((res[:,5]- death)**2))
    #weight for cases
    u = 0.5  #Brazil Italy UK 0.5 France 0.4
    #weight for deaths
    w = 0.3 #Brazil Italy UK 0.3
    #weight for recovered
    v = max(0,1. - u - w)
    return u*l1 + v*l2 + w*l3

#main program SIRD model

def main(country):
    
    START_DATE = {
  'Japan': '1/22/20',
  'Italy': '1/31/20',
  'Republic of Korea': '1/22/20',
  'Iran (Islamic Republic of)': '2/19/20'}

    countries, download, startdate, predict_range , s_0, e_0, a_0, i_0, r_0, d_0 = parse_arguments(country)

    if download:
        data_d = load_json("./data_url.json")
        download_data(data_d)

    sumCases_province('data/time_series_19-covid-Confirmed.csv', 'data/time_series_19-covid-Confirmed-country.csv')
    sumCases_province('data/time_series_19-covid-Recovered.csv', 'data/time_series_19-covid-Recovered-country.csv')
    sumCases_province('data/time_series_19-covid-Deaths.csv', 'data/time_series_19-covid-Deaths-country.csv')

    for country in countries:
        #learner = Learner(country, loss, startdate, predict_range, s_0, i_0, r_0, d_0)
        learner = Learner(country, lossOdeint, startdate, predict_range, s_0, e_0, a_0, i_0, r_0, d_0)
        #try:
        learner.train()
        #except BaseException:
        #    print('WARNING: Problem processing ' + str(country) +
        #        '. Be sure it exists in the data exactly as you entry it.' +
        #        ' Also check date format if you passed it as parameter.')

def savePlot(strFile):
    if os.path.isfile(strFile):
        os.remove(strFile)   # Opt.: os.system("del "+strFile)
    plt.savefig(strFile,dpi=600)

#initial vars
a = 0.0
b = 0.0
c = 0.0 
date = []

#load new confirmed cases
data_d = load_json("./data_url.json")
download_data(data_d)

#sum provinces under same country
sumCases_province('data/time_series_19-covid-Confirmed.csv', 'data/time_series_19-covid-Confirmed-country.csv')

#load CSV file
dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
df=pd.read_csv('data/time_series_19-covid-Confirmed-country.csv', \
    delimiter=',',parse_dates=True, date_parser=dateparse,header=None)
df=df.transpose()

#Initial parameters
#Choose here your options

#option
#opt=0 all plots
#opt=1 corona log plot
#opt=2 logistic model prediction
#opt=3 bar plot with growth rate
#opt=4 log plot + bar plot
#opt=5 SEAIR-D Model
opt=5

#prepare data for plotting
country1="US"
[time1,cases1]=getCases(df,country1)
country2="Italy"
[time2,cases2]=getCases(df,country2)
country3="Brazil"
[time3,cases3]=getCases(df,country3)
country4="France"
[time4,cases4]=getCases(df,country4)
country5="Germany"
[time5,cases5]=getCases(df,country5)

#plot version - changes the file name png
version="1"

#choose country for curve fitting
#choose country for growth curve
#one of countries above
country="Brazil"

#choose country for SEIRD model
# "Brazil"
# "China"
# "Italy"
# "France"
# "United Kingdom"
# "US"
# Countries above are already adjusted
countrySIRD="Brazil"

# For other countries you can run at command line
# but be sure to define S_0, I_0, R_0, d_0
# the sucess of fitting will depend on these paramenters
#
# usage: dataAndModelsCovid19.py [-h] [--countries COUNTRY_CSV] [--download-data]
#                  [--start-date START_DATE] [--prediction-days PREDICT_RANGE]
#                  [--S_0 S_0] [--I_0 I_0] [--R_0 R_0]

# optional arguments:
#   -h, --help            show this help message and exit
#   --countries COUNTRY_CSV
#                         Countries on CSV format. It must exact match the data
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

if opt==1 or opt==0 or opt==4:

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

    # Plot the data
    #ax.figure(figsize=(19.20,10.80))
    plt.rcParams['figure.figsize'] = [9, 7]
    plt.rc('font', size=14)
    plt.plot(time2, cases2,'r+-',label=country2) 
    plt.plot(time4, cases4,'mv-',label=country4) 
    plt.plot(time5, cases5,'cx-',label=country5) 
    plt.plot(time3, cases3,'go-',label=country3) 
    plt.plot(time1, cases1,'b-',label=country1) 
    plt.plot(x, y,'y--',label='33% per day',alpha=0.3) 
    plt.plot(x1, y1,'y-.',label='25% per day',alpha=0.3) 
    plt.rc('font', size=11)
    plt.annotate(country3+" {:.1f} K".format(cases3[len(cases3)-1]/1000), # this is the text
        (time3[len(cases3)-1],cases3[len(cases3)-1]), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(15,10), # distance from text to points (x,y)
        ha='left') # horizontal alignment can be left, right or center
    plt.annotate(country2+" {:.1f} K".format(cases2[len(cases2)-1]/1000), # this is the text
        (time2[len(cases2)-1],cases2[len(cases2)-1]), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,10), # distance from text to points (x,y)
        ha='center') # horizontal alignment can be left, right or center
    plt.annotate(country1+" {:.1f} K".format(cases1[len(cases1)-1]/1000), # this is the text
        (time1[len(cases1)-1],cases1[len(cases1)-1]), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,10), # distance from text to points (x,y)
        ha='center') # horizontal alignment can be left, right or center

    style = dict(size=10, color='gray')

    plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
            xy=(1.05, 0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90)
    plt.annotate('Source: https://github.com/CSSEGISandData/COVID-19.git', fontsize=10, 
            xy=(1.06,0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='left',rotation=90)

  
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

    if country==country1:
        casesFit=cases1
        timeFit=time1
        maxCases=27e4
        maxTime=80
        guessExp=2

    if country==country2:
        casesFit=cases2
        timeFit=time2
        maxCases=13e4
        maxTime=80
        guessExp=2

    if country==country3:
        casesFit=cases3
        timeFit=time3
        maxCases=50e3
        maxTime=50
        guessExp=0.5

    if country==country4:
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
    pred_x = list(range(len(time3)+1,maxTime))
    plt.rcParams['figure.figsize'] = [7, 7]
    plt.rc('font', size=14)
    # Real data
    plt.scatter(timeFit,casesFit,label="Real cases "+country,color="red")
    # Predicted logistic curve
    plt.plot(time3+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in time3+pred_x], label="Logistic model" )
    # Predicted exponential curve
    plt.plot(time3+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in time3+pred_x], label="Exponential model" )
    plt.legend()
    plt.xlabel("Days since 100th case")
    plt.ylabel("Total number of infected people in "+country)
    plt.ylim((min(y)*0.9,maxCases))

    plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
            xy=(1.05, -0.12), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90)
    plt.annotate('Source: https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d', fontsize=8, 
            xy=(1.06,-0.12), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='left',rotation=90)

    #save figs
    strFile ='./results/coronaPythonModelEN'+country+'.png'
    savePlot(strFile)

    plt.show() 
    plt.close()

if opt==3 or opt==0 or opt==4:
    plt.rcParams['figure.figsize'] = [9, 7]
    plt.rc('font', size=14)
    
    growth=[]

    if country==country1:
        casesGrowth=cases1
        timeGrowth=time1
        maxCases=27e4
        maxTime=80
        guessExp=2

    if country==country2:
        casesGrowth=cases2
        timeGrowth=time2
        maxCases=13e4
        maxTime=80
        guessExp=2

    if country==country3:
        casesGrowth=cases3
        timeGrowth=time3
        maxCases=30e3
        maxTime=50
        guessExp=0.5

    if country==country4:
        casesGrowth=cases4
        timeGrowth=time4
        maxCases=12e4
        maxTime=80
        guessExp=2

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
    plt.ylabel(country+' growth official cases per day [%]') 

    #Plot a line
    plt.axhline(y=33,color='r',linestyle='--')

    plt.annotate("doubling each 3 days", # this is the text
        (13,33), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,5), # distance from text to points (x,y)
        ha='center') # horizontal alignment can be left, right or center

    # Text on the top of each barplot
    for i in range(1,len(ind)):
        plt.text(x = ind[i]-0.5 , y = growth[i]+0.25, s = " {:.1f}%".format(growth[i]), size = 7)

    plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
            xy=(1.24, 0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90)
    plt.annotate('Source: https://github.com/CSSEGISandData/COVID-19.git', fontsize=10, 
            xy=(1.25,0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='left',rotation=90)

    #save figs
    strFile ='./results/coronaPythonGrowthEN_'+country+'.png'
    savePlot(strFile)

    plt.show() 
    plt.close()

if opt==5 or opt==0:

    #https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html

    if __name__ == '__main__':
        main(countrySIRD)
