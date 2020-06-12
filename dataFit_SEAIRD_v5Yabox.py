# Import the necessary packages and modules
import sys
import argparse
import json
import os
import urllib.request
import numpy as np
import pandas as pd
from csv import reader
from csv import writer
from datetime import datetime,timedelta
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.special import comb, binom
from yabox import DE
from tqdm import tqdm
import math
import sigmoid as sg

import matplotlib as mpl
# mpl.use('agg')
# mpl.use('TkAgg')
mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')

from matplotlib import cm
import matplotlib.font_manager as fm
# Font Imports
heading_font = fm.FontProperties(fname='/home/ats4i/playfair-display/PlayfairDisplay-Regular.ttf', size=22)
subtitle_font = fm.FontProperties(fname='/home/ats4i/Roboto/Roboto-Regular.ttf', size=12)

#marker and line sizes
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 7

#parallel computation
import ray
ray.shutdown()
ray.init(num_cpus=3)

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

def predictionsPlot(df,nPoints,startCases):
    cases=[]
    time=[]
    j1=0
    df.infected=df.infected+df.predicted_deaths+df.predicted_recovered
    for j in range(0,nPoints):
        if float(df.infected[j])>=startCases:
            cases.append(float(df.infected[j]))
            time.append(float(j1))
            j1+=1
    return time,cases

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
        j1=0
        if df[i][1]==country and not df[i][0]=="ignore":
            for j in range(4,tamLi):
                if float(df[i][j])>=100.0 and j1>=jx:
                    cases1.append(float(df[i][j]))
                    time1.append(float(j1))
                    jx=j1
                    j1+=1
    return [time1, cases1]

def loadDataFrame(filename):
    df= pd.read_pickle(filename)
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    df.columns = [c.lower().replace('(', '') for c in df.columns]
    df.columns = [c.lower().replace(')', '') for c in df.columns]
    return df

def parse_arguments():
    parser = argparse.ArgumentParser()

    date="2/25/20"
    s0=3e6*2
    e0=1e-4
    a0=1e-4
    i0=265
    r0=0
    k0=0
    #start fitting when the number of cases >= start
    start=0
    #how many days is the prediction
    prediction_days=150
    #weigth for fitting data
    weigthCases=0.65
    weigthRecov=0.1
    #weightDeaths = 1 - weigthCases - weigthRecov

    parser.add_argument(
        '--countries',
        dest='countries',
        type=str,
        default=country1)
    
    parser.add_argument(
        '--download-data',
        dest='download_data',
        default=True
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
        default=prediction_days)

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

    parser.add_argument(
        '--START',
        dest='startNCases',
        type=int,
        default=start)

    parser.add_argument(
        '--WCASES',
        dest='weigthCases',
        type=float,
        default=weigthCases)

    parser.add_argument(
        '--WREC',
        dest='weigthRecov',
        type=float,
        default=weigthRecov)

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

    return (country_list, args.download_data, args.start_date, args.predict_range, \
        args.s_0, args.e_0, args.a_0, args.i_0, args.r_0, args.d_0,\
            args.startNCases, args.weigthCases, args.weigthRecov)

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

#register function for parallel processing
# @ray.remote
class Learner(object):
    def __init__(self, country, start_date, predict_range,s_0, e_0, a_0,\
                 i_0, r_0, d_0, startNCases, weigthCases, weigthRecov, cleanRecovered):
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
        self.cleanRecovered=cleanRecovered

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
    def predict(self, beta0, beta01, startT, beta2, sigma, sigma2, sigma3, gamma, b, mu, gamma2, d, data, \
                recovered, death, country, s_0, e_0, a_0, i_0, r_0, d_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SEAIRD(y,t):
            rx=sg.sigmoid(t-startT)
            beta=beta0*rx+beta01*(1-rx)
            S = y[0]
            E = y[1]
            A = y[2]
            I = y[3]
            R = y[4]
            p=0.2
            # beta2=beta
            y0=-(beta2*A+beta*I)*S-mu*S #S
            y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
            y2=sigma*E*(1-p)-mu*A-gamma2*A #A
            y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I #I
            y4=b*I+d*A+sigma2*I-mu*R #R
            y5=(-(y0+y1+y2+y3+y4)) #D
            return [y0,y1,y2,y3,y4,y5]
        
        y0=[s_0,e_0,a_0,i_0,r_0,d_0]
        tspan=np.arange(0, size, 1)
        res=odeint(SEAIRD,y0,tspan) #,hmax=1)

        extended_actual = np.concatenate((data.values, \
                            [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, \
                            [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, \
                            [None] * (size - len(death.values))))
        return new_index, extended_actual, extended_recovered, extended_death, \
             res[:,0], res[:,1],res[:,2],res[:,3],res[:,4], res[:,5]

    #run optimizer
    def train(self):
        
        self.death = self.load_dead(self.country)
        self.recovered = self.load_recovered(self.country)
        if self.cleanRecovered:
            zeroRecDeaths=0
        else:
            zeroRecDeaths=1
        self.data = self.load_confirmed(self.country)-zeroRecDeaths*(self.recovered+self.death)

        size=len(self.data)

        bounds=[(1e-12, .2),(1e-12, .2),(5,size-5),(1e-12, .2),(1/120 ,0.4),(1/120, .4),
        (1/120, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4)]

        maxiterations=2500
        f=create_lossOdeint(self.data, self.recovered, \
            self.death, self.s_0, self.e_0, self.a_0, self.i_0, self.r_0, \
                self.d_0, self.startNCases, \
                self.weigthCases, self.weigthRecov)
        de = DE(f, bounds, maxiters=maxiterations)
        i=0
        ndims=len(bounds)
        with tqdm(total=maxiterations*ndims*125) as pbar:
            for step in de.geniterator():
                idx = step.best_idx
                norm_vector = step.population[idx]
                best_params = de.denormalize([norm_vector])
                pbar.update(i)
                i+=1
        p=best_params[0]

        #parameter list for optimization
        #beta0, beta01, startT, beta2, sigma, sigma2, sigma3, gamma, b, mu, gamma2, d

        beta0, beta01, startT, beta2, sigma, sigma2, sigma3, gamma, b, mu, gamma2, d  = p

        new_index, extended_actual, extended_recovered, extended_death, y0, y1, y2, y3, y4, y5 \
                = self.predict(beta0, beta01, startT, beta2, sigma, sigma2, sigma3, gamma, b, mu, gamma2, d,  \
                    self.data, self.recovered, self.death, self.country, self.s_0, \
                    self.e_0, self.a_0, self.i_0, self.r_0, self.d_0)

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

        df.to_pickle('./data/SEAIRDv5_Yabox_'+self.country+'.pkl')

        print(f"country={self.country}, beta0={beta0:.8f}, beta01={beta01:.8f}, startT={startT:.8f}, beta2={beta2:.8f}, 1/sigma={1/sigma:.8f},"+\
            f" 1/sigma2={1/sigma2:.8f},1/sigma3={1/sigma3:.8f}, gamma={gamma:.8f}, b={b:.8f},"+\
            f" gamma2={gamma2:.8f}, d={d:.8f}, mu={mu:.8f}, r_0:{((beta0+beta01)/gamma):.8f}")
        
        print(self.country+" is done!")

    #plotting
    def trainPlot(self):

        smoothType="Step22" #"SmoothStep2" #"SmoothStep" #"Step"

        df = loadDataFrame('./data/SEAIRDv5_Yabox_'+self.country+'.pkl')

        #calcula data máxima dos gráficos
        #100 dias é usado como máximo dos cálculos da derivada das mortes
        lastDate=df.index.max()
        maxDate= datetime.strptime(lastDate, '%m/%d/%y') + timedelta(days = 100) #"2020-08-31"
        # maxDateStr = maxDate.strftime("%Y-%m-%d")
        df = df[df.index<=datetime.strftime(maxDate, '%m/%d/%y')]
        self.predict_range=100

        color_bg = '#FEF1E5'
        # lighter_highlight = '#FAE6E1'
        darker_highlight = '#FBEADC'
        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(15, 10),facecolor=color_bg)
        ax.patch.set_facecolor(darker_highlight)
        # Hide the right and top spines
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Adding a title and a subtitle
        plt.text(x = 0.02, y = 1.1, s = "Novel SEAIR-D Model Results for "+self.country,
                    fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes, 
                    fontproperties=heading_font)
        plt.text(x = 0.02, y = 1.05,
                    s = "evolutionary optimization by Yabox",
                    fontsize = 26, alpha = .85,transform=ax.transAxes, 
                    fontproperties=subtitle_font)

        #limits for plotting
        ax.set_ylim((0, max(df.iloc[:]['susceptible'])*1.1))

        #plot general plot
        df.plot(ax=ax) #,style=['-','-','-','o','-','x','-','s','-'])

        #format legend
        ax.legend(frameon=False)

        #authorship
        plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=12, 
        xy=(1.04, 0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='right',rotation=90)
        plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', fontsize=12, 
        xy=(1.045,0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='left',rotation=90)

        # Hide grid lines
        # ax.grid(False)

        #set country
        country=self.country
        strFile ="./results/modelSEAIRDBeta"+smoothType+country+"Yabox.png"

        #remove previous file
        if os.path.isfile(strFile):
            os.remove(strFile)   # Opt.: os.system("del "+strFile)

        #plot layout
        fig.tight_layout()

        #save figure and close
        fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
        plt.show()
        plt.close()

        #X variables for plotting
        plotX=df.index[range(0,self.predict_range)]
        plotXt=df.index[range(0,len(df.infected))]

        #format background
        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(15, 10),facecolor=color_bg)
        ax.patch.set_facecolor(darker_highlight)

        # Hide the right and top spines
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Hide grid lines
        # ax.grid(False)

        # Adding a title and a subtitle
        plt.text(x = 0.02, y = 1.1, s = "Zoom at Novel SEAIR-D Model Results for "+self.country,
                    fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                    fontproperties=heading_font)
        plt.text(x = 0.02, y = 1.05,
                    s = "evolutionary optimization by Yabox",
                    fontsize = 26, alpha = .85,transform=ax.transAxes, 
                    fontproperties=subtitle_font)

        #plot thicks and limits
        plt.xticks(np.arange(0, self.predict_range, self.predict_range/8))
        ax.set_ylim(0,max(df.infected)*1.1)

        #plot Zoom figure
        ax.plot(plotX,df.infected,'y-',label="Infected")
        ax.plot(plotX,df.predicted_recovered,'c-',label="Recovered")
        ax.plot(plotX,df.predicted_deaths,'m-',label="Deaths")
        ax.plot(plotXt,df.infected_data,'o',label="Infected data")
        ax.plot(plotXt,df.death_data,'x',label="Death data")
        ax.plot(plotXt,df.recovered,'s',label="Recovered data")

        #format legend
        ax.legend(frameon=False)

        #authorship
        plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=12, 
        xy=(1.04, 0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='right',rotation=90)
        plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', fontsize=12, 
        xy=(1.045,0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='left',rotation=90)

        #plot layout
        fig.tight_layout()

        #file name to be saved
        strFile ="./results/ZoomModelSEAIRDBeta"+smoothType+country+"Yabox.png"

        #remove previous file
        if os.path.isfile(strFile):
            os.remove(strFile)   # Opt.: os.system("del "+strFile)

        #figure save and close
        fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
        plt.show()
        plt.close()

#objective function Odeint solver
def create_lossOdeint(data, recovered, \
            death, s_0, e_0, a_0, i_0, r_0, d_0, startNCases, \
                 weigthCases, weigthRecov):
    def lossOdeint(point):
        beta0, beta01, startT, beta2, sigma, sigma2, sigma3, gamma, b, mu, gamma2, d = point

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
            y2=sigma*E*(1-p)-gamma2*A-mu*A #A
            y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I #I
            y4=b*I+d*A+sigma2*I-mu*R #R
            y5=(-(y0+y1+y2+y3+y4)) #D
            return [y0,y1,y2,y3,y4,y5]

        #solve ODE system
        y0=[s_0,e_0,a_0,i_0,r_0,d_0]
        size = len(data)+1
        tspan=np.arange(0, size+100, 1)
        res=odeint(SEAIRD,y0,tspan)

        # calculate fitting error by using numpy.where
        ix= np.where(data.values >= startNCases)
        l1 = np.mean((res[ix[0],3] - data.values[ix])**2)
        l2 = np.mean((res[ix[0],5] - death.values[ix])**2)
        l3 = np.mean((res[ix[0],4] - recovered.values[ix])**2)

        #weight for cases
        u = weigthCases
        #weight for recovered
        w = weigthRecov 
        #weight for deaths
        v = max(0,1. - u - w)

        #for deaths
        dDeath=np.diff(res[1:size,5])
        dDeathData=np.diff(death.values)
        dErrorX=(dDeath-dDeathData)**2
        dErrorD=np.mean(dErrorX[-8:]) 

        #for infected
        dInf=np.diff(res[1:size,3])
        dInfData=np.diff(data.values)
        dErrorY=(dInf-dInfData)**2
        dErrorI=np.mean(dErrorY[-8:])

        #objective function
        gtot=u*(l1+0.05*dErrorI) + v*(l2+0.2*dErrorD) + w*l3

        #penalty function for negative derivative at end of deaths
        NegDeathData=np.diff(res[:,5])
        dNeg=np.mean(NegDeathData[-5:]) 
        correctGtot=max(abs(dNeg),0)**2

        #final objective function
        gtot=0*correctGtot-10*min(np.sign(dNeg),0)*correctGtot+gtot

        return gtot 
    return lossOdeint

#main program SEAIRD model

def main(countriesExt,opt):
    
    countries, download, startdate, predict_range , s0, e0, a0, i0, r0, k0, startNCases, \
        weigthCases, weigthRecov = parse_arguments()

    if not countriesExt=="":
        countries=countriesExt

    if download:
        data_d = load_json("./data_url.json")
        download_data(data_d)

    sumCases_province('data/time_series_19-covid-Confirmed.csv', 'data/time_series_19-covid-Confirmed-country.csv')
    sumCases_province('data/time_series_19-covid-Recovered.csv', 'data/time_series_19-covid-Recovered-country.csv')
    sumCases_province('data/time_series_19-covid-Deaths.csv', 'data/time_series_19-covid-Deaths-country.csv')

    cleanRecovered=False
    results=[]
    for country in countries:
        
        #OK 04/22
        if country=="Germany":
            startdate="3/3/20"
            s0=3e6*2*1.1*2
            e0=1e-4
            i0=265
            r0=-5000
            k0=0
            #start fitting when the number of cases >= start
            startNCases=0
            #how many days is the prediction
            predict_range=150
            #weigth for fitting data
            weigthCases=0.8
            weigthRecov=0.1
            #weightDeaths = 1 - weigthCases - weigthRecov
        
        #OK 04/22
        if country=="Spain":
            startdate="3/3/20"
            s0=3e6*2*1.2
            e0=1e-4
            i0=265
            r0=0
            k0=0
            #start fitting when the number of cases >= start
            startNCases=0
            #how many days is the prediction
            predict_range=150
            #weigth for fitting data
            weigthCases=0.5
            weigthRecov=0.1
            #weightDeaths = 1 - weigthCases - weigthRecov
        
        if country=="Belgium":
            startdate="3/3/20"
            s0=3e6/2/4*1.1*1.5
            e0=1e-4
            i0=265
            r0=0
            k0=0
            #start fitting when the number of cases >= start
            startNCases=0
            #how many days is the prediction
            predict_range=150
            #weigth for fitting data
            weigthCases=0.2
            weigthRecov=0.1
        #weightDeaths = 1 - weigthCases - weigthRecov
    
        if country=="Brazil":
            # fitting initial conditions
            # s0=6759434, date=0, i0=583, wrec=0.3212, wcases=0.1245
            startdate="3/2/20"
            s0=3.0e6*5.0 #3.0e6*3.5 not clean #3.0e6*2.5 clean
            e0=1e-4
            i0=500 #clean #500 not clean
            r0=10e3 #14000 #12e3 #14e3 #14000 #14000 not clean #0 clean
            k0=65
            #start fitting when the number of cases >= start
            startNCases=150
            #how many days is the prediction
            predict_range=200
            #weigth for fitting data
            weigthCases=0.55 #0.1221 opt #0.4 clean #0.15 not clean
            weigthRecov=0.1 #486
            #weightDeaths = 1 - weigthCases - weigthRecov
            cleanRecovered=False
    
        if country=="China":
            startdate="1/26/20"
            s0=120000*4.5 #600e3
            e0=1e-4
            i0=800
            r0=-31.5e3
            k0=0
            #start fitting when the number of cases >= start
            startNCases=0
            #how many days is the prediction
            predict_range=150
            #weigth for fitting data
            weigthCases=0.5
            weigthRecov=0.1
            #weightDeaths = 1 - weigthCases - weigthRecov
    
        if country=="Italy":
            startdate="2/28/20" #+3
            s0=1275525 #2.1e6 #3e6*4*2*2*0.7*1.2*1.1
            e0=1e-4
            i0=286 #200
            r0=592
            k0=424
            #start fitting when the number of cases >= start
            startNCases=100
            #how many days is the prediction
            predict_range=150
            #weigth for fitting data
            weigthCases=0.1043
            weigthRecov=0.1054
            #weightDeaths = 1 - weigthCases - weigthRecov
    
        if country=="France":
            startdate="3/4/20"
            s0=1113113 #1e6 #1.5e6*1.5*120/80*1.05
            e0=1e-4
            i0=678
            r0=1305
            k0=844
            #start fitting when the number of cases >= start
            startNCases=100
            #how many days is the prediction
            predict_range=150
            #weigth for fitting data
            weigthCases=0.2570
            weigthRecov=0.1093
            #weightDeaths = 1 - weigthCases - weigthRecov
    
        #OK 04/22
        if country=="United Kingdom":
            startdate="2/25/20"
            s0=500e3*4*2*2
            e0=1e-4
            i0=22
            r0=0 #-50
            k0=150
            #start fitting when the number of cases >= start
            startNCases=0
            #how many days is the prediction
            predict_range=150
            #weigth for fitting data
            weigthCases=0.4
            weigthRecov=0.1
            #weightDeaths = 1 - weigthCases - weigthRecov
    
        #OK 04/22
        if country=="US":
            startdate="2/20/20"
            s0=10e6*4
            e0=1e-4
            i0=500
            r0=0
            k0=300
            #start fitting when the number of cases >= start
            startNCases=0
            #how many days is the prediction
            predict_range=150
            #weigth for fitting data
            weigthCases=0.4
            weigthRecov=0.1
            #weightDeaths = 1 - weigthCases - weigthRecov
               
        learner = Learner(country, startdate, predict_range,\
            s0, e0, a0, i0, r0, k0, startNCases, weigthCases, weigthRecov, 
            cleanRecovered)
        if opt!=6 and opt!=0:
            results.append(learner.train())
        learner.trainPlot()
    # results = ray.get(results)
            
#Initial parameters
#Choose here your options

#option
#opt=0 all plots
#opt=1 corona log plot
#opt=2 logistic model prediction
#opt=3 bar plot with growth rate
#opt=4 log plot + bar plot
#opt=5 SEAIR-D Model and plot
#opt=6 only plot SEAIR-D Model
opt=5

#prepare data for plotting log chart
country1="US"
country2="Italy"
country3="Brazil"
country4="Japan"
country5="Sweden"

#load new confirmed cases
data_d = load_json("./data_url.json")
download_data(data_d)

#sum provinces under same country
sumCases_province('data/time_series_19-covid-Confirmed.csv', \
                  'data/time_series_19-covid-Confirmed-country.csv')

#load CSV file
dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')
df=pd.read_csv('data/time_series_19-covid-Confirmed-country.csv', \
    delimiter=',',parse_dates=True, date_parser=dateparse,header=None)
df=df.transpose()

#associate data to countries selected
[time1,cases1]=getCases(df,country1)
[time2,cases2]=getCases(df,country2)
[time3,cases3]=getCases(df,country3)
[time4,cases4]=getCases(df,country4)
[time5,cases5]=getCases(df,country5)

#plot version - changes the file name png
version="1"

#choose country for curve fitting
#choose country for growth curve
#one of countries above
country="Brazil"

#list of countries for SEAIRD model
#countriesExt=["Italy","United Kingdom","China","France","US", \
#                "Brazil", "Belgium", "Germany", "Spain"]
#countriesExt=["Italy","China","France", \
#                "Brazil", "Belgium", "Spain"]
#countriesExt=["Germany","United Kingdom","Italy"]
# countriesExt=["Brazil"]
# countriesExt=["China"]
countriesExt=["Brazil"]

#initial vars
a = 0.0
b = 0.0
c = 0.0 
date = []

if opt==1 or opt==0 or opt==4:

    model='SEAIRD' 

    df = loadDataFrame('./data/SEAIRDv5_Yabox_'+country+'.pkl')
    time6, cases6 = predictionsPlot(df,160,1200)

    #model
    #33% per day
    growth = 1.1
    x,y = logGrowth(growth,120)

    #50% per day
    growth1 = 1.25
    x1,y1 = logGrowth(growth1,60)

    # Plot the data
    #ax.figure(figsize=(19.20,10.80))
    color_bg = '#FEF1E5'
    # lighter_highlight = '#FAE6E1'
    darker_highlight = '#FBEADC'
    plt.rcParams['figure.figsize'] = [9, 7]
    plt.rc('font', size=14)
    fig, ax = plt.subplots(facecolor=color_bg)
    ax.patch.set_facecolor(darker_highlight)
    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.plot(time2, cases2,'r-',label=country2,markevery=3) 
    plt.plot(time4, cases4,'m-',label=country4,markevery=3) 
    plt.plot(time5, cases5,'c-',label=country5,markevery=3) 
    plt.plot(time3, cases3,'g-',label=country3,markevery=3) 
    plt.plot(time6, cases6,'--',c='0.6',label=country3+" "+model) 
    plt.plot(time1, cases1,'b-',label=country1) 
    plt.plot(x, y,'y--',label='{:.1f}'.format((growth-1)*100)+'% per day',alpha=0.3)
    plt.plot(x1, y1,'y-.',label='{:.1f}'.format((growth1-1)*100)+'% per day',alpha=0.3) 
    plt.rc('font', size=11)

    plt.annotate(country3+" {:.1f} K".format(cases3[len(cases3)-1]/1000), # this is the text
        (time3[len(cases3)-1],cases3[len(cases3)-1]), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(-10,2), # distance from text to points (x,y)
        ha='right') # horizontal alignment can be left, right or center

    idx=int(np.argmax(cases6))
    plt.annotate("Peak {:.1f} K".format(max(cases6)/1000), # this is the text
        (time6[idx],cases6[idx]), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(20,3), # distance from text to points (x,y)
        ha='right') # horizontal alignment can be left, right or center

    # plt.annotate(country2+" {:.1f} K".format(cases2[len(cases2)-1]/1000), # this is the text
    #     (time2[len(cases2)-1],cases2[len(cases2)-1]), # this is the point to label
    #     textcoords="offset points", # how to position the text
    #     xytext=(0,15), # distance from text to points (x,y)
    #     ha='center') # horizontal alignment can be left, right or center
    
    plt.annotate(country1+" {:.1f} K".format(cases1[len(cases1)-1]/1000), # this is the text
        (time1[len(cases1)-1],cases1[len(cases1)-1]), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(-8,10), # distance from text to points (x,y)
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
    # Hide grid lines
    # ax.grid(False)

    # Adding a title and a subtitle
    plt.text(x = 0.02, y = 1.1, s = "Corona virus growth",
                fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                fontproperties=heading_font)
    plt.text(x = 0.02, y = 1.05,
                s = "Comparison selected countries and model for "+country3,
                fontsize = 26, alpha = .85,transform=ax.transAxes, 
                    fontproperties=subtitle_font)
    plt.legend(frameon=False)
    fig.tight_layout()

    #save figs
    strFile ='./results/coronaPythonEN_'+version+'.png'
    fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())

    # Show the plot
    plt.show() 
    plt.close()

if opt==2 or opt==0:

    if opt==2:
        #model
        #33% per day
        x,y=logGrowth(1.33,90)

        #25% per day
        x1,y1=logGrowth(1.25,90)

    #model fitting

    casesFit=cases3
    timeFit=time3
    maxCases=50e3
    maxTime=50
    guessExp=0.5

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
        maxCases=1.5e6
        maxTime=160
        guessExp=2

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
    exp_fit = curve_fit(exponential_model,timeFit,casesFit,p0=[guessExp*2,guessExp/2,guessExp/4],maxfev=10000)

    # Plot the data
    #ax.figure(figsize=(19.20,10.80))
    color_bg = '#FEF1E5'
    # lighter_highlight = '#FAE6E1'
    darker_highlight = '#FBEADC'
    plt.rcParams['figure.figsize'] = [9, 7]
    plt.rc('font', size=14)
    fig, ax = plt.subplots(facecolor=color_bg)
    ax.patch.set_facecolor(darker_highlight)
    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #plot
    pred_x = list(range(len(time3)+1,maxTime))
    # Predicted logistic curve
    ax.plot(time3+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in time3+pred_x], label="Logistic model" )
    # Predicted exponential curve
    ax.plot(time3+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in time3+pred_x], label="Exponential model" )
    # Real data
    ax.scatter(timeFit,casesFit,label="Real cases "+country,color="red")

    #axis, limits and legend
    plt.xlabel("Days since 100th case")
    plt.ylabel("Total number of infected people in "+country)
    plt.ylim((min(y)*0.9,maxCases))
    plt.legend(frameon=False)

    plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=9, 
            xy=(1.05, 0.05), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90)
    plt.annotate('Source: https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions', fontsize=7, 
            xy=(1.06,0.05), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='left',rotation=90)
    
    plt.annotate('Total infected = {:.2f} M'.format(fit[0][2]/1e6), fontsize=12, 
            xy=(0.97,0.60), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right')

    plt.annotate('Max Infection at {:.0f} day'.format(fit[0][1]), fontsize=12, 
            xy=(fit[0][1],logistic_model(fit[0][1],fit[0][0],fit[0][1],fit[0][2])),
            xytext=(-35, 0), textcoords='offset points', arrowprops={'arrowstyle': '-|>'},
            ha='right')

    # Hide grid lines
    # ax.grid(False)

    # Adding a title and a subtitle
    plt.text(x = 0.02, y = 1.1, s = "Curve Fitting with Simple Math Functions",
                fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                fontproperties=heading_font)
    plt.text(x = 0.02, y = 1.05,
                s = "Logistic and Exponential Function fitted with real data from "+country,
                fontsize = 26, alpha = .85,transform=ax.transAxes, 
                    fontproperties=subtitle_font)
    plt.legend(frameon=False)
    fig.tight_layout()

    #save figs
    strFile ='./results/coronaPythonModelEN'+country+'.png'
    fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())

    plt.show() 
    plt.close()

if opt==3 or opt==0 or opt==4:
    
    casesGrowth=cases3
    timeGrowth=time3
    maxCases=30e3
    maxTime=50
    guessExp=0.5
    
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

    plt.rcParams['figure.figsize'] = [9, 7]
    plt.rc('font', size=14)
    #ax.figure(figsize=(19.20,10.80))
    color_bg = '#FEF1E5'
    # lighter_highlight = '#FAE6E1'
    darker_highlight = '#FBEADC'

    colors = cm.rainbow(np.asfarray(growth,float) / float(max(np.asfarray(growth,float))))
    fig, ax = plt.subplots(facecolor=color_bg)
    plot = ax.scatter(growth, growth, c = growth, cmap = 'rainbow')
    fig.colorbar(plot)
    ax.cla()

    # Plot the data
    ax.patch.set_facecolor(darker_highlight)
    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #Plot bars
    plt.bar(ind, bars, color=colors)
    plt.xlabel('Days since 100th case')

    # Make the y-axis label and tick labels match the line color.
    plt.ylabel(country+' growth official cases per day [%]') 

    #Plot a line
    plt.axhline(y=10,color='r',linestyle='--')

    plt.annotate("doubling each 10 days", # this is the text
        (75,10), # this is the point to label
        textcoords="offset points", # how to position the text
        xytext=(0,5), # distance from text to points (x,y)
        ha='right', weight='bold',fontsize=14) # horizontal alignment can be left, right or center

    # # Text on the top of each barplot
    # for i in range(1,len(ind),5):
    #     plt.text(x = ind[i]-0.5 , y = 1.15*growth[i], s = " {:.1f}%".format(growth[i]), size = 8)

    plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=10, 
            xy=(1.24, 0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90)
    plt.annotate('Source: https://github.com/CSSEGISandData/COVID-19.git', fontsize=10, 
            xy=(1.25,0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='left',rotation=90)

    # Hide grid lines
    # ax.grid(False)

    # Adding a title and a subtitle
    plt.text(x = 0.02, y = 1.1, s = "Relative Growth per Day",
                fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                fontproperties=heading_font)
    plt.text(x = 0.02, y = 1.05,
                s = "Real Data for "+country3,
                fontsize = 26, alpha = .85,transform=ax.transAxes, 
                    fontproperties=subtitle_font)
    fig.tight_layout()

    #save figs
    strFile ='./results/coronaPythonGrowthEN_'+country+'.png'
    fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())

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

    plt.rcParams['figure.figsize'] = [9, 7]
    plt.rc('font', size=14)
    #ax.figure(figsize=(19.20,10.80))
    color_bg = '#FEF1E5'
    # lighter_highlight = '#FAE6E1'
    darker_highlight = '#FBEADC'

    colors = cm.rainbow(np.asfarray(growth,float) / float(max(np.asfarray(growth,float))))
    fig, ax = plt.subplots(facecolor=color_bg)
    plot = ax.scatter(growth, growth, c = growth, cmap = 'rainbow')
    fig.colorbar(plot)
    ax.cla()

    ax.patch.set_facecolor(darker_highlight)
    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    #Plot bars
    plt.bar(ind, bars, color=colors)
    plt.xlabel('Days since 100th case')

    # Make the y-axis label and tick labels match the line color.
    plt.ylabel(country+' growth official cases per day [cases]') 

    # Plot a line
    # plt.axhline(y=300,color='r',linestyle='--')

    # plt.annotate("Expected per day", # this is the text
    #     (5,310), # this is the point to label
    #     textcoords="offset points", # how to position the text
    #     xytext=(0,5), # distance from text to points (x,y)
    #     ha='center') # horizontal alignment can be left, right or center

    # # Text on the top of each barplot
    # for i in range(1,len(ind),5):
    #     plt.text(x = ind[i]-0.5 , y = 1.3*growth[i], s = " {:.0f}".format(growth[i]), size = 10)

    plt.annotate('Dr. Guilherme A. L. da Silva, www.ats4i.com', fontsize=10, 
            xy=(1.28, 0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90)
    plt.annotate('Source: https://github.com/CSSEGISandData/COVID-19.git', fontsize=10, 
            xy=(1.29,0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='left',rotation=90)

    # Hide grid lines
    # ax.grid(False)

    # Adding a title and a subtitle
    plt.text(x = 0.02, y = 1.1, s = "Absolute Growth per Day",
                fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                fontproperties=heading_font)
    plt.text(x = 0.02, y = 1.05,
                s = "Real Data for "+country3,
                fontsize = 26, alpha = .85,transform=ax.transAxes, 
                    fontproperties=subtitle_font)
    fig.tight_layout()

    #save figs
    strFile ='./results/coronaPythonGrowthDeltaCasesEN_'+country+'.png'
    fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())

    plt.show() 
    plt.close()

if opt==0 or opt==5 or opt==6:

    if __name__ == '__main__':
        main(countriesExt,opt)
