# Import the necessary packages and modules
import sys
import argparse
import json
import os
import urllib.request
from matplotlib import cm
import numpy as np
import pandas as pd
from csv import reader
from csv import writer
from datetime import datetime,timedelta
from scipy.optimize import minimize
from scipy.integrate import odeint
from yabox import DE
from tqdm import tqdm

import matplotlib as mpl
# mpl.use('agg')
# mpl.use('TkAgg')
# mpl.use('Qt4Agg')
import matplotlib.pyplot as plt
import matplotlib.style as style
# style.use('fivethirtyeight')

import matplotlib.font_manager as fm
# Font Imports
heading_font = fm.FontProperties(fname='/home/ats4i/playfair-display/PlayfairDisplay-Regular.ttf', size=22)
subtitle_font = fm.FontProperties(fname='/home/ats4i/Roboto/Roboto-Regular.ttf', size=12)

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 7

#parallel computation
# import ray
# ray.shutdown()
# ray.init(num_cpus=3)

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
    for j in range(0,nPoints):
        if float(df.infected[j])>=startCases:
            cases.append(float(df.infected[j]))
            time.append(float(j1))
            j1+=1
    return time,cases

def savePlot(strFile):
    if os.path.isfile(strFile):
        os.remove(strFile)   # Opt.: os.system("del "+strFile)
    plt.savefig(strFile,dpi=600)

def savePlotBG(strFile):
    if os.path.isfile(strFile):
        os.remove(strFile)   # Opt.: os.system("del "+strFile)
    plt.savefig(strFile,dpi=300,facecolor='#FBEADC', edgecolor='#FEF1E5') #, transparent=True)

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
    def predict(self, beta, beta2, sigma, sigma2, sigma3, gamma, b, mu, data, \
                recovered, death, country, s_0, e_0, a_0, i_0, r_0, d_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SEAIRD(y,t):
            S = y[0]
            E = y[1]
            A = y[2]
            I = y[3]
            R = y[4]
            p=0.2
            # beta2=beta
            y0=-(beta2*A+beta*I)*S-mu*S #S
            y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
            y2=sigma*E*(1-p)-gamma*A-mu*A #A
            y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I #I
            y4=b*I+b*A+sigma2*I-mu*R #R
            y5=(-(y0+y1+y2+y3+y4)) #D
            return [y0,y1,y2,y3,y4,y5]
        
        y0=[s_0,e_0,a_0,i_0,r_0,d_0]
        tspan=np.arange(0, size, 1)
        res=odeint(SEAIRD,y0,tspan,hmax=0.01)

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

        bounds=[(1e-12, .2),(1e-12, .2),(1/120 ,0.4),(1/120, .4),
        (1/120, .4),(1e-12, .4),(1e-12, .4),(1e-12, .4)]
        maxiterations=1000
        f=create_lossOdeint(self.data, self.recovered, self.death, self.s_0, \
            self.e_0, self.a_0, self.i_0, self.r_0, self.d_0, self.startNCases, \
            self.weigthCases, self.weigthRecov)
        de = DE(f, bounds, maxiters=maxiterations)
        i=0
        with tqdm(total=maxiterations*750) as pbar:
            for step in de.geniterator():
                idx = step.best_idx
                norm_vector = step.population[idx]
                best_params = de.denormalize([norm_vector])
                pbar.update(i)
                i+=1
        p=best_params[0]   

        #parameter list for optimization
        #beta, beta2, sigma, sigma2, sigma3, gamma, b, mu

        beta, beta2, sigma, sigma2, sigma3, gamma, b, mu = p
        new_index, extended_actual, extended_recovered, extended_death, y0, y1, y2, y3, y4, y5 \
                = self.predict(beta, beta2, sigma, sigma2, sigma3, gamma, b, mu, \
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

        df.to_pickle('./data/SEAIRD_results_'+self.country+'.pkl')

        print(f"country={self.country}, beta={beta:.8f}, beta2={beta2:.8f}, 1/sigma={1/sigma:.8f},"+\
            f" 1/sigma2={1/sigma2:.8f},1/sigma3={1/sigma3:.8f}, gamma={gamma:.8f}, b={b:.8f},"+\
            f" mu={mu:.8f}, r_0:{(beta/gamma):.8f}")
        
        print(self.country+" is done!")

    #plotting
    def trainPlot(self):

        df = loadDataFrame('./data/SEAIRD_results_'+self.country+'.pkl')
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

        ax.set_ylim((0, max(df.iloc[:]['susceptible'])*1.1))
        df.plot(ax=ax) #,style=['-','-','-','o','-','x','-','s','-'])
        ax.legend(frameon=False)

        plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=12, 
        xy=(1.04, 0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='right',rotation=90)
        plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', fontsize=12, 
        xy=(1.045,0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='left',rotation=90)

        # Hide grid lines
        ax.grid(False)

        country=self.country
        strFile ="./results/modelSEAIRDOptGlobalOptimum"+country+"AdjustICresults.png"
        #savePlotBG(strFile)
        fig.tight_layout()
        fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
        plt.show()
        plt.close()

        plotX=df.index[range(0,self.predict_range)]
        plotXt=df.index[range(0,len(df.infected))]

        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(15, 10),facecolor=color_bg)
        ax.patch.set_facecolor(darker_highlight)
        # Hide the right and top spines
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Hide grid lines
        ax.grid(False)

        # Adding a title and a subtitle
        plt.text(x = 0.02, y = 1.1, s = "Zoom at Novel SEAIR-D Model Results for "+self.country,
                    fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                    fontproperties=heading_font)
        plt.text(x = 0.02, y = 1.05,
                    s = "evolutionary optimization by Yabox",
                    fontsize = 26, alpha = .85,transform=ax.transAxes, 
                    fontproperties=subtitle_font)

        plt.xticks(np.arange(0, self.predict_range, self.predict_range/8))
        ax.set_ylim(0,max(df.infected)*1.1)
        ax.plot(plotX,df.infected,'y-',label="Infected")
        ax.plot(plotX,df.predicted_recovered,'c-',label="Recovered")
        ax.plot(plotX,df.predicted_deaths,'m-',label="Deaths")
        ax.plot(plotXt,df.infected_data,'o',label="Infected data")
        ax.plot(plotXt,df.death_data,'x',label="Death data")
        ax.plot(plotXt,df.recovered,'s',label="Recovered data")
        ax.legend(frameon=False)

        plt.annotate('Dr. Guilherme Araujo Lima da Silva, www.ats4i.com', fontsize=12, 
        xy=(1.04, 0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='right',rotation=90)
        plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', fontsize=12, 
        xy=(1.045,0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='left',rotation=90)

        fig.tight_layout()
        strFile ="./results/ZoomModelSEAIRDOpt"+country+"AdjustICresults.png"
        #savePlotBG(strFile)
        fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
        plt.show()
        plt.close()

#objective function Odeint solver

def create_lossOdeint(data, recovered, \
            death, s_0, e_0, a_0, i_0, r_0, d_0, startNCases, \
                 weigthCases, weigthRecov):
    def lossOdeint(point):
        size = len(data)
        beta, beta2, sigma, sigma2, sigma3, gamma, b, mu = point
        def SEAIRD(y,t):
            S = y[0]
            E = y[1]
            A = y[2]
            I = y[3]
            R = y[4]
            p=0.2
            # beta2=beta
            y0=-(beta2*A+beta*I)*S-mu*S #S
            y1=(beta2*A+beta*I)*S-sigma*E-mu*E #E
            y2=sigma*E*(1-p)-gamma*A-mu*A #A
            y3=sigma*E*p-gamma*I-sigma2*I-sigma3*I-mu*I #I
            y4=b*I+b*A+sigma2*I-mu*R #R
            y5=(-(y0+y1+y2+y3+y4)) #D
            return [y0,y1,y2,y3,y4,y5]

        y0=[s_0,e_0,a_0,i_0,r_0,d_0]
        tspan=np.arange(0, size, 1)
        res=odeint(SEAIRD,y0,tspan, hmax=0.01)

        tot=0
        l1=0
        l2=0
        l3=0
        for i in range(0,len(data.values)):
            if data.values[i]>startNCases:
                l1 = l1+(res[i,3] - data.values[i])**2
                l2 = l2+(res[i,5] - death.values[i])**2
                l3 = l3+(res[i,4] - recovered.values[i])**2
                tot+=1
        l1=np.sqrt(l1/max(1,tot))
        l2=np.sqrt(l2/max(1,tot))
        l3=np.sqrt(l3/max(1,tot))
        
        #weight for cases
        u = weigthCases
        #weight for recovered
        w = weigthRecov 
        #weight for deaths
        v = max(0,1. - u - w)
        return u*l1*0.5 + v*l2*0.3 + w*l3*0.2 
    return lossOdeint
