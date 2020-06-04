import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime,timedelta
from scipy.optimize import curve_fit
from matplotlib import cm

import matplotlib.font_manager as fm
# Font Imports
heading_font = fm.FontProperties(fname='/home/ats4i/playfair-display/PlayfairDisplay-Regular.ttf', size=22)
subtitle_font = fm.FontProperties(fname='/home/ats4i/Roboto/Roboto-Regular.ttf', size=12)


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

def load_confirmed(state, startdate):
        dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
        df = pd.read_csv('./data/confirmados.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
        y=[]
        x=[]
        for i in range(0,len(df.date)):
            y.append(df[state].values[i])
            x.append(df.date.values[i])
        df2=pd.DataFrame(data=y,index=x,columns=[""])
        df2=df2[startdate:]
        return df2

def load_dead(state, startdate):
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv('./data/mortes.csv',delimiter=',',parse_dates=True, date_parser=dateparse)
    y=[]
    x=[]
    for i in range(0,len(df.date)):
        y.append(df[state].values[i])
        x.append(df.date.values[i])
    df2=pd.DataFrame(data=y,index=x,columns=[""])
    df2=df2[startdate:]
    return df2

def extend_index(index, new_size):
    values = index.values
    #current = datetime.strptime(index[-1], '%Y-%m-%d')
    current = index[-1]
    while len(values) < new_size:
        current = current + timedelta(days=1)
        values = np.append(values, current)
    return values

def covid_plots(state, state4Plot,\
                startdate="2020-03-15",predict_range = 60, startCase = 180, opt = 5, version = "1", show = False):
    
    #Initial parameters
    #Choose here your options

    #option
    #opt=0 all plots
    #opt=1 corona log plot
    #opt=2 logistic model prediction
    #opt=3 bar plot with growth rate
    #opt=4 log plot + bar plot
    #opt=5 SEAIR-D Model

    #plot version - changes the file name png

    #choose state
    stateSEAIRD=state

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
    df = pd.read_csv('./data/confirmados.csv',delimiter=',',parse_dates=True, date_parser=dateparse)

    #prepare data for plotting
    state1=state4Plot[0]
    [time1,cases1]=getCases(df,state1)
    state2=state4Plot[1]
    [time2,cases2]=getCases(df,state2)
    state3=state4Plot[2]
    [time3,cases3]=getCases(df,state3)
    state4=state4Plot[3]
    [time4,cases4]=getCases(df,state4)
    state5=state4Plot[4]
    [time5,cases5]=getCases(df,state5)

    if opt==1 or opt==0 or opt==4:

        model='SEAIRD_sigmaOpt'

        df = loadDataFrame('./data/SEAIRD_'+state+version+'.pkl')
        time6, cases6 = predictionsPlot(df,startCase)
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
        plt.ylim(1e2, 1e5)
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

        plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS', fontsize=10, 
                xy=(1.05, 0.1), xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                ha='right',rotation=90)

        plt.xlabel('Days after 100th case')
        plt.ylabel('Official registered cases')
        plt.yscale('log')
        plt.title("Corona virus growth")
        plt.legend()

        #save figs
        savePlot('./results/coronaPythonEN_'+version+'.png')

        # Show the plot
        if show:
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
            maxCases=50e3
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

        plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS', fontsize=10, 
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
        
        if show:
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

        plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS', fontsize=10, 
                xy=(1.24, 0.1), xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                ha='right',rotation=90)

        #save figs
        strFile ='./results/coronaPythonGrowthEN_'+state+'.png'
        savePlot(strFile)
        
        if show:
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

        plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS', fontsize=10, 
                xy=(1.24, 0.1), xycoords='axes fraction',
                xytext=(0, 0), textcoords='offset points',
                ha='right',rotation=90)

        #save figs
        strFile ='./results/coronaPythonGrowthDeltaCasesEN_'+state+'.png'
        savePlot(strFile)
        
        if show:
            plt.show() 
            plt.close()
        
        
    if opt==5 or opt==0:
        df = loadDataFrame('./data/SEAIRD_'+state+version+'.pkl')
        
        actual = load_confirmed(state, startdate)
        death = load_dead(state,startdate)
        extended_actual = actual.values
        extended_death = death.values
        
        new_index = extend_index(df.index, predict_range)

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
        plt.text(x = 0.02, y = 1.1, s = "SEAIR-D Model for "+state,
                    fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes, 
                    fontproperties=heading_font)
        plt.text(x = 0.02, y = 1.05,
                    s = "Optimization fitted real data",
                    fontsize = 26, alpha = .85,transform=ax.transAxes, 
                    fontproperties=subtitle_font)

        #limits for plotting
        ax.set_ylim((0, max(df.iloc[:]['susceptible'])*1.1))

        ax.set_title("SEAIR-D Model for "+state)
        ax.xaxis_date()
        ax.plot(df['susceptible'],'g-',label="Susceptible")
        ax.plot(df['exposed'],'r-',label="Exposed")
        ax.plot(df['asymptomatic'],'b-',label="Asymptomatic")
        #plt.xticks(np.arange(0, predict_range, predict_range/8))
        ax.plot(df['infected'],'y-',label="Infected")
        ax.plot(df['recovered'],'c-',label="Recovered")
        ax.plot(df['deaths'],'m-',label="Deaths")
        ax.plot(new_index[range(0,len(extended_actual))],extended_actual,'o',label="Infected data")
        ax.plot(new_index[range(0,len(extended_death))],extended_death,'x',label="Death data")

        #format legend
        ax.legend(frameon=False)

        #plot margin annotation
        plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS', fontsize=10, 
        xy=(1.04, 0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='right',rotation=90)
        plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', fontsize=10, 
        xy=(1.045,0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='left',rotation=90)

        #plot layout
        fig.tight_layout()

        #file name to be saved
        strFile ="./results/modelSEAIRDOpt"+state+version+".png"

        #remove previous file
        if os.path.isfile(strFile):
            os.remove(strFile)   # Opt.: os.system("del "+strFile)

        #figure save and close
        fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
        if show:
            plt.show()
            plt.close()

        #format background
        plt.rc('font', size=14)
        fig, ax = plt.subplots(figsize=(15, 10),facecolor=color_bg)
        ax.patch.set_facecolor(darker_highlight)

        # Hide the right and top spines
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
  
        ax.set_ylim(0,max(max(df['infected']),max(extended_actual))*1.1)

        # Adding a title and a subtitle
        plt.text(x = 0.02, y = 1.1, s = "Zoom SEAIR-D Model for "+state,
                    fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                    fontproperties=heading_font)
        plt.text(x = 0.02, y = 1.05,
                    s = "Optimization fitted real data",
                    fontsize = 26, alpha = .85,transform=ax.transAxes, 
                    fontproperties=subtitle_font)


        ax.xaxis_date()
        #plt.xticks(np.arange(0, predict_range, predict_range/8))
        ax.plot(df['infected'],'y-',label="Infected")
        ax.plot(df['recovered'],'c-',label="Recovered")
        ax.plot(df['deaths'],'m-',label="Deaths")
        ax.plot(new_index[range(0,len(extended_actual))],extended_actual,'o',label="Infected data")
        ax.plot(new_index[range(0,len(extended_death))],extended_death,'x',label="Death data")
        #format legend
        ax.legend(frameon=False)
               
        plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS', fontsize=10, 
        xy=(1.04, 0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='right',rotation=90)
        plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', fontsize=10, 
        xy=(1.045,0.1), xycoords='axes fraction',
        xytext=(0, 0), textcoords='offset points',
        ha='left',rotation=90)

        #plot layout
        fig.tight_layout()

        #file name to be saved
        strFile ="./results/ZoomModelSEAIRDOpt"+state+version+".png"

        #remove previous file
        if os.path.isfile(strFile):
            os.remove(strFile)   # Opt.: os.system("del "+strFile)

        #figure save and close
        fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
        if show:
            plt.show()
            plt.close()