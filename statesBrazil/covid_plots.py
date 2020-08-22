import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime,timedelta
from scipy.optimize import curve_fit
from matplotlib import cm

# Font Imports
from tempfile import NamedTemporaryFile
import urllib.request
import matplotlib.font_manager as fm

github_url = 'https://github.com/google/fonts/blob/master/ofl/playfairdisplay/static/PlayfairDisplay-Regular.ttf'
url = github_url + '?raw=true'  # You want the actual file, not some html
request = urllib.request.Request(url)
response = urllib.request.urlopen(request)
f = NamedTemporaryFile(delete=False, suffix='.ttf')
f.write(response.read())
f.close()
heading_font = fm.FontProperties(fname=f.name, size=26)

github_url = 'https://github.com/google/fonts/blob/master/apache/roboto/static/Roboto-Regular.ttf'
url = github_url + '?raw=true'  # You want the actual file, not some html
request = urllib.request.Request(url)
response = urllib.request.urlopen(request)
f = NamedTemporaryFile(delete=False, suffix='.ttf')
f.write(response.read())
f.close()
subtitle_font = fm.FontProperties(fname=f.name, size=18)

github_url = 'http://antiyawn.com/uploads/Humor-Sans-1.0.ttf'
url = github_url + '?raw=true'  # You want the actual file, not some html

request = urllib.request.Request(url)
response = urllib.request.urlopen(request)
f = NamedTemporaryFile(delete=False, suffix='.ttf')
f.write(response.read())
f.close()
comic_font = fm.FontProperties(fname=f.name, size=18)

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
    dateparse = lambda x: datetime.strptime(x,  '%Y-%m-%d')
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
                startdate="2020-03-15",predict_range = 60, \
                    startCase = 180, opt = 5, version = "1", \
                        show = False, ratio=0.15, maxDate="2020-08-31",model=""):
    
    #choose state
    stateSEAIRD=state

    #initial vars
    a = 0.0
    b = 0.0
    c = 0.0 
    date = []

    #load CSV file
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    df1 = pd.read_csv('./data/confirmados.csv',delimiter=',',parse_dates=True, date_parser=dateparse,index_col=0)
    df2 = pd.read_csv('./data/mortes.csv',delimiter=',',parse_dates=True, date_parser=dateparse,index_col=0)
    df1=df1.select_dtypes(exclude=['object', 'datetime']) * (1-ratio)
    df=df1.subtract(df2)

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
    
    #load CSV file
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv('./data/confirmados.csv',delimiter=',',parse_dates=True, date_parser=dateparse,index_col=0)
    state10=state
    [time10,cases10]=getCases(df,state10)

    if opt==1 or opt==0 or opt==4:

        model='SEAIRD_Yabox'

        df = loadDataFrame('./data/SEAIRD_'+state+version+'.pkl')
        time6, cases6 = predictionsPlot(df,startCase)
#         time6 = time6[0:60]
#         cases6 = cases6[0:60]

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
        plt.rcParams['figure.figsize'] = [12, 9]
        plt.rc('font', size=14)

        with plt.xkcd():        
            fig, ax = plt.subplots(facecolor=color_bg)
            ax.patch.set_facecolor(darker_highlight)
            # Hide the right and top spines
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #fonts for the thicks
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontproperties(comic_font)
                label.set_fontsize(16) # Size here overrides font_prop

            plt.plot(time2, cases2,'r-',label=state2,markevery=3) 
            plt.plot(time4, cases4,'m-',label=state4,markevery=3) 
            plt.plot(time5, cases5,'c-',label=state5,markevery=3) 
            plt.plot(time3, cases3,'g-',label=state3,markevery=3) 
            plt.plot(time6, cases6,'--',c='0.6',label=state+" "+model) 
            plt.plot(time1, cases1,'b-',label=state1) 
            plt.plot(x, y,'y--',label='{:.1f}'.format((growth-1)*100)+'% per day',alpha=1)
            plt.plot(x1, y1,'y-.',label='{:.1f}'.format((growth1-1)*100)+'% per day',alpha=1) 
            plt.rc('font', size=11)

            plt.annotate(state3+" {:.1f} K".format(cases3[len(cases3)-1]/1000), # this is the text
                (time3[len(cases3)-1],cases3[len(cases3)-1]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(-10,2), # distance from text to points (x,y)
                ha='right',fontproperties=comic_font,fontsize=16) # horizontal alignment can be left, right or center

            idx=int(np.argmax(cases6))
            plt.annotate("Peak {:.1f} K".format(max(cases6)/1000), # this is the text
                (time6[idx],cases6[idx]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(20,5), # distance from text to points (x,y)
                ha='right',fontproperties=comic_font,fontsize=16) # horizontal alignment can be left, right or center

            plt.annotate(state1+" {:.1f} K".format(cases1[len(cases1)-1]/1000), # this is the text
                (time1[len(cases1)-1],cases1[len(cases1)-1]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(12,-20), # distance from text to points (x,y)
                ha='center',fontproperties=comic_font,fontsize=16) # horizontal alignment can be left, right or center

            plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS',fontproperties=subtitle_font,fontsize = 16,
                    xy=(1.06, 0.1), xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    ha='right',rotation=90)
            plt.annotate('Source: https://data.brasil.io',fontproperties=subtitle_font,fontsize = 16,
                    xy=(1.06,0.1), xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    ha='left',rotation=90)

            plt.xlabel('Days after 100th case', fontproperties=comic_font)
            plt.ylabel('Official registered cases', fontproperties=comic_font)
            plt.yscale('log')
            # Hide grid lines
            # ax.grid(False)

            # Adding a title and a subtitle
            plt.text(x = 0.02, y = 1.1, s = "Corona virus growth",
                        fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                        fontproperties=heading_font)
            plt.text(x = 0.02, y = 1.05,
                        s = "Comparison selected states and model for "+state,
                        fontsize = 26, alpha = .85,transform=ax.transAxes, 
                            fontproperties=subtitle_font)
            ax.legend(frameon=False,prop=comic_font,fontsize=16,loc='lower right')
            ax.grid(True, linestyle='--', linewidth='2', color='white',alpha=0.2)

            fig.tight_layout()

            #save figs
            strFile ='./results/coronaPythonEN_'+version+'.png'
            fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())

            # Show the plot
            plt.show() 
            plt.close()

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

        casesFit=cases10
        timeFit=time10
        maxCases=700e3
        maxTime=200
        guessExp=2

        #logistic curve
        fit = curve_fit(logistic_model,timeFit,casesFit,p0=[5,60,500e3])
        print ("Infection speed=",fit[0][0])
        print ("Day with the maximum infections occurred=",int(fit[0][1]))
        print ("Total number of recorded infected people at the infection’s end=",int(fit[0][2]))
    
        #exponential curve
        exp_fit = curve_fit(exponential_model,timeFit,casesFit,p0=[guessExp*2,guessExp/2,guessExp/4],maxfev=10000)
        
        # Plot the data
        #ax.figure(figsize=(19.20,10.80))
        color_bg = '#FEF1E5'
        # lighter_highlight = '#FAE6E1'
        darker_highlight = '#FBEADC'
        plt.rcParams['figure.figsize'] = [12, 9]
        plt.rc('font', size=14)
        
        with plt.xkcd():
            fig, ax = plt.subplots(facecolor=color_bg)
            ax.patch.set_facecolor(darker_highlight)
            # Hide the right and top spines
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            #fonts for the thicks
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontproperties(comic_font)
                label.set_fontsize(16) # Size here overrides font_prop

            #plot
            pred_x = np.arange(len(timeFit),maxTime,1)      
            extendT=np.concatenate([timeFit,pred_x])
            
            # Predicted logistic curve
            ax.plot(extendT, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) 
                                   for i in extendT], label="Logistic model" )
            # Predicted exponential curve
            ax.plot(extendT, [exponential_model(i,exp_fit[0][0],
                                   exp_fit[0][1],exp_fit[0][2]) for i in extendT], label="Exponential model" )
            # Real data
            ax.scatter(timeFit,casesFit,label="Real cases "+state,color="red")

            #axis, limits and legend
            plt.xlabel("Days since 100th case", fontproperties=comic_font)
            plt.ylabel("Total number of infected people in "+state, fontproperties=comic_font)
            plt.ylim((min(y)*0.9,int(1.05*fit[0][2])))
            plt.legend(frameon=False)

            plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS',fontproperties=subtitle_font,fontsize = 16,
                    xy=(1.06, 0.1), xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    ha='right',rotation=90)
            plt.annotate('Source: https://data.brasil.io',fontproperties=subtitle_font,fontsize = 16,
                    xy=(1.06,0.1), xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    ha='left',rotation=90)
            

            plt.annotate('Total infected = {:.2f} M'.format(fit[0][2]/1e6), fontsize=16, 
                    xy=(0.97,0.60), xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    ha='right', fontproperties=comic_font)
            
            plt.annotate('Infection speed = {:.2f}'.format(fit[0][0]), fontsize=16, 
                    xy=(0.96,0.55), xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    ha='right', fontproperties=comic_font)

            plt.annotate('Max Infection at {:.0f} day'.format(fit[0][1]), fontsize=16, 
                    xy=(fit[0][1],logistic_model(fit[0][1],fit[0][0],fit[0][1],fit[0][2])),
                    xytext=(-35, 0), textcoords='offset points', arrowprops={'arrowstyle': '-|>'},
                    ha='right', fontproperties=comic_font)

            # Adding a title and a subtitle
            plt.text(x = 0.02, y = 1.1, s = "Curve Fitting with Simple Math Functions",
                        fontsize = 30, weight = 'bold', alpha = .75,transform=ax.transAxes,
                        fontproperties=heading_font)
            plt.text(x = 0.02, y = 1.05,
                        s = "Logistic and Exponential Function fitted with real data from "+state,
                        fontsize = 22, alpha = .85,transform=ax.transAxes, 
                            fontproperties=subtitle_font)
            ax.legend(frameon=False,prop=comic_font,fontsize=26)
            ax.grid(True, linestyle='--', linewidth='2', color='white',alpha=0.2)
            
            fig.tight_layout()

            #save figs
            strFile ='./results/coronaPythonModelEN'+state+'.png'
            savePlot(strFile)

            if show:
                plt.show() 
                plt.close()

    if opt==3 or opt==0 or opt==4:

        plt.rcParams['figure.figsize'] = [12, 9]
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

        if state==state5:
            casesGrowth=cases5
            timeGrowth=time5
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

        plt.rcParams['figure.figsize'] = [12, 9]
        plt.rc('font', size=14)
        #ax.figure(figsize=(19.20,10.80))
        color_bg = '#FEF1E5'
        # lighter_highlight = '#FAE6E1'
        darker_highlight = '#FBEADC'
        colors = cm.rainbow(np.asfarray(growth,float) / float(max(np.asfarray(growth,float))))
        
        with plt.xkcd():
        
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

            #fonts for the thicks
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontproperties(comic_font)
                label.set_fontsize(16) # Size here overrides font_prop

            #Plot bars
            plt.bar(ind, bars, color=colors)
            plt.xlabel('Days since 100th case', fontproperties=comic_font)

            # Make the y-axis label and tick labels match the line color.
            plt.ylabel(state+' growth official cases per day [%]', fontproperties=comic_font)

            #Plot a line
            plt.axhline(y=10,color='r',linestyle='--')

            plt.annotate("doubling each 10 days", # this is the text
                (75,10), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(10,5), # distance from text to points (x,y)
                ha='right', weight='bold',fontsize=18,fontproperties=comic_font) 
                # horizontal alignment can be left, right or center

            plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS',fontproperties=subtitle_font,fontsize = 16,
                    xy=(1.24, 0.1), xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    ha='right',rotation=90)
            plt.annotate('Source: https://data.brasil.io',fontproperties=subtitle_font,fontsize = 16, 
                    xy=(1.24,0.1), xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    ha='left',rotation=90)
            
            
            # Hide grid lines
            ax.grid(True, linestyle='--', linewidth='2', color='white',alpha=0.2)

            # Adding a title and a subtitle
            plt.text(x = 0.02, y = 1.1, s = "Relative Growth per Day",
                        fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                        fontproperties=heading_font)
            plt.text(x = 0.02, y = 1.05,
                        s = "Real Data for "+state,
                        fontsize = 26, alpha = .85,transform=ax.transAxes, 
                            fontproperties=subtitle_font)
            fig.tight_layout()

            #save figs
            strFile ='./results/coronaPythonGrowthEN_'+state+'.png'
            fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())

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
#         bars = growth
        bars = [x / 1000 for x in growth]
        growth = bars

        plt.rcParams['figure.figsize'] = [12, 9]
        plt.rc('font', size=14)
        #ax.figure(figsize=(19.20,10.80))
        color_bg = '#FEF1E5'
        # lighter_highlight = '#FAE6E1'
        darker_highlight = '#FBEADC'
        colors = cm.rainbow(np.asfarray(growth,float) / float(max(np.asfarray(growth,float))))

        with plt.xkcd():
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
            plt.xlabel('Days since 100th case', fontproperties=comic_font)

            # Make the y-axis label and tick labels match the line color.
            plt.ylabel(state+' growth official cases per day [*1000]', fontproperties=comic_font)

            plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS',fontproperties=subtitle_font,fontsize = 16,
                    xy=(1.24, 0.1), xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    ha='right',rotation=90)
            plt.annotate('Source: https://data.brasil.io',fontproperties=subtitle_font,fontsize = 16,
                    xy=(1.24,0.1), xycoords='axes fraction',
                    xytext=(0, 0), textcoords='offset points',
                    ha='left',rotation=90)

            # Hide grid lines
            ax.grid(True, linestyle='--', linewidth='2', color='white',alpha=0.2)

            # Adding a title and a subtitle
            plt.text(x = 0.02, y = 1.1, s = "Absolute Growth per Day",
                        fontsize = 34, weight = 'bold', alpha = .75,transform=ax.transAxes,
                        fontproperties=heading_font)
            plt.text(x = 0.02, y = 1.05,
                        s = "Real Data for "+state,
                        fontsize = 26, alpha = .85,transform=ax.transAxes, 
                            fontproperties=subtitle_font)
            fig.tight_layout()

            #save figs
            strFile ='./results/coronaPythonGrowthDeltaCasesEN_'+state+'.png'
            fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())

            if show:
                plt.show() 
                plt.close()
        
    if opt==5 or opt==0:
        df = loadDataFrame('./data/SEAIRD_'+state+version+'.pkl')
        
        death = load_dead(state,startdate)
        actual = load_confirmed(state, startdate)
        extended_actual = np.int32(actual.values*(1-ratio))-np.int32(death.values)
        extended_death = np.int32(death.values)
        
        new_index = extend_index(df.index, predict_range)
        df = df[df.index<=datetime.strptime(maxDate, '%Y-%m-%d')]

        color_bg = '#FEF1E5'
        # lighter_highlight = '#FAE6E1'
        darker_highlight = '#FBEADC'
#         plt.rc('font', size=14)
        
        with plt.xkcd():        
            fig, ax = plt.subplots(figsize=(15, 10),facecolor=color_bg)
            ax.patch.set_facecolor(darker_highlight)
            # Hide the right and top spines
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            #fonts for the thicks
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontproperties(comic_font)
                label.set_fontsize(16) # Size here overrides font_prop

            # Adding a title and a subtitle
            plt.text(x = 0.02, y = 1.1, s = "SEAIR-D Model for "+state+" Brazil State",
                        fontsize = 30, weight = 'bold', alpha = .75,transform=ax.transAxes, 
                        fontproperties=heading_font)
            plt.text(x = 0.02, y = 1.05,
                        s = "Optimization fitted with real data",
                        fontsize = 26, alpha = .85,transform=ax.transAxes, 
                        fontproperties=subtitle_font)

            #limits for plotting
            ax.set_ylim((0, max(df.iloc[:]['susceptible'])*1.1))

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
            ax.legend(frameon=False,prop=comic_font,fontsize=20)
            ax.grid(True, linestyle='--', linewidth='2', color='white',alpha=0.4)

            #plot margin annotation
            plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS', 
            xy=(1.04, 0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90,fontproperties=subtitle_font,fontsize = 16)
            plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', 
            xy=(1.045,0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='left',rotation=90,fontproperties=subtitle_font,fontsize = 16)
            
            #labels for x and y axis
#             plt.xlabel("Date", fontproperties=comic_font)
#             plt.ylabel("Number of People", fontproperties=comic_font)

            #plot layout
            fig.tight_layout()

            #file name to be saved
            strFile ="./results/modelSEAIRDOpt"+state+version+model+".png"

            #remove previous file
            if os.path.isfile(strFile):
                os.remove(strFile)   # Opt.: os.system("del "+strFile)

            #figure save and close
            fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
            if show:
                plt.show()
                plt.close()

        #format background

        with plt.xkcd():        
            fig, ax = plt.subplots(figsize=(15, 10),facecolor=color_bg)
            ax.patch.set_facecolor(darker_highlight)

            # Hide the right and top spines
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            #fonts for the thicks
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontproperties(comic_font)
                label.set_fontsize(16) # Size here overrides font_prop

            ax.set_ylim(0,max(max(df['infected']),max(np.int32(extended_actual)))*1.1)

            # Adding a title and a subtitle
            plt.text(x = 0.02, y = 1.1, s = "Zoom SEAIR-D Model for "+state+" Brazil State",
                        fontsize = 30, weight = 'bold', alpha = .75,transform=ax.transAxes,
                        fontproperties=heading_font)
            plt.text(x = 0.02, y = 1.05,
                        s = "Optimization fitted with real data",
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
            ax.legend(frameon=False,prop=comic_font,fontsize=20)
            ax.grid(True, linestyle='--', linewidth='2', color='white',alpha=0.2)

            plt.annotate('Modeling Team for Sao Paulo State IPT, USP, ATS', 
            xy=(1.04, 0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='right',rotation=90,fontproperties=subtitle_font,fontsize = 16)
            plt.annotate('Original SEAIR-D with delay model, São Paulo, Brazil', 
            xy=(1.045,0.1), xycoords='axes fraction',
            xytext=(0, 0), textcoords='offset points',
            ha='left',rotation=90,fontproperties=subtitle_font,fontsize = 16)
 
            #labels for x and y axis
#             plt.xlabel("Date", fontproperties=comic_font)
#             plt.ylabel("Number of People", fontproperties=comic_font)

            #plot layout
            fig.tight_layout()

            #file name to be saved
            strFile ="./results/ZoomModelSEAIRDOpt"+state+version+model+".png"

            #remove previous file
            if os.path.isfile(strFile):
                os.remove(strFile)   # Opt.: os.system("del "+strFile)

            #figure save and close
            fig.savefig(strFile, facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
            if show:
                plt.show()
                plt.close()