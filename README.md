# dataAndModelsCovid19

## Installation

Clone this repository

```
git clone https://github.com/gasilva/dataAndModelsCovid19.git
```

Or use GitHub Desktop [https://desktop.github.com/](https://desktop.github.com/) and File, Clone repository

## Run Code

If you are using Visual Code Studio, select the folder you are running.

To run SIR-D model with opt=5:

```
python dataFit_SIRD.py
```
To run SEIR model with opt=5:

```
python dataFit_SEIR.py
```

To run SEIR-D model with opt=5:

```
python dataFit_SEIRD.py
```

## Usage

Make changes directly in the dataAndModelsCovid19.py file.

Select one option 1 to 5 by variable opt

```
#Initial parameters
#Choose here your options

#option
#opt=0 all plots
#opt=1 corona log plot
#opt=2 logistic model prediction
#opt=3 bar plot with growth rate
#opt=4 log plot + bar plot
#opt=5 SIR-D Model
opt=0
```

Select countries to be plotted in log to analyze growth rate. All countries available.

```
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

```
Choose version to be place in the .png file name of log plot. This allows to you to analyze different set of countries.

```
#plot version - changes the file name png
version="2"
```


Select country to have the exponential and logistic function fitting. Choose one of the countries in the list above.

```
#choose country for curve fitting
#choose country for growth curve
#one of countries above
country="Brazil"

```

Choose country to analyze data by SIRD model. Some countries are already adjusted. Other countries may need extra work to adjust S_0, I_0, R_0 and K_0, i.e., the initial conditions. Plus it may required you to set the start date correctly. If a delay/lag exist in the recovery or dead data, it may be required to set negative values for R_0 and K_0. It may simulate the lag.

```
#choose country for SIRD model
# "Brazil"
# "China"
# "Italy"
# "France"
# "United Kingdom"
# "US"
# Countries above are already adjusted
countrySIRD="Brazil"
```

## Command line use for SIRD model

This implementation comes from SIR model of [https://github.com/Lewuathe/COVID19-SIR](https://github.com/Lewuathe/COVID19-SIR)

It was added the K_0 value because SIRD model has initial deaths value in addition to S_0, I_0 and R_0.

You can analayze several countries by making a CSV list like: Brazil,Italy,US,France. Do not put spaces before or after commas.

```
 For other countries you can run at command line
 but be sure to define S_0, I_0, R_0, K_0
 the sucess of fitting will depend on these paramenters

 usage: dataAndModelsCovid19.py [-h] [--countries COUNTRY_CSV] [--download-data]
                  [--start-date START_DATE] [--prediction-days PREDICT_RANGE]
                  [--S_0 S_0] [--I_0 I_0] [--R_0 R_0]

 optional arguments:
   -h, --help            show this help message and exit
   --countries COUNTRY_CSV
                         Countries on CSV format. It must exact match the data
                         names or you will get out of bonds error.
   --download-data       Download fresh data and then run
   --start-date START_DATE
                         Start date on MM/DD/YY format ... I know ...It
                         defaults to first data available 1/22/20
   --prediction-days PREDICT_RANGE
                         Days to predict with the model. Defaults to 150
   --S_0 S_0             S_0. Defaults to 100000
   --I_0 I_0             I_0. Defaults to 2
   --R_0 R_0             R_0. Defaults to 0
```

## Databases Used in This Study
 
### Data

This code has data from Repository by Johns Hopkins CSSE

https://github.com/CSSEGISandData/COVID-19

### Data Analysis

The log-curve of exponential growth in several countries is based on

https://www.ft.com/content/ae83040c-18e6-440e-bd00-e4c5cefdda26

The growth rate is original from Guilherme A. L. da Silva - http://www.at4i.com

### Curve Fitting

The exponential and logistic curves fitting is based on

https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d

## Theory

### Mathematical Models

The matematical Models are based in Lotka-Volterra equations, it is like a predator-prey type of model.

A simple mathematical description of the spread of a disease in a population is the so-called SIR model, which divides the (fixed) population of N individuals into three "compartments" which may vary as a function of time, t:

- S(t) are those susceptible but not yet infected with the disease;
- I(t) is the number of infectious individuals;
- R(t) are those individuals who have recovered from the disease and now have immunity to it.

The SIR model describes the change in the population of each of these compartments in terms of two parameters, β and γ. β describes the effective contact rate of the disease: an infected individual comes into contact with βN other individuals per unit time (of which the fraction that are susceptible to contracting the disease is S/N). γ is the mean recovery rate: that is, 1/γ is the mean period of time during which an infected individual can pass it on.

The differential equations describing this model were first derived by Kermack and McKendrick [Proc. R. Soc. A, 115, 772 (1927)]:

#### SIR - Suscetible, Infected and Recovered Model

![](https://user-images.githubusercontent.com/7212952/77828643-775e6a80-70fb-11ea-8428-73d7a2f176a8.gif)

Here, the number of 'recovery' englobes both recovered and deaths. This parameter is represented by γ.

The SIR model code is based on

https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html

https://github.com/Lewuathe/COVID19-SIR

https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model

https://triplebyte.com/blog/modeling-infectious-diseases

The γ is split in two by γ = a + b, where a is the rate of recoveries, and b is the rate of death. Since the death rate seems to be linear (1.5% in China, for example), this linear decomposition of γ is precise enough. 

So we can add a new variable k, (Kill rate), and add to the system of equations. Therefore:

#### SIR-D - SIR model extended to have deaths and recovered separated

![](https://user-images.githubusercontent.com/7212952/77828649-7f1e0f00-70fb-11ea-8b59-d7f722305847.gif)

The SIR-D model code is based on the contribution of Giuliano Belinassi, from IME-USP, Brazil

https://github.com/Lewuathe/COVID19-SIR/issues/13#issue-589616803

The the Pyhton notebook of

https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model#Scenario-in-Italy

### References:

William Ogilvy Kermack, A. G. McKendrick and Gilbert Thomas Walker 1997A contribution to the mathematical theory of epidemicsProc. R. Soc. Lond. A115700–721 https://doi.org/10.1098/rspa.1927.0118


