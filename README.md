# dataAndModelsCovid19

## Python code to analyze data and predict Covid-19 infection
 
### Data

This code has data from Repository by Johns Hopkins CSSE

https://github.com/CSSEGISandData/COVID-19

### Data Analysis and Curve Fitting

The log-curve of exponential growth in several countries is based on

https://www.ft.com/content/ae83040c-18e6-440e-bd00-e4c5cefdda26

The growth rate is original from Guilherme A. L. da Silva - http://www.at4i.com

The exponential and logistic curves fitting is based on

https://towardsdatascience.com/covid-19-infection-in-italy-mathematical-models-and-predictions-7784b4d7dd8d

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

#### SIR-D - SIR model extended to have deaths and recovered separated

The γ is split in two by γ = a + b, where a is the rate of recoveries, and b is the rate of death. Since the death rate seems to be linear (1.5% in China, for example), this linear decomposition of γ is precise enough. 

So we can add a new variable k, (Kill rate), and add to the system of equations. Therefore:

![](https://user-images.githubusercontent.com/7212952/77828649-7f1e0f00-70fb-11ea-8b59-d7f722305847.gif)

The SIR-D model code is based on the contribution of Giuliano Belinassi, from IME-USP, Brazil

https://github.com/Lewuathe/COVID19-SIR/issues/13#issue-589616803

The the Pyhton notebook of

https://www.kaggle.com/lisphilar/covid-19-data-with-sir-model#Scenario-in-Italy

###References:

William Ogilvy Kermack, A. G. McKendrick and Gilbert Thomas Walker 1997A contribution to the mathematical theory of epidemicsProc. R. Soc. Lond. A115700–721 https://doi.org/10.1098/rspa.1927.0118


