library(forecast)
library(splines)


'''
load and prepare data 
Data:
Demand- hourly domestic electricity demand (exclusive of solar)
in kilowatt hours for Melbourne metropolitan area for July 1st 2016 to June 30 2017.
Temp - hourly temperature readings for same period
Hour - hour of day from 0 (midnight) to 23 (11pm)
Date - date
Weekday - "WEEKEND" if day is Saturday or Sunday, "WEEKDAY" otherwise 
Other weather variables not used in this analysis
'''

CitiPower <- read.csv("CitiPower.csv")
CitiPower$Date <- as.Date(CitiPower$Date,format='%d/%m/%Y')
CitiPower$Hour <- as.factor(CitiPower$Hour)
CitiPower$Weekend <- as.factor(CitiPower$Weekend)


dem<-ts(CitiPower$Demand,frequency=24)

'''
divide demand into training and test sets
'''
h <- 24
train_size=length(dem)*0.9/24
train <- window(dem,end=train_size-.01)
train_x <- window(temp,end=train_size)
test <- window(dem,start=train_size-.01)
test_x <- window(temp,start=train_size)


'''
fit seasonal ARIMA model on training set
'''

fit <- auto.arima(train)
summary(fit)
checkresiduals(fit)

'''
Get the forecasts on the test set for up to 24 hours ahead
'''
n <- length(test) - h + 1

#matrix containing forecasts for up to 24 hours from every hour in the test set

fc <- ts(matrix(0,n,24), start=train_size+(h-1)/24,freq=24) 

'''
Rolling forecast for 24 hours ahead for every observation in test set
'''
for(i in 1:n)
{  
  x <- window(dem, end=train_size-0.01 + (i-1)/24) #all observations up to now
  refit <- Arima(x, model=fit) 
  forecast <-forecast(refit, h=h) #forecast next 24 hours
  fc[i,] <- forecast$mean #add current forecast to matrix 
  
}

write.csv(fc,'work/pred_arima',row.names = F)

'''
plot actual vs forecasted demand for 1 and 24 hours ahead
'''
 

'''
Calculate MAPE (mean absolute percentage error) for forecasts on test set forom 1 to 24 hours ahead
'''
mapes<-numeric(24)
for (i in 1:24)
{
  preds<-fc[1:(dim(fc)[1]-1),i] #predicted demand for observations in test set
  obs<-test[(i):(length(test)-25+i)] #observed demand for same period
  ape<-abs(preds-obs)/obs #absolute percentage error
  mapes[i]=mean(ape)*100 

}

write.csv(mapes,'work/arima',row.names=F)

'''
Now include temperature as a coregressor in the ARIMA model-
since the relationship  between demand and temperature is non-linear, use spline regression
with knots at the tertiles of the temperature distribution 
-number and position of knots selected by  AIC  
'''

knots =quantile(CitiPower$Temp,probs=c(0.333,0.667))

#temp as basis splines - divide into train and test
temp = ts(bs(CitiPower$Temp,knots=knots),frequency=24)
train_x <- window(temp,end=train_size-.01)
test <- window(dem,start=train_size)
test_x <- window(temp,start=train_size)



'''
fit seasonal ARIMA model with coreggressors
'''

fit_x <- auto.arima(train,xreg=train_x)
summary(fit_x)
checkresiduals(fit_x)

'''
Plot spline function - the implied relationship between temperature and demand
'''
t_x<-seq(min(CitiPower$Temp),max(CitiPower$Temp),length.out=10000)
y<-bs(t_x,knots=knots)
z = y%*%as.numeric(fit_x$coef[6:10])
plot(t_x,z)



'''
Rolling forecast for test set with new ARIMAX model with temps
'''
n <- length(test) - 23
fc_x <- ts(matrix(0,n,24), start=train_size+(h-1)/24,freq=24)

for(i in 1:n)
{  
  d <- window(dem, end=train_size-.01 + (i-1)/24)
  x <-window(temp,end=train_size-.01+(i-1)/24)
  refit <- Arima(d,xreg=x, model=fit_x)
  x_n = window(temp,start=train_size-.01+(i-1)/24,end=train_size+.99+(i-1)/24)
  forecast <-forecast(refit,xreg=x_n)
  fc_x[i,] <- forecast$mean
  print(i)
}

write.csv(fc_x,'work/arimax',row.names = F)

'''
Get MAPE on test set for up to 24 hour ahead predictions.
'''
mapes_x<-numeric(24)
for (i in 1:24)
{
  a<-fc_x[1:(dim(fc_x)[1]-1),i]
  b<-test[(i):(length(test)-25+i)]
  e<-abs(a-b)/b
  mapes_x[i]=mean(e)*100
  plot(a[1:100])
  lines(b[1:100])
}

write.csv(mapes+x,'work/APE_arimax',row.names=F)



