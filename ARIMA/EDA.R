library(ggplot2)
library(forecast)

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
colnames(CitiPower) <- c('Date','Month','Hour','Temp','DewTemp','Wind','COP','Weekend','Demand')
CitiPower$Date <- as.Date(CitiPower$Date,format='%d/%m/%Y')
CitiPower$Hour <- as.factor(CitiPower$Hour)
CitiPower$Weekend <- as.factor(CitiPower$Weekend)


'''
Plot the hourly demand for June  of 2017
'''
CitiPower.June2017<-subset(CitiPower,Date>as.Date("2017-06-01"))


dev.copy(png,'images/June.Demand.png')
plot(CitiPower.June2017$Demand,type='l',col=2,
     xlab='Date',ylab='Demand (kW)',, xaxt='n',
     main='Hourly Demand for June 2017')
axis(1, at=120*(1:6)-12, labels=c('5/6','10/6','15/6','20/6','25/6','30/6'))
dev.off()    

#Scatterplot for temperature and demand (with LOESS smoother)


ggplot(CitiPower, aes(x = Temp, y = Demand)) + 
  geom_point(size=0.3,color='blue') + 
  geom_smooth(method='loess',color='#C47FE7')+
  ylab('Demand (kW)')+ 
  xlab('Temp (C)') +
  theme_classic()
ggsave('images/temp_vs_dem.png')


#Look at relationships between demand, temperatature and time of day

dev.copy(png,'images/pairplots.png')
pairs(CitiPower[c(3,4,9)],pch='.')
dev.off()

'''
boxplot for demand by time of day
'''

ggplot(CitiPower, aes(x = Hour, y = Demand)) + 
  geom_boxplot(outlier.color='#DA4323',fill=	'#7FA0E7') + 
  ylab('Demand (kW)') +
  theme_classic()


ggsave('images/hourlybox.png')

'''
Scatterplots for demand and temperature grouped by time of day, with LOESS smoother
'''

ggplot(CitiPower, aes(x = Temp, y = Demand, color=Weekday)) +
  geom_point(size=0.5) + 
  facet_wrap(~Hour,ncol=6)+
  geom_smooth() +
  xlab('Temperaure (C)')+
  ylab('Demand kW')+
  theme_bw()

ggsave('images/dem_temp_by_hour.png')

###Time series analysis

'''
Looking at the June demand graph, there is clear daily seasonality in the data, and some evidence of weekly seasonality
'''
demand <- msts(CitiPower$Demand,seasonal.periods = c(24,168),ts.frequency = 24)

'''
Decompose into multiple seasonal, trend and irregular components using loess (mstl for forecast package)
'''
dev.copy(png,'images/decompose.png')
autoplot(mstl(demand))
dev.off()

#Auto and partial auto-correlation functions for demand time series
dev.copy(png,'images/acf_pacf.png')
par(mfrow=c(2,1))
acf(demand)
pacf(demand)
dev.off()

