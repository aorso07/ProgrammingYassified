library(mdsr)
#install.packages('macleish')
library(macleish)
library(lubridate)
library(tidyverse)
library(plotly) 

head(whately_2015)

data = whately_2015


data$date_only=substr(data$when,1,10)

data$date_only=as.Date(data$date_only)

data$day = wday(data$date_only,label=TRUE)
data$month = month(data$when,label=TRUE)


head(data)

for_viz<- data %>%
  group_by(day,month) %>%
  summarize(high_temp = max(temperature),min_temp=min(temperature),avg_temp=mean(temperature))%>%
  as.data.frame()

for_viz

temp_plot=ggplot(data=for_viz,aes(day,group=1))+
  geom_line(aes(y=high_temp,color="high_temp"))+
  geom_line(aes(y=min_temp,color="min_temp"))+
  geom_line(aes(y=avg_temp, color= "avg_temp"))+
  facet_wrap(~month)

final_plot = ggplotly(temp_plot) 
final_plot
