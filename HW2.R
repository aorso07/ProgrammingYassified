library("dplyr")
library("tidyr")
#install.packages("Lahman")
require("Lahman")
library(tidyverse)
library(mdsr)
install.packages("mdsr")
library(ggplot2)
data(Teams)
install.packages("RColorBrewer")
library(RColorBrewer)

#Q1

summary(Teams)

team_filter = Teams %>%
  filter(Teams$yearID %in% c(2000:2009)) %>%
  select(yearID,W,L)


####

newer = select(filter(Teams, Teams$yearID >= 2000 & Teams$yearID <= 2009),yearID,W,L)
class(newer)
head(newer)

###

?Teams

cubs = Teams %>%
  filter(teamID=='CHN' & HR >200) %>%
  summarize(NumYears= n(),
            MedianWins=median(W))

cubs

###

prez_term = c(1789:2021)

index=c(rep(1:58,each=4),59)

index

prez4factor=data.frame(prez_term,index)

colnames(prez4factor) = c("Year","TermNo")


joined = Teams %>%
  inner_join(prez4factor,c("yearID" = "Year"))

joined$TermNo = as.factor(joined$TermNo)


term_runs=joined %>%
  group_by(TermNo) %>%
  summarize(runs=sum(HRA+HR))


term_runs[which.max(term_runs$runs),]



###
is.na(Teams$lgID)


yes_league=Teams%>%
  filter(lgID != "NA")

summary(yes_league$lgID)


all_runs= yes_league %>%
  group_by(yearID,lgID) %>%
  summarize(total_runs=sum(HRA+HR))

tail(all_runs)

summary(Teams$lgID)

Teams[Teams$lgID=="AA",]

line_plot = ggplot()+
  geom_line(data=all_runs, aes(x=yearID,y=total_runs))+
  facet_wrap(~lgID,nrow=6)+
  ylab("Total Runs by League")+
  xlab("Year")

line_plot
###

win_record = Teams %>%
  mutate(Winning_Record = case_when(
    W > L ~ TRUE,
    L >= W ~ FALSE
  )%>%
    as.numeric() %>%
    as.factor()
  )

summary(win_record$Winning_Record)
class(win_record$Winning_Record)

win_plot=ggplot()+
  geom_point(data=win_record, aes(x=R, y=RA, col=Winning_Record))+
  labs(x="Runs",y="Runs Against")
  
win_plot
###

?write.csv

setwd("~/Documents")
write.csv(Teams,file="teams.csv")


#Q2




install.packages("nycflights13")
require("nycflights13")
data(flights)
class(flights)

data(flights)

write.csv(flights,"flights.csv")

fly = as.data.frame(flights)
class(fly)
summary(fly)

fly[c("dest","origin","carrier","tailnum")]

cols=c("dest","origin","carrier","tailnum")

fly[,cols]=lapply(fly[,cols], factor)
summary(fly)
class(fly)

summary(flights)



sapply(fly, function(y) sum(length(which(is.na(y)))))

?sapply

library(tidyverse)


cancel = fly %>%
  mutate(cancelled=case_when(
    !is.na(dep_time) ~ 0,
    TRUE ~ 1) 
  )


cancel[cancel$cancelled==1,]

summary(cancel$cancelled)

prop_cancel = cancel %>%
  group_by(month)%>%
  summarize(cancel_prop=sum(cancelled)/n())

prop_cancel

##

delayed = fly %>%
  filter(dep_delay > 60) %>%
  group_by(origin)%>%
  summarize(avg_delay = sum(dep_delay)/n())

delayed

##

fly_time = fly %>%
  filter(!is.na(air_time))%>%
  group_by(carrier)%>%
  summarize(avg_fly_time=sum(air_time)/n())


fly_time
fly_time[which.max(fly_time$avg_fly_time),]

##

delay0 = fly%>%
  filter(dep_delay > 0,!is.na(dep_delay),!is.na(month)) %>%
  ggplot(.,aes(x=dep_delay,col=month))+
  geom_histogram(bins=30)+
  facet_wrap(~month,nrow=4)+
  xlim(0,700)


delay0

##


delaybox = fly %>%
  filter(dep_delay>60,!is.na(dep_delay))%>%
  group_by(carrier)%>%
  summarize(carrier_count=n())%>%
  arrange(desc(carrier_count))

head(delaybox[,1],5)
top5=c("EV","B6","UA","DL","AA")

delay_final = fly %>%
  filter(dep_delay>60,!is.na(dep_delay),carrier %in% top5)%>%
  group_by(carrier)%>%
  ggplot()+
  geom_boxplot(aes(x=dep_delay, col=carrier),size=0.5)+
  facet_wrap(~carrier)+
  xlab("Delay Time (in mins)")+
  scale_color_brewer(palette="Spectral")


delay_final




write.csv(flights,file="flights_final.csv")
dim(flights)
