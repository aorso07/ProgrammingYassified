#HW4

install.packages('mosaicData')
library(mosaicData)
library(tidyverse)

#1a
mns = HELPrct %>%
  summarize(across(where(is.numeric), mean, na.rm = TRUE))

mns
#1b

cuts = seq(0,100,by=10)

strat = HELPrct %>%
  mutate(age_cut = as.factor(cut(age, breaks = cuts, labels = c("0","1","2","3","4","5","6","7","8",'9'),right=FALSE)))%>%
  group_by(age_cut,sex) %>%
  summarize(across(where(is.numeric), mean, na.rm = TRUE))

strat


#1c


strat = as.data.frame(strat)


viz = function(y){
  strat %>%
    ggplot(aes_string(x="age",y=y))+
    geom_line()+
    facet_wrap(~sex)
}

col = colnames(strat)

cols = col[4:length(col)]

cols %>%
  map(viz)

#Q2


library(Lahman)

bk_teams <- c("BR1", "BR2", "BR3", "BR4", "BRO", "BRP", "BRF")


count_seasons=function(team_name){
  Teams %>%
    filter(teamID==team_name)%>%
    summarize(num_years=n())
  }

bk_teams %>%
  map(count_seasons)


 
