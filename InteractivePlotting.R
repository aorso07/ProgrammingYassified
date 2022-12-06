install.packages('plotly')
library(plotly) 
library(tidyverse)


library(plotly)
data(iris)
fig <- plot_ly(data = iris, x = ~Sepal.Width, y = ~Sepal.Length, color = ~Species, 
               type = "scatter", mode = "markers")%>%
  layout(title="A Plotly Figure", legend=list(title=list(text='species')),
         plot_bgcolor='#e5ecf6', 
         xaxis = list( 
           zerolinecolor = '#ffff', 
           zerolinewidth = 2, 
           gridcolor = 'ffff'), 
         yaxis = list( 
           zerolinecolor = '#ffff', 
           zerolinewidth = 2, 
           gridcolor = 'ffff'))
fig


#Scatterplots

library(plotly)
fig <- plot_ly(data = iris, x = ~Sepal.Length, y = ~Petal.Length)
fig

#Plotting markers and lines

library(plotly)
trace_0 <- rnorm(100, mean = 5)
trace_1 <- rnorm(100, mean = 0)
trace_2 <- rnorm(100, mean = -5)
x <- c(1:100)
data <- data.frame(x, trace_0, trace_1, trace_2)
fig <- plot_ly(data, x = ~x)
fig <- fig %>% add_trace(y = ~trace_0, name = 'trace 0',mode = 'lines')
fig <- fig %>% add_trace(y = ~trace_1, name = 'trace 1', mode = 'lines+markers')
fig <- fig %>% add_trace(y = ~trace_2, name = 'trace 2', mode = 'markers')
fig


#Plotly works with ggplot!


library(tidyverse)
library(mdsr)
library(babynames)
Beatles <- babynames %>%
  filter(name %in% c("John", "Paul", "George", "Ringo") & sex == "M") %>%
  mutate(name = factor(name, levels = c("John", "George", "Paul", "Ringo")))
beatles_plot <- ggplot(data = Beatles, aes(x = year, y = n)) +
  geom_line(aes(color = name), size = 2)
beatles_plot


library(plotly)
ggplotly(beatles_plot)


#We can also do interactive tables!
#With datatable!

library(DT)
datatable(Beatles, options = list(pageLength = 10))


#Dygraphs

install.packages('dygraphs')
library(dygraphs)
a = Beatles %>% 
  filter(sex == "M") %>% 
  select(year, name, prop) %>%
  pivot_wider(names_from = name, values_from = prop)

class(a$year)


#In class exercises


setwd("Documents")
ins <- read.csv("inspections_clean.csv")
#Make an interactive plot that shows pass rate by month with inspection type represented by color.  
#Repeat this for a dygraph

library(lubridate)
library(tidyverse)
library(xts)
ins_graph = ins %>%
  mutate(month = month(inspection_date),year=year(inspection_date))%>%
  mutate(full_date = as.Date(paste(year, month, "01", sep = '-')))%>%
  group_by(full_date,inspection_type)%>%
  summarize(pass_rate = sum(results == 'Pass')/n())%>%
  pivot_wider(names_from = inspection_type, values_from = pass_rate)

don = xts(x = ins_graph[,-1], order.by = ins_graph$full_date)

viz = dygraph(don)

viz
  
  


