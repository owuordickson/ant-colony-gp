# Title     : TODO
# Objective : TODO
# Created by: owuordickson
# Created on: 8/27/19

library(dplyr)

# data <- read.csv(file="~owuordickson/PyCharm/ant-colony-gp/R/steps_vs_solns.csv", sep=" ", header=TRUE)
# data <- read.table(file="~owuordickson/PyCharm/ant-colony-gp/results/steps_vs_solns.csv",header=TRUE, sep=" ", colClasses = c(rep("NULL", 3), rep("numeric", 3)))
data <- read.table(file="~owuordickson/PyCharm/ant-colony-gp/results//attr_vs_patterns.csv",header=TRUE, sep=" ", colClasses = c(rep("NULL", 1), rep("numeric", 2)))
data
s <- sapply(data[10:12,1:2], function(x) c(#"n" = length(x), 
                         #"Minimum" = min(x),
                         #"Maximun" = max(x),
                         "Mean"= mean(x,na.rm=TRUE),
                         "Stand dev" = sd(x),
                         #"Median" = median(x),
                         "CoeffofVariation" = sd(x)/mean(x,na.rm=TRUE)
                         #"Upper Quantile" = quantile(x,1),
                         #"LowerQuartile" = quantile(x,0)
                         )
       )
s
#for (row in seq(1,nrow(data),3))
#  step <- data[row, "T"]
  #N <- data[row, "N"]
  #for (i in row:(row+2))
   # soln <- data[i, "Solutions"]
   # win <- data[i, "Winners"]
   # time <- data[i, "Runtime"]
  #soln
library(xtable)
print(xtable(s), file="~owuordickson/PyCharm/ant-colony-gp/results//temp.tex")

