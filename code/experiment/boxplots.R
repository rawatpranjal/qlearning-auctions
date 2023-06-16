library(gtable)
library(gridExtra)
library(tidyverse)
library(xtable)
library(stargazer)
library(experiment)
library(FSM)
library(allestimates)
library(sensemakr)

# Load the data
setwd("/Users/pranjal/Desktop/AlgorithmicCompetition/code/experiment")
df <- read_csv("main_experiment_data.csv")
df$design <- factor(df$design)
# Create three sample graphs
df$design <- factor(df$design)

# Create the first boxplot
graph1 <- ggplot(df, aes(x = design, y = bid2val, fill = design)) + geom_boxplot()  + ylab("bid2val") + coord_flip()+guides(fill = guide_legend(title = NULL)) + theme(legend.position = "none")
graph2 <- ggplot(df, aes(x = design, y = volatility, fill = design)) + geom_boxplot() + ylab("volatility") + coord_flip()+guides(fill = guide_legend(title = NULL)) + theme(legend.position = "none")
graph3 <- ggplot(df, aes(x = design, y = episodes, fill = design)) + geom_boxplot() + ylab("episodes") + coord_flip()+guides(fill = guide_legend(title = NULL)) + theme(legend.position = "none")
stacked_graph <- grid.arrange(graph1, graph2, graph3, ncol = 1)
ggsave("boxplot_stacked.png", stacked_graph, width = 10, height = 6, units = "in")
