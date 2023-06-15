# Load the required packages
install.packages("ggplot2")
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

# subset
df_outcomes <- df[, c("bid2val", "episodes","bid2val_std","bid2val_min", "bid2val_max")]
df_covariates <- df[, c("N", "alpha", "gamma", "egreedy", "design", "asynchronous", "feedback", "num_actions", "decay")]

# export summary statistics to latex
stargazer(as.data.frame(df_outcomes))
stargazer(as.data.frame(df_covariates))
stargazer(as.data.frame(df))

# Create three sample graphs
df$design <- factor(df$design)

# Create the first boxplot
graph1 <- ggplot(df, aes(x = design, y = bid2val, fill = design)) + geom_boxplot() + xlab(NULL) + ylab("bid2val") + coord_flip()+guides(fill = guide_legend(title = NULL)) + scale_x_discrete(labels = NULL) + theme(legend.position = "none")
graph2 <- ggplot(df, aes(x = design, y = volatility, fill = design)) + geom_boxplot() + xlab(NULL) + ylab("volatility") + coord_flip()+guides(fill = guide_legend(title = NULL)) + scale_x_discrete(labels = NULL) + theme(legend.position = "none")
graph3 <- ggplot(df, aes(x = design, y = episodes, fill = design)) + geom_boxplot() + xlab(NULL) + ylab("episodes") + coord_flip()+guides(fill = guide_legend(title = NULL)) + scale_x_discrete(labels = NULL) + theme(legend.position = "none")
stacked_graph <- grid.arrange(graph1, graph2, graph3, ncol = 1)
ggsave("boxplot_stacked.png", stacked_graph, width = 10, height = 6, units = "in")






# Rotate the box plot
p + coord_flip()
# Notched box plot
ggplot(ToothGrowth, aes(x=dose, y=len)) + 
  geom_boxplot(notch=TRUE)
# Change outlier, color, shape and size
ggplot(ToothGrowth, aes(x=dose, y=len)) + 
  geom_boxplot(outlier.colour="red", outlier.shape=8,
                outlier.size=4)

# Neyman difference-in-means estimator and variance. 
ATEnocov(df$bid2val,df$design,data=df)


# Create an empty data frame to store the selected elements
selectedData <- data.frame(Variable = character(),
                           Element6 = numeric(),
                           Element7 = numeric(),
                           stringsAsFactors = FALSE)



# fisher sharp null
perm_test(df$bid2val,df$design,df(,c("bid2val","design")))


# all possible regressions
vlist <- c("N", "alpha", "gamma", "egreedy", "design", "asynchronous", "feedback", "num_actions", "decay")
all_results <- all_lm(crude = "bid2val ~ design", xlist = vlist, data = df)


library(sensemakr)
model <- lm(bid2val ~ N + alpha + egreedy + asynchronous + design + feedback + num_actions + decay , data = df)
model <- lm(bid2val ~ design, data = df)

partial_r2(model)