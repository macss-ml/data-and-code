# Clustering
# Intro to ML, pdwaggoner@uchicago.edu

# HAC
# k-means
# GMMs


#
# Hierarchical agglomerative clustering

# First, load a few libraries
library(tidyverse)
library(skimr)
library(dendextend) # for "cutree" function

# Using data from the 1977 US census statistical abstract
# store as a data frame (currently a matrix)
s <- as.data.frame(state.x77)

# take a look at the summary stats and distributions for each
skim(s)

# select a few related features, standardize, and calculate euclidean distance matrix
s_sub <- s %>% 
  select(Income, Illiteracy, `Life Exp`, `HS Grad`) %>% 
  scale() %>% 
  dist()


s_sub # inspect to make sure features are on the same scale

# Fit and viz all in a single pane
par(mfrow = c(2,2))

hc_single <- hclust(s_sub, 
                    method = "single"); plot(hc_single, hang = -1)

hc_complete <- hclust(s_sub, 
                      method = "complete"); plot(hc_complete, hang = -1)

hc_average <- hclust(s_sub, 
                     method = "average"); plot(hc_average, hang = -1)

hc_centroid <- hclust(s_sub,
                      method = "centroid"); plot(hc_centroid, hang = -1)

# reset plot space
par(mfrow = c(1,1))


# And we can cut and compare trees if we aren't sure about 3 or 4 clusters, e.g.
cuts <- cutree(hc_complete, 
               k = c(3,4))

### Inspect assignments for each iteration...
cuts

### Or, a matrix of assignments by cut
table(`3 Clusters` = cuts[,1], 
      `4 Clusters` = cuts[,2])

# what do you see?



#
## k-means and 2012 pres vote shares

# Load data and update the pres vote data for 2012
library(here)
library(amerika)

pres <- read_csv("2012_DVS.csv") %>% 
  mutate(State = X1,
         DVS = dem_vs) %>% 
  select(-c(X1, dem_vs))

head(pres)

# fit the algorithm
set.seed(634)

kmeans <- kmeans(pres[ ,2], 
                 centers = 2,
                 nstart = 15)

# Inspect the kmeans object
str(kmeans)

# Or call individual values, such as...
kmeans$cluster
kmeans$centers
kmeans$size

pres$Cluster <- as.factor(kmeans$cluster) # save clusters as factor for plotting

# Assess a little more descriptively
t <- as.table(kmeans$cluster)
(t <- data.frame(t))
rownames(t) <- pres$State
colnames(t)[colnames(t)=="Freq"] <- "Assignment"
t$Var1 <- NULL

head(t, 10)

# evaluate the distribution of states based on their cluster assignment
ggplot(pres, aes(DVS, 
                 fill = Cluster)) + 
  geom_histogram(binwidth = 3) + 
  theme_minimal() +
  scale_fill_manual(values=c(amerika_palettes$Democrat[3], 
                             amerika_palettes$Republican[3])) +
  labs(x = "Democratic Vote Share",
       y = "Count of States") +
  geom_vline(xintercept = 50, linetype="solid", 
             color = "darkgray", size=1.2)

# What do you see?


# Searching for the likely "misclassified" state
which(pres$DVS < 50 & pres$DVS > 47) 


# GMMs

# Load a few extra libraries
library(mixtools)
library(plotGMM)

# Take a look at the density
ggplot(pres, aes(x = DVS)) +
  geom_density() + 
  xlim(min(pres$DVS) - 10, 
       max(pres$DVS) + 10) +
  theme_minimal() +
  labs(x = "Democratic Vote Share")

# best guess at component means (fig from lecture)
ggplot(pres, aes(x = DVS)) +
  geom_density() + 
  xlim(min(pres$DVS) - 10, 
       max(pres$DVS) + 10) +
  theme_minimal() +
  labs(x = "Democratic Vote Share") +
  geom_vline(xintercept = 41, 
             col = amerika_palettes$Republican[3]) + 
  geom_vline(xintercept = 53, 
             col = amerika_palettes$Democrat[3])

# Start by fitting a two component (cluster) gmm
set.seed(7355) 

gmm1 <- normalmixEM(pres$DVS, k = 2) 

ggplot(data.frame(x = gmm1$x)) +
  geom_histogram(aes(x, ..density..), fill = "darkgray") +
  stat_function(geom = "line", fun = plot_mix_comps,
                args = list(gmm1$mu[1], gmm1$sigma[1], lam = gmm1$lambda[1]),
                colour = amerika_palettes$Republican[3]) +
  stat_function(geom = "line", fun = plot_mix_comps,
                args = list(gmm1$mu[2], gmm1$sigma[2], lam = gmm1$lambda[2]),
                colour = amerika_palettes$Democrat[3]) +
  xlab("Democratic Vote Shares") +
  ylab("Density") + 
  theme_minimal()


# next attempt
set.seed(7355)
gmm2 <- normalmixEM(pres$DVS, k = 3)

ggplot(data.frame(x = gmm2$x)) +
  geom_histogram(aes(x, ..density..), fill = "darkgray") +
  stat_function(geom = "line", fun = plot_mix_comps,
                args = list(gmm2$mu[1], gmm2$sigma[1], lam = gmm2$lambda[1]),
                colour = amerika_palettes$Republican[3]) +
  stat_function(geom = "line", fun = plot_mix_comps,
                args = list(gmm2$mu[2], gmm2$sigma[2], lam = gmm2$lambda[2]),
                colour = amerika_palettes$Democrat[3]) +
  stat_function(geom = "line", fun = plot_mix_comps,
                args = list(gmm2$mu[3], gmm2$sigma[3], lam = gmm2$lambda[3]),
                colour = "black") +
  xlab("Democratic Vote Shares") +
  ylab("Density") + 
  theme_minimal()


# Searching for (potentially) problematic observation, given poor fit of GMM
which(pres$DVS > 80)

# now we can try again without outlier
pres2 <- pres[-c(17), ]

# quickly compare to make sure it worked
withDC <- head(pres$DVS, 20)
withoutDC <- head(pres2$DVS, 20)

head(data.frame(cbind(withDC, withoutDC)), 20)


# now on with the GMM
set.seed(1)

dvs.nodc <- pres2$DVS
gmm.nodc <- normalmixEM(dvs.nodc, k = 2)

ggplot(data.frame(x = gmm.nodc$x)) +
  geom_histogram(aes(x, ..density..), fill = "darkgray", bins = 20) +
  stat_function(geom = "line", fun = plot_mix_comps,
                args = list(gmm.nodc$mu[1], gmm.nodc$sigma[1], lam = gmm.nodc$lambda[1]),
                colour = amerika_palettes$Republican[3]) +
  stat_function(geom = "line", fun = plot_mix_comps,
                args = list(gmm.nodc$mu[2], gmm.nodc$sigma[2], lam = gmm.nodc$lambda[2]),
                colour = amerika_palettes$Democrat[3]) +
  xlab("Democratic Vote Shares") +
  ylab("Density") + 
  theme_minimal()

# Call specific values from the output
# means
gmm.nodc$mu

# sd's
gmm.nodc$sigma

# weights
gmm.nodc$lambda


## Explore component densities a bit
# Table for viz
posterior <- data.frame(cbind(gmm.nodc$x, gmm.nodc$posterior))
rownames(posterior) <- pres2$State
round(head(posterior, 10), 3)

# get counts for each component 
posterior$component <- ifelse(posterior$comp.1 > 0.3, 1, 2)
table(posterior$component) 

# View the DVS by component (again, saying nothing of parties explicitly)
ggplot(posterior, aes(x = V1)) + 
  geom_histogram(aes(fill = factor(component)), stat ="bin", binwidth = 3) +
  labs(x = "Democratic Vote Share",
       y = "Count of States",
       title = "Gaussian Mixture Model") +
  scale_fill_manual(values=c(amerika_palettes$Republican[3], amerika_palettes$Democrat[3]),
                    name="Component",
                    breaks=c("1", "2"),
                    labels=c("1", "2")) +
  geom_vline(xintercept = 50, linetype="solid", 
             color = "darkgray", size=1.2) +
  theme_minimal()
