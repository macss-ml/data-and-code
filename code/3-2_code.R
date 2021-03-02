# Dimension reduction for viz (33002)
# NOTE: As before, much of this is taken from my newest book under contract with Cambridge University Press; so, please don't share the code beyond this class
# Philip Waggoner, pdwaggoner@uchicago.edu

# Techniques: t-SNE, UMAP

#
# t-SNE first

# libraries needed for this section
library(tidyverse)
library(here)
library(amerika)
library(tictoc)
library(patchwork)
library(Rtsne)
library(umap)
library(tidymodels)
library(embed)

# Read in cleaned and preprocessed 2019 ANES Pilot Data (35 FTs + democrat party feature)
anes <- read_rds(here("Data", "anes.rds"))

set.seed(1234)

{
  tic()
  
  # perplexity = 2
  tsne_2 <- Rtsne(as.matrix(anes[ ,1:35]), 
                  perplexity = 2)
  
  perp_2 <- anes %>%
    ggplot(aes(tsne_2$Y[,1], tsne_2$Y[,2], 
               col = factor(democrat))) +
    geom_point() +
    stat_ellipse() +
    scale_color_manual(values=c(amerika_palettes$Republican[1], 
                                amerika_palettes$Democrat[1]),
                       name="Democrat",
                       breaks=c("0", "1"),
                       labels=c("No", 
                                "Yes")) +
    ylim(-100, 100) +
    xlim(-100, 100) +
    labs(x = "First dimension",
         y = "Second dimension",
         subtitle = "Perplexity = 2") +
    theme_minimal()
  
  # perplexity = 25
  tsne_25 <- Rtsne(as.matrix(anes[ ,1:35]), 
                   perplexity = 25) 
  
  perp_25 <- anes %>%
    ggplot(aes(tsne_25$Y[,1], tsne_25$Y[,2], 
               col = factor(democrat))) +
    geom_point() +
    stat_ellipse() +
    scale_color_manual(values=c(amerika_palettes$Republican[1], 
                                amerika_palettes$Democrat[1]),
                       name="Democrat",
                       breaks=c("0", "1"),
                       labels=c("No", 
                                "Yes")) +
    ylim(-100, 100) +
    xlim(-100, 100) +
    labs(x = "First dimension",
         y = "Second dimension",
         subtitle = "Perplexity = 25") +
    theme_minimal()


  # perplexity = 50
  tsne_50 <- Rtsne(as.matrix(anes[ ,1:35]), 
                    perplexity = 50) 
  
  perp_50 <- anes %>%
    ggplot(aes(tsne_50$Y[,1], tsne_50$Y[,2], 
               col = factor(democrat))) +
    geom_point() +
    stat_ellipse() +
    scale_color_manual(values=c(amerika_palettes$Republican[1], 
                                amerika_palettes$Democrat[1]),
                       name="Democrat",
                       breaks=c("0", "1"),
                       labels=c("No", 
                                "Yes")) +
    ylim(-100, 100) +
    xlim(-100, 100) +
    labs(x = "First dimension",
         y = "Second dimension",
         subtitle = "Perplexity = 50") +
    theme_minimal()
  
  
  # perplexity = 500
  tsne_500 <- Rtsne(as.matrix(anes[ ,1:35]), 
                    perplexity = 500) 
  
  perp_500 <- anes %>%
    ggplot(aes(tsne_500$Y[,1], tsne_500$Y[,2], 
               col = factor(democrat))) +
    geom_point() +
    stat_ellipse() +
    scale_color_manual(values=c(amerika_palettes$Republican[1], 
                                amerika_palettes$Democrat[1]),
                       name="Democrat",
                       breaks=c("0", "1"),
                       labels=c("No", 
                                "Yes")) +
    ylim(-100, 100) +
    xlim(-100, 100) +
    labs(x = "First dimension",
         y = "Second dimension",
         subtitle = "Perplexity = 500") +
    theme_minimal()
  
  toc()
} # ~1 minute


# Visualize
tsne_plots <- (perp_2 + perp_25) /
  (perp_50 + perp_500)

tsne_plots

## with annotation if desired
#tsne_plots + plot_annotation(title = "t-SNE Results Across a Range of Perplexity",
#                             subtitle = "Color conditional on Party Affiliation")


#
# UMAP next

# epochs = 500
umap_fit_5 <- anes[,1:35] %>% 
  umap(n_neighbors = 5,
       metric = "euclidean",
       n_epochs = 500)
  
umap_fit_5 <- anes %>% 
  mutate_if(.funs = scale,
            .predicate = is.numeric,
            scale = FALSE) %>% 
  mutate(First_Dimension = umap_fit_5$layout[,1],
         Second_Dimension = umap_fit_5$layout[,2]) %>% 
  gather(key = "Variable",
         value = "Value",
         c(-First_Dimension, -Second_Dimension, -democrat))

k_5 <- ggplot(umap_fit_5, aes(First_Dimension, Second_Dimension, 
                              col = factor(democrat))) + 
  geom_point(alpha = 0.6) +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Democrat",
                     breaks=c("-0.418325434439179", 
                              "0.581674565560822"),
                     labels=c("No", 
                              "Yes")) +
  labs(title = " ",
       subtitle = "Neighborhood size: 5; Epochs = 500",
       x = "First Dimension",
       y = "Second Dimension") +
  theme_minimal()


# epochs = 20 
umap_fit_e_20 <- anes[,1:35] %>% 
  umap(n_neighbors = 5,
       metric = "euclidean",
       n_epochs = 20)

umap_fit_e_20 <- anes %>% 
  mutate_if(.funs = scale,
            .predicate = is.numeric,
            scale = FALSE) %>% 
  mutate(First_Dimension = umap_fit_e_20$layout[,1],
         Second_Dimension = umap_fit_e_20$layout[,2]) %>% 
  gather(key = "Variable",
         value = "Value",
         c(-First_Dimension, -Second_Dimension, -democrat))

e_20 <- ggplot(umap_fit_e_20, aes(First_Dimension, Second_Dimension, 
                                  col = factor(democrat))) + 
  geom_point(alpha = 0.6) +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Democrat",
                     breaks=c("-0.418325434439179", 
                              "0.581674565560822"),
                     labels=c("No", 
                              "Yes")) +
  labs(title = " ",
       subtitle = "Neighborhood size: 5; Epochs = 20",
       x = "First Dimension",
       y = "Second Dimension") +
  theme_minimal()

# side by side
k_5 + e_20


# SIDE NOTE: I have some code implementing UMAP in tidymodels. Let me know if you'd like this code; happy to share to show you what UMAP might look like in a tidy workflow.


##
## ON YOUR OWN
## 

# 1. The global vs. local representation of data in a t-SNE fit is controlled by the perplexity hyperparameter, where larger values mean a more global version, compared to smaller values which mean a more local version. Crank up perplexity to 1000 for a t-SNE fit to the ANES data. Plot the results, colored by party. *Caution*: this will take ~5-7 minutes to run.



# 2. The tradeoff in UMAP between global and local behavior is controlled by the n_neighbors hyperparameter, where larger values mean more neighbors to include in the fuzzy search region, versus fewer neighbors with smaller values for this hyperparameter. Fit a similarly global version of UMAP to the ANES data by cranking up the n_neighbors hyperparameter to 1000. Plot the results, colored by party. *Caution*: this will take about 5-10 minutes to run.



# 3. Do these global version of the algorithms reveal similar structure in the projection space or not? Give just a couple sentences describing your thoughts on global vs. local behavior, and also in comparing the t-SNE and UMAP algorithms. 

