# Introduction to Machine Learning (33002)
# Philip Waggoner (https://pdwaggoner.github.io/)

# NOTE: much of this code is from or based on my book under contract with Cambridge University Press; please do not share beyond this class. 

# Code for class: LLE

# libraries needed for this section
library(tidyverse)
library(here)
library(lle)
library(amerika)
library(parallel)
library(ggrepel)
library(tictoc)
library(patchwork)
library(plot3D)
library(corrr)

# The s-curve data
data("lle_scurve_data") 

# SCURVE figure
scatter3D(lle_scurve_data[,1], 
          lle_scurve_data[,2],  
          lle_scurve_data[,3], 
          bty = "f",
          pch = 19,
          phi = 7,
          theta = 25,
          colkey = FALSE,
          col = ramp.col(c(amerika_palettes$Republican[1], 
                           amerika_palettes$Democrat[1])))

# 2D figure
cores <- detectCores() - 1

opt_k <- calc_k(lle_scurve_data,
                m = 2, 
                kmin = 1,
                kmax = 25,
                parallel = TRUE,
                cpus = cores)

ok <- opt_k[which.min(opt_k$rho), ]

lle_fit <- lle(lle_scurve_data,
               m = 2,
               k = ok$k)

lle_scurve_data %>% 
  tibble() %>% 
  ggplot(aes(lle_fit$Y[,1], lle_fit$Y[,2], 
             col = lle_scurve_data[,3])) +
  geom_point() +
  labs(x = "X",
       y = "Y",
       col = "") +
  scale_color_gradient(low = amerika_palettes$Democrat[1],
                       high = amerika_palettes$Republican[1]) +
  theme_classic() +
  theme(axis.ticks.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank()) +
  theme(legend.position = "none")


# Now 3D plots with ANES

# First, read in cleaned and preprocessed 2019 ANES Pilot Data
anes <- read_rds(here("Data", "anes.rds"))

# first, correlations
## fttrump vs. all others
anes %>%
  dplyr::select(-democrat) %>% 
  correlate(use = "pairwise.complete.obs",
            method = "pearson",
            quiet = TRUE) %>% 
  focus(Trump) %>%
  mutate(rowname = reorder(rowname, Trump)) %>%
  ggplot(aes(rowname, Trump)) +
  geom_col() + 
  coord_flip() + 
  labs(y = "Feelings Toward Trump", 
       x = "All Other Feeling Thermometers") +
  theme_minimal()

## ftjapan vs. all others
anes %>%
  dplyr::select(-democrat) %>% 
  correlate(use = "pairwise.complete.obs",
            method = "pearson",
            quiet = TRUE) %>% 
  focus(Japan) %>%
  mutate(rowname = reorder(rowname, Japan)) %>%
  ggplot(aes(rowname, Japan)) +
  geom_col() + 
  coord_flip() + 
  labs(y = "Feelings Toward Japan", 
       x = "All Other Feeling Thermometers") + 
  theme_minimal()

## network viz
anes %>%
  dplyr::select(-democrat) %>% 
  correlate(use = "pairwise.complete.obs",
            method = "pearson",
            quiet = TRUE) %>% 
  network_plot(colors = c(amerika_palettes$Democrat[1], 
                          amerika_palettes$Republican[1]),
               curved = FALSE) 


set.seed(1234)

# 3D versions of ANES data??
anes_scaled <- anes[ ,1:35] %>% 
  scale() %>% 
  as_tibble()

{
  par(mfrow = c(2,2))
  scatter3D(anes_scaled$Trump, 
            anes_scaled$Obama, 
            anes_scaled$Sanders,
            bty = "f",
            pch = 1,
            phi = 7,
            theta = 25,
            colkey = FALSE,
            col = ramp.col(c(amerika_palettes$Republican[1], 
                             amerika_palettes$Democrat[1])),
            main = "Politicians",
            xlab = "Donald Trump",
            ylab = "Barack Obama",
            zlab = "Bernie Sanders"
  )
  
  scatter3D(anes_scaled$NRA, 
            anes_scaled$NATO, 
            anes_scaled$UN,
            bty = "f",
            pch = 1,
            phi = 7,
            theta = 25,
            colkey = FALSE,
            col = ramp.col(c(amerika_palettes$Republican[1], 
                             amerika_palettes$Democrat[1])),
            main = "Institutions",
            xlab = "NRA",
            ylab = "NATO",
            zlab = "UN"
  )
  scatter3D(anes_scaled$Illegal, 
            anes_scaled$Immigrants, 
            anes_scaled$ICE,
            bty = "f",
            pch = 1,
            phi = 7,
            theta = 25,
            colkey = FALSE,
            col = ramp.col(c(amerika_palettes$Republican[1], 
                             amerika_palettes$Democrat[1])),
            main = "Issues (Immigration)",
            xlab = "Illegal Immigrants",
            ylab = "Immigrants",
            zlab = "ICE"
  )
  scatter3D(anes_scaled$Palestine, 
            anes_scaled$`Saudi Arabia`, 
            anes_scaled$Israel,
            bty = "f",
            pch = 1,
            phi = 7,
            theta = 25,
            colkey = FALSE,
            col = ramp.col(c(amerika_palettes$Republican[1], 
                             amerika_palettes$Democrat[1])),
            main = "Countries (Middle East)",
            xlab = "Palestine",
            ylab = "Saudi Arabia",
            zlab = "Israel"
  )
  par(mfrow = c(1,1))
  }

# not too great...


# Learn the ANES structure

# First, find optimal k (lle's version of a grid search)
tic() 
find_k <- calc_k(anes_scaled,
                 m = 2, 
                 parallel = TRUE,
                 cpus = cores) 
toc() # ~ 10.9 minutes on 3 cores; ~ 9.2 minutes on 7 cores

# inspect -- optimal k? (a couple options...)

## option 1: manual
find_k %>% 
  arrange(rho) 

## option 2: extracting via which.min()
find_k[which.min(find_k$rho), ] 

# extract optimal k based on rho
optimal_k_rho <- find_k %>% 
  arrange(rho) %>% 
  filter(rho == min(.))

# viz
find_k %>% 
  arrange(rho) %>% 
  ggplot(aes(k, rho)) +
  geom_line() +
  geom_point(color = ifelse(find_k$k == min(find_k$k), 
                            "red", 
                            "black")) +
  geom_vline(xintercept = optimal_k_rho$k, 
             linetype = "dashed", 
             color = "red") +
  geom_label_repel(aes(label = k),
                   box.padding = unit(0.5, 'lines')) +
  labs(x = "Neighborhood Size (k)",
       y = expression(rho)) +
  theme_minimal()

# fit
{
  tic() 
  lle_fit <- lle(anes_scaled,
                 m = 2,
                 nnk = TRUE,
                 k = 19)
  toc() # ~ 1.5 minutes on 3 cores; ~ 1.4 minutes on 7 cores
}

# full LLE viz
anes %>%
  ggplot(aes(x = lle_fit$Y[,1], 
             y = lle_fit$Y[,2], 
             col = factor(democrat))) +
  geom_point() +
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "First Dimension",
       y = "Second Dimension",
       title = "LLE") + 
  theme_minimal()


# compare with raw inputs
p1 <- anes %>% 
  ggplot(aes(Trump, Obama, 
             color = factor(democrat))) +
  geom_density_2d() + 
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "Feelings Toward Trump",
       y = "Feelings Toward Obama") +
  theme_minimal()

p2 <- anes %>% 
  ggplot(aes(ICE, Illegal, 
             color = factor(democrat))) +
  geom_density_2d() + 
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "Feelings Toward ICE",
       y = "Feelings Toward Illegal Immigrants") +
  theme_minimal()

p3 <- anes %>% 
  ggplot(aes(UN, NATO, 
             color = factor(democrat))) +
  geom_density_2d() + 
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "Feelings Toward the United Nations",
       y = "Feelings Toward NATO") +
  theme_minimal()

p4 <- anes %>% 
  ggplot(aes(Palestine, Israel, 
             color = factor(democrat))) +
  geom_density_2d() + 
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", 
                              "Democrat")) +
  labs(x = "Feelings Toward Palestine",
       y = "Feelings Toward Israel") +
  theme_minimal()

# viz together
(p1 + p2) /
  (p3 + p4)
