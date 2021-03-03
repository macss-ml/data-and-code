# Autoencoders (33002)
# NOTE: As before, much of this is taken from my newest book under contract with Cambridge University Press; so, please don't share the code beyond this class
# Philip Waggoner, pdwaggoner@uchicago.edu

# Load libraries
library(tidyverse)
library(here)
library(amerika)
library(tictoc)
library(h2o)
library(bit64) # speeds up some h2o computation

# read in ANES 2019
anes <- read_rds(here("Data", "anes.rds")) # go back to the same 2019 ANES data we've been using the past few weeks

# fitting
set.seed(1234)

anes$democrat <- factor(anes$democrat)

# initializing the h2o cluster; have to do this to work with the h2o engine
my_h2o <- h2o.init()

# h2o df
anes_h2o <- anes %>% 
  as.h2o()

# train, val, test
split_frame <- h2o.splitFrame(anes_h2o, 
                              ratios = c(0.6, 0.2), 
                              seed = 1234)   

split_frame %>% 
  str()

train <- split_frame[[1]]
validation <- split_frame[[2]]
test <- split_frame[[3]]

# Store response and predictors separately (per h2o syntax)
response <- "democrat"

predictors <- setdiff(colnames(train), response)

# vanilla AE
{
  tic()
autoencoder <- h2o.deeplearning(x = predictors,
                                training_frame = train, 
                                autoencoder = TRUE, 
                                reproducible = TRUE,
                                seed = 1234, 
                                hidden = c(16), 
                                epochs = 100, 
                                activation = "Tanh")
  toc()
} # ~ 4.5 seconds

# save model, if desired
#h2o.saveModel(autoencoder, 
#              path = "autoencoder", 
#              force = TRUE)

# load the model directly, if desired
#autoencoder <- h2o.loadModel(".../file/path/here")

# feature extraction
codings_train <- h2o.deepfeatures(autoencoder, 
                                  data = train, 
                                  layer = 1) %>% 
  as.data.frame() %>%
  mutate(democrat = as.vector(train[ , 36]))

##  --> NOTE: "layer" in the above chunk is referring to the specific hidden layer where the codes are stored, and thus which we want to use as "features"; especially useful when there are multiple hidden layers, which are used for feature extraction

## Output is read as, e.g., DF.L1.C1 - "data frame, layer number, column number"


# Numeric inspection of the "scores"
codings_train %>% 
  head(10)


# viz inspection of deep features

# Substantively our goal here is checking to see whether our AE has detected the party labels or not over the first two deep features

{
p1 <- ggplot(codings_train, aes(x = DF.L1.C1, 
                                y = DF.L1.C2, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 1 & 2",
       color = "Democrat") + 
  theme_minimal()

# (3 and 4)
p2 <- ggplot(codings_train, aes(x = DF.L1.C3, 
                                y = DF.L1.C4, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 3 & 4",
       color = "Democrat") + 
  theme_minimal()

# 5 & 6
p3 <- ggplot(codings_train, aes(x = DF.L1.C5, 
                                y = DF.L1.C6, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 5 & 6",
       color = "Democrat") + 
  theme_minimal()

# 7 & 8
p4 <- ggplot(codings_train, aes(x = DF.L1.C7, 
                                y = DF.L1.C8, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 7 & 8",
       color = "Democrat") + 
  theme_minimal()

# 9 & 10
p5 <- ggplot(codings_train, aes(x = DF.L1.C9, 
                                y = DF.L1.C10, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 9 & 10",
       color = "Democrat") + 
  theme_minimal()

# 11 & 12
p6 <- ggplot(codings_train, aes(x = DF.L1.C11, 
                                y = DF.L1.C12, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 11 & 12",
       color = "Democrat") + 
  theme_minimal()

# 13 & 14
p7 <- ggplot(codings_train, aes(x = DF.L1.C13, 
                                y = DF.L1.C14, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 13 & 14",
       color = "Democrat") + 
  theme_minimal()

# 15 & 16
p8 <- ggplot(codings_train, aes(x = DF.L1.C15, 
                                y = DF.L1.C16, 
                                color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Deep Features 15 & 16",
       color = "Democrat") + 
  theme_minimal()

# view together
library(patchwork)

(p1 + p2 + p3 + p4) / 
  (p5 + p6 + p7 + p8)
}

# Let's go deeper (pun)

#  --> this time, instead of just color, let's predict party ID (a supervised task), and then explore feature importance

## predict party afiliation

# first, feature extraction
codings_val <- h2o.deepfeatures(object = autoencoder, 
                                data = validation, 
                                layer = 1) %>%
  as.data.frame() %>%
  mutate(democrat = as.factor(as.vector(validation[ , 36]))) %>%
  as.h2o()

deep_features <- setdiff(colnames(codings_val), response)

# fit the "deep" neural net (2 hidden layers, each with 8 nodes, giving 16 total)
deep_net <- h2o.deeplearning(y = response,
                             x = deep_features,
                             training_frame = codings_val,
                             activation = "Tanh",
                             hidden = c(8, 8), 
                             epochs = 100)

## preds on test set
test_3 <- h2o.deepfeatures(object = autoencoder, 
                           data = test, 
                           layer = 1)

test_pred <- h2o.predict(deep_net, test_3, type = "response") %>%
  as.data.frame() %>%
  mutate(truth = as.vector(test[, 36]))

# confusion matrix
print(h2o.predict(deep_net, test_3) %>%
        as.data.frame() %>%
        mutate(truth = as.vector(test[, 36])) %>%
        group_by(truth, predict) %>%
        summarise(n = n()) %>%
        mutate(freq = n / sum(n)))
# 


## Finally: Exploring feature importance

# calculate first
fimp <- as.data.frame(h2o.varimp(deep_net)) %>% 
  arrange(desc(relative_importance))

# viz relative importance
fimp %>% 
  ggplot(aes(x = relative_importance, 
             y = reorder(variable, -relative_importance))) +
  geom_point(color = "dark red", 
             fill = "dark red", 
             alpha = 0.5,
             size = 5) +
  labs(title = "Relative Feature Importance",
       subtitle = "Deep Neural Network (2 hidden layers with 16 total neurons)",
       x = "Relative Importance",
       y = "Feature") + 
  theme_minimal()
# 


# Let's take a closer look at the most important deep features cross the training and validation sets; here, we're expecting substantively similar patterns across each set

## Note: your "most important" features may be different from mine to due random processes; if so, simply update the x and y axes in the code below for *both* plots (training first starting on line 291, then validation starting on line 306)

codings_val2 <- h2o.deepfeatures(object = autoencoder, 
                                 data = validation, 
                                 layer = 1) %>%
  as.data.frame() %>%
  mutate(democrat = as.factor(as.vector(validation[ , 36]))) 

# training plot
tr <- ggplot(codings_train, aes(x = DF.L1.C8, 
                                y = DF.L1.C6, 
                               color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Training Set",
       color = "Democrat") + 
  theme_minimal()

# validation plot
val <- ggplot(codings_val2, aes(x = DF.L1.C8, 
                                y = DF.L1.C6, 
                          color = factor(democrat))) +
  geom_point(alpha = 0.6) + 
  stat_ellipse() +
  scale_color_manual(values=c(amerika_palettes$Republican[1], 
                              amerika_palettes$Democrat[1]),
                     name="Party",
                     breaks=c("0", "1"),
                     labels=c("Non-Democrat", "Democrat")) +
  labs(title = "Validation Set",
       color = "Democrat") + 
  theme_minimal()

# now side by side
# similar patterns as expected
(tr + val)

# Shut down h2o cluster when you're finished (it will ask you to make sure you want to; which you do)
h2o.shutdown()

# A final note: For bigger data applications, you might consider using parallel processing like we've done for a few examples in past classes. This will speed up computation a bit. 

# You have plenty of code to get you started. Play around with the code and apply it for your own work. My door is always opened (virtually of course; send me an email whenever you want). Good luck!
