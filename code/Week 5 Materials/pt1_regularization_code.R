# Regularization in regression problems
# Introduction to Machine Learning (33002)
# Philip Waggoner, pdwaggoner@uchicago.edu

# load libs
library(glmnet)
library(foreach)
library(pROC)
library(ggfortify)
library(ggpubr)
library(gridExtra)
library(tidyverse)


# read in the state medicaid data
dataset <- read_csv(file.choose())

# outcome is whether a state will oppose expanding their state-run medicaid programs (oppose_expansion)
# inputs: gop_governor (dichotomous), percent_favorable_aca (survey ratings of affordable care act), gop_leg (whether republican controlled legislature), bal2012 (2012 vote share), multiplier (medicaid multipler/state level), percent_nonwhite, percent_uninsured, percent_metro, percent_poverty

# select a subset of the full features to include these meain features to explain/predict whether a state is likely to oppose expanding their state-run medicaid programs
dataset <- dataset %>% 
  select(oppose_expansion, gop_governor, percent_favorable_aca, gop_leg, bal2012, multiplier,
           percent_nonwhite, percent_uninsured, percent_metro, percent_poverty) 

set.seed(1244)

# store some things to make fitting the model a bit easier; that is, moving around individual objects rather than calling objects from a dataset. This is just a choice I have made, it's not "required". Play around with different approaches if you'd like.
d <- dataset
f <- oppose_expansion ~ scale(d[,-1])
y <- d$oppose_expansion
X <- model.matrix(f, d)[,-1]
n <- length(y)

## LASSO WITH ALPHA = 1 (as recall, if the mixing parameter is 1, we are fitting just a LASSO)
# CV to find optimal lambda
cv1 <- cv.glmnet( # super handy function in the glmnet package to fit many regularized models across a CV range/ that is, no external packages need for CV in this context
  X, y, # inputs and outcome
  family = "binomial", # binomial here, because technically outcomeis dichotomous; but we are still treating this as a regression problem as we want to predict probabilities of opposition to expansion, rather than classify; change to gaussian for regression approach (see more: https://www.rdocumentation.org/packages/glmnet/versions/4.1/topics/glmnet)
  nfold = 10, # k-fold CV, setting k = 10
  alpha = 1) # specifying LASSO, not ridge or EN

plot(cv1) # viz cv


# output is many lasso model (one for each point + error bars) at different values of the penality, lambda. The left dotted line means minimium error, and th right dotted line means 1 SE from that minimum; read more about the 1 SE rule in your book; either are appropriate for "optimality"

# also note the numbers at the top of the plot mean the total number of features left in the model (that is *not* dropped) at that given model. So, e.g., the highest value of penalty toward the right of the plot, only 2 features remain in the model; this is the "feature selection" part we discussed in the notes

# Store fitted values across multiple values of lambda, and then plot the "tuned" coefficient plot, which shows the same thing as the previous plot, only at the feature levele (each colored line is a feature), rather than at the model level (e.g., each red dot was a fit of the model at a different value of lambda)

cv1.glmnet.fit <- (cv1$glmnet.fit)
plot(cv1.glmnet.fit, xvar = "lambda")

## NOTE: when a line (feature) hits the 0 line, that is no variance at the verion of the model, then that feature drops out of the specification, and the numbers atthe top of the plot decrease

## FURTHER: read this plot as a cross-section, where at some specific value of lambda on the X axis, we are observing a single version of the model. 

# Fit the optimal model based on our CV procedure from above; note, you could change "lambda.min" to "lambda.1se" for the right dotted line in the above CV plot to pick the value of lambda 1 standard error away from the minimum; again, either is generally well accepted in practice
lassomod <- glmnet(X, y, 
                   family = "binomial", 
                   lambda = cv1$lambda.min, 
                   alpha = 1)
(lassomod$beta) # inspect the coefficients, though not terribly helpful beyond noting the features with "." next to them mean they are dropped from the optimal model. 


#
## RIDGE WITH ALPHA = 0 - everything from above is exactly the same, except alpha is now set to 0, meaning fit a ridge regression
# CV to find optimal lambda
cv2 <- cv.glmnet(X, y, 
                 family = "binomial", 
                 nfold = 10, 
                 alpha = 0)

# plot cv error across lambda
plot(cv2)

# Store the fitted values for plotting all smoothed models
cv2.glmnet.fit <- (cv2$glmnet.fit)
plot(cv2.glmnet.fit, xvar = "lambda")

## Importantly: note that none of the features intersect the 0 line, meaning non are dropped from the model, and thus the number at the top of the plot are constant at 9 for all fits of the model; this is as we discussed in the lecture notes.

# Fit the optimal model and inspect coefs
ridgemod <- glmnet(X, y, 
                   family = "binomial", 
                   lambda = cv2$lambda.min, 
                   alpha = 0)
(ridgemod$beta) # here again, not terribly useful to inspect raw coefficient values, but note we have all features remaining in the model


# 
## ELASTIC NET WITH 0 < ALPHA < 1 -- that is, allow alph to range, and we will set up a grid search to find the optimal value of alpha, which recall, controls the mixing of the ridge (L2) and LASSO (L1) penalties

# CV for search for alpha values (ranging between 0 and 1, for mixing of L1 and L2) - note, I am doing this in "parallel" to speed up computation; you don't have to do this, but it can be slow, especially for really large data sets (which this one isn't)
doParallel::registerDoParallel(cores=2) # saying, use 2 core on my computer's harddrive to do all of the following computation

a <- seq(0.1, 0.9, 0.05) # create a sequence to constrain the search - that is values of alpha from 0.1 to 0.9, by increments of 0.05, which give us 17 candidate values of alpha to search over

# now, start the search over this range, using CV
search <- foreach(i = a, .combine = rbind) %dopar% {
  # calcluate multiple lambda values at all alpha values to find the optimal mix between L1 and L2
  cv <- cv.glmnet(X, y, 
                  family = "binomial",
                  nfold = 10, 
                  alpha = i) # note, this was either 1 or 0 in the previous examples: now, its "i", meaning search of the range of alpha vals we just set up
  # Store in a data frame to search next
  data.frame(cvm = cv$cvm[cv$lambda == cv$lambda.min], 
             lambda.min = cv$lambda.min, alpha = i)
}

# select and plot the alpha value for the optimal lambda value that minimizes MSE 
(cv3 <- search[search$cvm == min(search$cvm), ])

# plot the values of alpha across different CV error rates, and put a red line at the minimum value, suggesting the best version of the model at this value of alpha, which in my case looks very similar to a LASSO regression (yours might be a little different because of the randomness with the CV procedure)
ggplot(search, aes(alpha, cvm)) +
  geom_line() +
  geom_vline(aes(xintercept = cv3[, 3], 
                 color="red"), 
             show.legend = FALSE) +
  labs(y = "Mean Squared Error",
       x = "Alpha") +
  theme_minimal()


# Now, CV again to find the optimal value of the OTHER hyperparameter, lambda, but based on out optimal/tuned elastic net model at the specified value of alpha
cv3.1 <- cv.glmnet(X, y, 
                   family = "binomial", 
                   nfold = 10,
                   alpha = cv3$alpha)
plot(cv3.1)

cv3.glmnet.fit <- (cv3.1$glmnet.fit)
plot(cv3.glmnet.fit, xvar = "lambda") # coefficient tuning plot at the feature level as before; note it looks a lot like a LASSO (intersecting the 0 line), because alpha was pretty high, again, in my case

# Fit and inspect the optimal model
elasticnetmod <- glmnet(X, y, 
                        family = "binomial", 
                        lambda = cv3$lambda.min, 
                        alpha = cv3$alpha)

elasticnetmod$beta # coefs (and dropped features)
(elasticnetmod$lambda == cv3$lambda.min) # check lambda values; should evaluate to TRUE


#

# you could stop at the point, but for a much prettier, but slightly more complex visualization of our final models, we could use ggplot()

# better plot of the output via autoplot (ggplot2)
lassoplot <- autoplot(cv1$glmnet.fit, "lambda", label = TRUE, main = "LASSO (alpha = 1)") + 
  theme(legend.position="right") + 
  scale_colour_discrete(name = "Features", 
                        labels = c("Intercept", "GOP Governor", "% Favorable ACA", "GOP Legislature", "2012 Ballot", "Multiplier", "% Nonwhite", "% Uninsured", "% Metropolitan", "% Poverty")) + 
  theme(legend.title = element_text(size=20)) + 
  theme(legend.text = element_text(size = 18)) + 
  geom_vline(data = NULL, 
             xintercept = log(cv1$lambda.min), 
             na.rm = FALSE, show.legend = TRUE)

ridgeplot <- autoplot(cv2$glmnet.fit, "lambda", label = TRUE, main = "Ridge (alpha = 0)") + 
  geom_vline(data = NULL, 
             xintercept = log(cv2$lambda.min), 
             na.rm = FALSE, 
             show.legend = TRUE)

elasticnetplot <- autoplot(cv3.1$glmnet.fit, "lambda", label = TRUE, main = "Elastic Net (alpha = 0.9") + # again, alpha may be a bit different for you 
  geom_vline(data = NULL, 
             xintercept = log(cv3$lambda.min), 
             na.rm = FALSE, 
             show.legend = TRUE)

# manually create a legend for the 3 plots
g_legend <- function(plot){
  tmp <- ggplot_gtable(ggplot_build(plot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

mylegend <- g_legend(lassoplot)

# and show them side by side - if the parts of the plots are overlapping, just make the plot window much bigger or zoom in on the plot
final_plot <- grid.arrange(arrangeGrob(ridgeplot + 
                                     theme_bw() +
                                     theme(legend.position="none"),
                                   lassoplot + 
                                     theme_bw() +
                                     theme(legend.position="none"), 
                                   elasticnetplot + 
                                     theme_bw() +
                                     theme(legend.position="none"), 
                                   mylegend, nrow = 1))

# fin 
