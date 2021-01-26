# Introduction to Machine Learning (33002)
# Philip Waggoner (https://pdwaggoner.github.io/)

# Code for class

# SVM 
## theoretical first

# load libs and set seed
library(tidyverse)
library(e1071)

set.seed(2345)

# create the data for classification
x <- matrix(rnorm(20*2), ncol=2)
class <- c(rep(-1,10), rep(1,10))
x[class == 1, ] = x[class == 1, ] + 1

# perfectly linearly seperable?
ggplot(data.frame(x), 
       aes(X1, X2, color = factor(class))) +
  geom_point(size = 2) +
  theme_minimal()

# encode as factor for classification, rather than regression
train <- data.frame(x = x, class = as.factor(class)) 

svmfit <- svm(class ~ ., 
              data = train, 
              kernel = "linear", 
              cost = 10, 
              scale = FALSE); summary(svmfit)

# now plot fit
plot(svmfit, train)
svmfit$index

#
# what about a smaller cost value
svmfit <- svm(class ~ ., 
              data = train, 
              kernel = "linear", 
              cost = 0.1, 
              scale = TRUE); summary(svmfit)

plot(svmfit, train)
svmfit$index

# now we get a larger number of support vectors, because the margin is now wider given the lesser penalty which allowed for more observations to be considered in the range of the margin and threshold placement, i.e., more support vectors with a wider margin (smaller cost to widening the margin)


# CV
set.seed(2345)

tune_c <- tune(svm, 
               class ~ ., 
               data = train, 
               kernel = "linear", 
               ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

# best?
tuned_model <- tune_c$best.model
summary(tuned_model)


#
# Now we can predict the class label on a set of test obs
xtest <- matrix(rnorm(20*2), ncol = 2)
ytest <- sample(c(-1,1), 20, rep = TRUE)
xtest[ytest == 1,] = xtest[ytest == 1,] + 1
test <- data.frame(x = xtest, class = as.factor(ytest))

# predict class labels
class_pred <- predict(tuned_model, 
                      test)

table(predicted = class_pred, 
      true = test$class)

# Based on our tuned SVM, around 14 test observations are correctly classified; yours may be a little different as there is randomness involved here (data creation, CV for tuning, and so on)

svmfit_01 <- svm(class ~ ., 
                 data = train, 
                 kernel = "linear", 
                 cost = .01, 
                 scale = FALSE)

class_01 <- predict(svmfit_01, 
                    test)

table(predicted = class_01, 
      true = test$class)


# 
# Let's now increase the overlap 
set.seed(2345)

x <- matrix(rnorm(200*2), ncol = 2)
x[1:100,] = x[1:100,]+2
x[101:150,] = x[101:150,]-2
class <- c(rep(1,150),rep(2,50))
overlap_data <- data.frame(x = x, class = as.factor(class))

ggplot(overlap_data, aes(x.1, x.2, color = factor(class))) +
  geom_point(size = 2) +
  theme_minimal()

# let's split 70/30, train/test
train_overlap <- overlap_data %>%
  sample_frac(0.7)

test_overlap <- overlap_data %>%
  setdiff(train_overlap)

# Now fit the SVM
svmfit_overlap <- svm(class ~ ., 
                      data = train_overlap, 
                      kernel = "radial",  
                      gamma = 1, 
                      cost = 1)

plot(svmfit_overlap, train_overlap)



# Let's see what happens when we dramatically increase cost to reduce training errors
svmfit_overlap2 <- svm(class ~ ., 
                       data = train_overlap, 
                       kernel = "radial", 
                       gamma = 1, 
                       cost = 1e5)

plot(svmfit_overlap2, train_overlap)



# Let's return to CV to tune the SVM in this overlapping case
set.seed(2345)

tune_c <- tune(svm, 
               class ~ ., 
               data = train_overlap, 
               kernel = "radial",
               ranges = list(cost = c(0.1, 1, 10, 100, 1000), 
                             gamma = c(0.5, 1, 2, 3)
               ))

tuned_overlap_model <- tune_c$best.model
summary(tuned_overlap_model)

# A note on gamma:
# Recall, gamma controls the influence that each case has on the position of the hyperplane and is used by all the kernel functions except the linear kernel.

# The larger gamma, the more granular the contours of the decision boundary will be (potentially leading to overfitting).

# The smaller gamma, the less granular the contours of the decision boundary will be (potentially leading to underfitting)

plot(tuned_overlap_model, train_overlap)

table(true = test_overlap$class, 
      pred = predict(tuned_overlap_model, 
                     newdata = test_overlap))

# how did we do?


## REAL APPLICATION
library(tidyverse)
library(here)
library(caret)
library(tictoc)
library(tidymodels)

set.seed(1234)
theme_set(theme_minimal())

# first (and focus): Herron's study/Krehbiel's data
herron <- read_csv(here("data", "Herron.csv")) 


# OUTCOME: "House members who chose to cosponsor this bill, formally known as H.R. 3266" (1 = yes, 0 = no); From Herron's replication of Krehbiel 1995, on p. 94 of Herron 

# The value of SVM in the Herron scenario of incorporating uncertainty in estimates of percentage of correct predictions (ePCP), we can use 10-fold cross-validation and look to the data to get significantly higher accuracy over all five models of Herron's via PCP or ePCP as reported in table 1 (p. 95 from Herron). And further, as we are NOT estimating a probability model, we are able to bypass the core threats presented by uncertainty estimates (SEs) at the heart of Herron's method/approach. Thus, by looking to the data (via CV) and using a geometric solution (via SVM), instead of a probability-based solution, we not only bypass these issues that Herron rightly points out in probability-based classification, which are namely overstating predictive accuracy (i.e., not accounting for uncertainty in postestimation, which gives the traditional PCP that Herron was updating), but we ALSO get a much more accurate solution (~ 0.96 AUC, regardless of kernel). 

# set up response for using caret to fit the models
herron <- herron %>%
  mutate(cosp_fact = factor(cosp, 
                            levels = c(0, 1), 
                            labels = c("No", "Yes")))


# Approach 1: linear kernel (SVC)
cv_ctrl <- trainControl(method = "cv",
                        number = 10,
                        savePredictions = "final",
                        classProbs = TRUE)

# fit model with linear kernel 
{
  tic()
  svm_linear <- train(
    cosp_fact ~ ada + ntu + democrat + firstelected + 
      margin + appromember + budgetmember, 
    data = herron, 
    method = "svmLinear",
    trControl = cv_ctrl,
    tuneLength = 10)
  toc()
}

# can draw indiv ROC curve and calc AUC
#svm_linear_roc <- roc(predictor = svm_linear$pred$Yes,
#                      response = svm_linear$pred$obs,
#                      levels = rev(levels(herron$cosp_fact)))

#plot(svm_linear_roc)
#auc(svm_linear_roc)

# polynomial kernel
{
  tic()
  svm_poly <- train(
    cosp_fact ~ ada + ntu + democrat + firstelected + 
      margin + appromember + budgetmember, 
    data = herron, 
    method = "svmPoly",
    trControl = cv_ctrl)
  toc()
}

# radial kernel
{
  tic()
  svm_radial <- train(
    cosp_fact ~ ada + ntu + democrat + firstelected + 
      margin + appromember + budgetmember, 
    data = herron, 
    method = "svmRadial",
    trControl = cv_ctrl)
  toc()
}

# Plot ROC for all kernels, and overlay curves
bind_rows(
  Linear = svm_linear$pred,
  Polynomial = svm_poly$pred,
  Radial = svm_radial$pred,
  .id = "kernel"
) %>%
  group_by(kernel) %>%
  roc_curve(truth = obs, 
            estimate = Yes) %>%
  ggplot(aes(x = 1 - specificity, 
             y = sensitivity, 
             color = kernel)) +
  geom_path() +
  geom_abline(lty = 3) +
  scale_color_brewer(type = "qual") +
  coord_flip() +
  labs(title = "Comparison of ROC Curves by Kernel",
       subtitle = "10-fold CV",
       x = "Specificity",
       y = "Sensitivity",
       color = NULL) +
  theme(legend.position = "bottom")

# Plot AUC for all kernels and directly compare (note the consistency across each; but note linear is the best, with the smallest amount (by a hair) under the curve left unexplained)
bind_rows(
  Linear = svm_linear$pred,
  Polynomial = svm_poly$pred,
  Radial = svm_radial$pred,
  .id = "kernel"
) %>%
  group_by(kernel) %>%
  roc_auc(truth = obs, Yes) %>%
  group_by(kernel) %>%
  summarize(.estimate = mean(.estimate)) %>%
  ggplot(aes(fct_reorder(kernel, -.estimate), .estimate)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(title = "Comparison of Area Under the Curve (AUC) by Kernel",
       subtitle = "10-fold CV",
       x = "Algorithm",
       y = "1 - AUC")

# These are cross-validated measures, so it's not as if they should be heavily biased. However they are all really close to each other, so the differences across kernels are likely not that substantial, suggesting the SVM performs extremely well on these data. 

# But what about in comparison to other classifiers we have learned about? Let's compare to the logistic regression from last week. 
{
  tic()
  herron_glm <- train(
    cosp_fact ~ ada + ntu + democrat + firstelected + 
      margin + appromember + budgetmember, 
    data = herron, 
    method = "glm",
    family = "binomial",
    trControl = cv_ctrl)
  toc()
  }


## Now, let's compare fit across all
# ROC
bind_rows(
  `SVM (linear)` = svm_linear$pred,
  `SVM (polynomial)` = svm_poly$pred,
  `SVM (radial)` = svm_radial$pred,
  `Logistic regression` = herron_glm$pred,
  .id = "kernel"
) %>%
  group_by(kernel) %>%
  roc_curve(truth = obs, estimate = Yes) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity, color = kernel)) +
  geom_path() +
  geom_abline(lty = 3) +
  scale_color_brewer(type = "qual") +
  coord_flip() +
  labs(title = "Comparison of ROC Curves by Classifer",
       subtitle = "10-fold CV",
       x = "Specificity",
       y = "Sensitivity",
       color = NULL)

# AUC
bind_rows(
  `SVM (linear)` = svm_linear$pred,
  `SVM (polynomial)` = svm_poly$pred,
  `SVM (radial)` = svm_radial$pred,
  `Logistic regression` = herron_glm$pred,
  .id = "kernel"
) %>%
  group_by(kernel) %>%
  roc_auc(truth = obs, Yes) %>%
  group_by(kernel) %>%
  summarize(.estimate = mean(.estimate)) %>%
  ggplot(aes(fct_reorder(kernel, -.estimate), .estimate)) +
  geom_col() +
  scale_y_continuous(labels = scales::percent) +
  coord_flip() +
  labs(title = "Comparison of Area Under the Curve (AUC) by Classifier",
       subtitle = "10-fold CV",
       x = "Algorithm",
       y = "1 - AUC")
