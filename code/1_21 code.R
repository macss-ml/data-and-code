# Introduction to Machine Learning (33002)
# Philip Waggoner (https://pdwaggoner.github.io/)

# Code for class (logistic regression and LDA; classification pt 2)

# load some libraries
library(tidyverse)
library(here)
library(patchwork)
library(tidymodels)

# load the 2016 ANES pilot study data
anes <- read_csv(here("data", "anes_pilot_2016.csv"))

# select some features and clean: party, and 2 fts
anes_short <- anes %>% 
  select(pid3, fttrump, ftobama) %>% 
  mutate(democrat = as.factor(ifelse(pid3 == 1, 1, 0)),
         fttrump = replace(fttrump, fttrump > 100, NA),
         ftobama = replace(ftobama, ftobama > 100, NA)) %>%
  drop_na()

anes_short <- anes_short %>% 
  select(-c(pid3)) %>% 
  relocate(c(democrat))

anes_short %>% 
  skimr::skim()


# model fitting via tidymodels
# define mod and engine
mod <- logistic_reg() %>% # this is the only thing that changes from last time
  set_engine("glm") %>% 
  set_mode("classification")

# fit 
logit <- mod %>% 
  fit(democrat ~ ., 
      data = anes_short)
logit 

# eval
logit %>% 
  predict(anes_short) %>% 
  bind_cols(anes_short) %>% 
  metrics(truth = democrat,
          estimate = .pred_class)

# predict and viz
logit %>% 
  predict(anes_short) %>% 
  mutate(model = "logit", 
         truth = anes_short$democrat) %>% 
  mutate(correct = if_else(.pred_class == truth, "Yes", "No")) %>% 
  ggplot() +
  geom_bar(alpha = 0.8, aes(correct, fill = correct)) + 
  labs(x = "Correct?",
       y = "Count",
       fill = "Correctly\nClassified") +
  theme_minimal()

# explore some of the output
library(broom)

tidy(logit)

# Predicted probabilities
dont_like_trump <- tibble(fttrump = 0:10,
                          ftobama = mean(anes_short$ftobama))

predicted_probs <- predict(logit, 
                           dont_like_trump, 
                           type = "prob")
# visualize results
dont_like_trump %>%
  bind_cols(predicted_probs) %>%
  ggplot(aes(x = fttrump, 
             y = .pred_1)) +
  geom_point() +
  geom_errorbar(aes(ymin = (.pred_1) - sd(.pred_1), 
                    ymax = (.pred_1) + sd(.pred_1)), 
                width = 0.2) +
  geom_hline(yintercept = 0.50, linetype = "dashed") +
  ylim(0, 1) +
  labs(x = "Feelings toward Trump",
       y = "Probability of Being a Democrat") + 
  theme_minimal()

# hmm... what happened?

dont_like_trump_love_obama <- tibble(fttrump = 0:10,
                                     ftobama = 90:100)

predicted_probs_new <- predict(logit, 
                               dont_like_trump_love_obama, 
                               type = "prob")
# visualize results
dont_like_trump_love_obama %>%
  bind_cols(predicted_probs_new) %>%
  ggplot(aes(x = fttrump, 
             y = .pred_1)) +
  geom_point() +
  geom_errorbar(aes(ymin = (.pred_1) - sd(.pred_1), 
                    ymax = (.pred_1) + sd(.pred_1)), 
                width = 0.2) +
  geom_hline(yintercept = 0.50, linetype = "dashed") +
  ylim(0, 1) +
  labs(x = "Feelings toward Trump",
       y = "Probability of Being a Democrat") + 
  theme_minimal()

# Cross validating a final, full model
# Logit via kNN approach with tidymodels from last class
## split
set.seed(1234)

split <- initial_split(anes_short,
                       prop = 0.70) 
train <- training(split)
test <- testing(split)

cv_train <- vfold_cv(train, 
                     v = 10)

## Now, create a recipe
recipe <- recipe(democrat ~ ., 
                 data = anes_short) 


# define mod and engine
mod <- logistic_reg() %>% # this is the only thing that changes from last time
  set_engine("glm") %>% 
  set_mode("classification")


# Define a workflow
workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(mod)

res <- workflow %>%
  fit_resamples(resamples = cv_train,
                metrics = metric_set(roc_auc, accuracy))


# finalize workflow and evaluate
final <- res %>% 
  select_best(metric = "accuracy")

workflow <- workflow %>%
  finalize_workflow(final)

final_mod <- workflow %>%
  last_fit(split) 


# inspect (if desired)
final_mod %>% 
  collect_predictions() 

final_mod %>%  
  collect_metrics() 

# create confusion matrix
final_mod %>% 
  collect_predictions() %>% 
  conf_mat(truth = democrat, 
           estimate = .pred_class,
           dnn = c("Pred", "Truth"))

# bar plot like last time
logit_plot <- final_mod %>% 
  collect_predictions() %>% 
  ggplot() +
  geom_bar(aes(x = .pred_class, 
               fill = democrat)) +
  facet_wrap(~ democrat) +
  labs(title = "From Logit Fit",
       x = "Predicted Party Affiliations", 
       fill = "Democrat") +
  theme_minimal()
logit_plot

# Finally, LDA

library(discrim)

mod <- discrim_linear() %>% # this is the only thing we are changing, again
  set_engine("MASS") %>% 
  set_mode("classification")


# Define a workflow
workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(mod)

res <- workflow %>%
  fit_resamples(resamples = cv_train,
                metrics = metric_set(roc_auc, accuracy))

# finalize workflow and evaluate
final <- res %>% 
  select_best(metric = "accuracy")

workflow <- workflow %>%
  finalize_workflow(final)

final_mod <- workflow %>%
  last_fit(split) 


# inspect (if desired)
final_mod %>% 
  collect_predictions() 

final_mod %>%  
  collect_metrics() 

# create confusion matrix
final_mod %>% 
  collect_predictions() %>% 
  conf_mat(truth = democrat, 
           estimate = .pred_class,
           dnn = c("Pred", "Truth"))

# bar plot like last time
lda_plot <- final_mod %>% 
  collect_predictions() %>% 
  ggplot() +
  geom_bar(aes(x = .pred_class, 
               fill = democrat)) +
  facet_wrap(~ democrat) +
  labs(title = "From LDA Fit",
       x = "Predicted Party Affiliations", 
       fill = "Democrat") +
  theme_minimal()


# side by side
library(patchwork)

logit_plot + lda_plot


# a quick tangent: a non-tidy approach for those less excited about the tidy approach (no judgement of course)
library(MASS)

set.seed(1234)

samples <- sample(nrow(anes_short), 
                      size = 0.8*nrow(anes_short))  

train <- anes_short[samples, ]
test <- anes_short[-samples, ]

lda <- lda(democrat ~ .,
           data = train)
lda

# some checks for accuracy
democrat <- test$democrat # set aside for ease

lda_pred <- predict(lda, test) 

# first few
data.frame(lda_pred)[1:5,]

# confusion matrix
table(lda_pred$class, democrat)

# accuracy rate
mean(lda_pred$class == democrat)

# density viz
true <- ggplot() +
  geom_density(aes(lda_pred$x, 
                   col = democrat),
               alpha = 0.5,
               linetype = "solid") +
  ylim(0.0, 0.9) +
  labs(title = "True Density") +
  theme_minimal() +
  theme(legend.position = "none")
true

both <- ggplot() +
  geom_density(aes(lda_pred$x, 
                   col = lda_pred$class),
               alpha = 0.5,
               linetype = "dashed") +
  geom_density(aes(lda_pred$x, 
                   col = democrat),
               alpha = 0.5,
               linetype = "solid") +
  ylim(0.0, 0.9) +
  labs(title = "True and Predicted Densities",
       caption = "Solid line = True density\nDashed line = Predicted density") +
  theme_minimal() +
  theme(legend.position = "none")
both

# side by side
true + both

#
