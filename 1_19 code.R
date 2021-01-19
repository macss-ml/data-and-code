# Introduction to Machine Learning (33002)
# Philip Waggoner (https://pdwaggoner.github.io/)

# Code for class

# kNN + ML Tasks

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
  glimpse()

# viz
anes_short %>% 
  ggplot(aes(fttrump, ftobama, 
             col = democrat)) +
  geom_point() +
  theme_minimal()

# obama density
o_d <- anes_short %>% 
  ggplot(aes(ftobama, 
             col = democrat)) +
  geom_density() +
  ggtitle("Obama") +
  theme_minimal()

# trump density
t_d <- anes_short %>% 
  ggplot(aes(fttrump, 
             col = democrat)) +
  geom_density() +
  ggtitle("Trump") +
  theme_minimal()

# side by side
o_d + t_d

# Change shapes and colors for more descriptive plots if you want
anes_short %>% 
  ggplot(aes(fttrump, ftobama, 
             shape = democrat)) +
  geom_point() +
  theme_minimal()

anes_short %>% 
  ggplot(aes(fttrump, ftobama, 
             shape = democrat,
             col = democrat)) +
  geom_point() +
  theme_minimal()

# scale continuous inputs
anes_scaled <- anes_short %>% 
  mutate_at(c("fttrump", "ftobama"), 
            ~(scale(.)))


## train the model on the full data (via tidymodels)
# define model type
mod <- nearest_neighbor() %>% 
  set_engine("kknn") %>% 
  set_mode("classification")

# fit 
knn <- mod %>% 
  fit(democrat ~ ., 
      data = anes_scaled)

# eval
knn %>% 
  predict(anes_scaled) %>% 
  bind_cols(anes_scaled) %>% 
  metrics(truth = democrat, 
          estimate = .pred_class)

# predict and viz
knn %>% 
  predict(anes_scaled) %>% 
  mutate(model = "knn", 
         truth = anes_scaled$democrat) %>% 
  mutate(correct = if_else(.pred_class == truth, "Yes", "No")) %>% 
  ggplot() +
  geom_bar(alpha = 0.8, aes(correct, fill = correct)) + 
  labs(x = "Correct?",
       y = "Count",
       fill = "Correctly\nClassified") +
  theme_minimal()


# k-fold CV
## first split
set.seed(1234)

split <- initial_split(anes_scaled,
                       prop = 0.70) 
train <- training(split)
test <- testing(split)

## CV 
cv_train <- vfold_cv(train, 
               v = 10)

cv_train


## Now, create a recipe to make things easier
recipe <- recipe(democrat ~ ., 
                 data = anes_scaled)  

# define model type from earlier, but with `k` addition
mod_new <- nearest_neighbor() %>% 
  set_args(neighbors = tune()) %>%  
  set_engine("kknn") %>% 
  set_mode("classification")

# This is just a way to keep things neat and tidy (pun intended)
workflow <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(mod_new)

# Now, we tune() instead of fit()
grid <- expand.grid(neighbors = c(1:25))

res <- workflow %>%
  tune_grid(resamples = cv_train, 
            grid = grid,
            metrics = metric_set(roc_auc, accuracy))

# inspect 
res %>% 
  collect_metrics(summarize = TRUE) 

res %>% 
  select_best(metric = "roc_auc") 

res %>% 
  select_best(metric = "accuracy")

# final/best
final <- res %>% 
  select_best(metric = "roc_auc")

workflow <- workflow %>%
  finalize_workflow(final)

# train and eval in one
final_mod <- workflow %>%
  last_fit(split) 

# inspect
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

# viz, of course
final_mod %>% 
  collect_predictions() %>% 
  ggplot() +
  geom_bar(aes(x = .pred_class, 
                   fill = democrat)) +
  facet_wrap(~ democrat) +
  labs(x = "Predicted Party Affiliations", 
       fill = "Democrat",
       caption = "Note: facets are ground truth\nFor '1' in truth of '0' (53), kNN predicted incorrectly\nVice verse for the other class (29), via confusion matrix") +
  theme_minimal()
