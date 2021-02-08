# Regression Trees

For this first part, we will fit a basic regression tree using the `AmesHousing` data (canned; no need to load externally). We will also use the `rpart` package to fit the tree predicting home `Sale_Price`, and then the `rpart.plot` package to create a nice, relatively clean plot. As a side exercise, consider trying to clean up and customize the `rpart.plot()` we make below. Hint: start by running `?rpart.plot` to see some options. Consider also Googling to see what others have done with the package. 

```{r}
library(tidymodels) # for data splitting
library(rpart) # for model fitting
library(rpart.plot) # for plotting the rpart output

set.seed(1234)

ames_split <- initial_split(AmesHousing::make_ames(), 
                            prop = .7)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

reg_tree <- rpart(
  formula = Sale_Price ~ .,
  data    = ames_train,
)

rpart.plot(reg_tree)
```

# Random Forests and Boosting

Now, let's use the `Boston` housing data from `MASS` to explore random forests and boosting. It gives housing values and other statistics in each of 506 suburbs of Boston based on a 1970 census.

## Random Forests

Random forests build lots of "bushy trees", and then average them to reduce the variance.

```{r}
library(randomForest) # for model fitting
library(MASS) # for the data

set.seed(1234)

# create the training set the "manual" way
train <- sample(1:nrow(Boston),300)
```

Now, fit a random forest and see how well it performs. We will use the response `medv`, the median housing value (measured in thousands of US dollars).

```{r}
rf <- randomForest(medv ~ .,
                   data = Boston,
                   subset = train)
```

Here we get some useful output, e.g., % of variance explaining in `medv` by our tree. Not too bad. 

To see how the training error flattens out as the number of trees grows, we can `plot()` this:

```{r}
plot(rf, 
     main = "Random Forest Results\nError Over Number of Trees")
```

## Boosted Regression Trees

Unlike random forests, each new tree in boosting tries to patch up the deficiencies of the current ensemble by learning and correcting for mistakes made by prior models.

```{r}
library(gbm)

boosted_mod <- gbm(medv ~ ., # predict median home vals
                    data = Boston[train, ], # use the training set from the Boston housing data
                    distribution = "gaussian", # assume a Gaussian process for basic regression problem
                    n.trees = 10000, # B from the lecture notes
                    shrinkage = 0.01, #lambda from the lecture notes
                    interaction.depth = 4) # d from the lecture notes
```

Now, we can make prediction on the test set over a range of values for our hyperparameters. 

Remember, with boosting, the number of trees is a tuning parameter, and if we have too many we can overfit (which is not the case in RFs and Bagging). 

So, let's allow the number of trees to range to find the optimal model, with the lowest est error.

```{r}
# set up sequence over numberof trees from min of 100 to max of 10,000, by 100, which gives us 100 models to fit
tree_num <- seq(from = 100, 
                to = 10000, 
                by = 100)

# make predictions
preds <- predict(boosted_mod, # use our trained model from above to base preds
                 newdata = Boston[-train,], # use indexing to pass the test set to predict() (this is, not equal to the training set via "-")
                 n.trees = tree_num) # and use our trees range to calculate test error at each version with a different number of trees

# store the MSE/generalization error in a more "manual" way, which is read: "using the test set that we just made predictions with, calculate the difference between predicted values of median home price compared to the true median home price for all observations in the testing set, square that value, and then take the mean, which give us the "MSE" or the "mean squared error""
mse <- with(Boston[-train,], apply((preds - medv)^2, 2, mean))

plot(tree_num, # "X", the number of trees used to make predictions
     mse, # "Y", the MSE
     type = "b", # both = points connected by a line
     pch = 19, # shape, filled in circles
     cex = 0.5, # size of points
     # and of course, some plot labels
     main = "Generalization/Test Error from Gradient Boosting",
     ylab = "Mean Squared Error", 
     xlab = "Number of Trees")
```

So there you go: looks like the optimal number of trees is around 1700-2000, where the error starts to flatten out. 

As an exercise, and to practice your `ggplot2`/tidyverse skills, try creating the plot we just made in base R, but using `ggplot()` syntax. It's not too hard, and would look a lot better. Give it a try!

