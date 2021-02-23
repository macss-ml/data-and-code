# Introduction to Machine Learning (33002)
# Philip Waggoner (https://pdwaggoner.github.io/)

# PCA

# load some libraries
library(tidyverse)
library(here)
library(skimr)
library(GGally)
library(tictoc)
library(factoextra)

# read in data
chocolate <- read_csv(here("data", "chocolates.csv"))

# select key features
choc_tibble <- chocolate[ ,4:14] %>% 
  as_tibble()

choc_tibble$Type <- as.factor(choc_tibble$Type) # factor for plotting

# take a look at the data
skim(choc_tibble)

ggpairs(choc_tibble, mapping = aes(col = Type)) + 
  theme_minimal()

# fit the PCA model
pca_fit <- select(choc_tibble, -Type) %>%
  scale() %>% 
  prcomp(); summary(pca_fit)

# PC loadings by hand
map_dfc(1:10, ~ pca_fit$rotation[, .] * sqrt(pca_fit$sdev^2)[.]) 

# Or simply extract...
pca_out <- get_pca(pca_fit)
print(pca_out$coord)

# viz: scree
fviz_screeplot(pca_fit, addlabels = TRUE, choice = "variance")

# viz: biplot
fviz_pca_biplot(pca_fit, label = "var")

# viz: loadings
fviz_pca_var(pca_fit)

# customized plot
data_for_plot <- choc_tibble %>%
  mutate(PC1 = pca_fit$x[, 1], 
         PC2 = pca_fit$x[, 2])

data_for_plot %>% 
  ggplot(aes(PC1, PC2, col = Type)) +
  geom_point() +
  stat_ellipse() +
  labs(title = "PCA Solution",
       subtitle = "Coloring by Chocolate Type",
       x = "PC1",
       y = "PC2") +
  theme_minimal()


# is scaling really *that* big of a deal?
pca_un <- select(choc_tibble, -Type) %>%
  prcomp(scale = FALSE)

summary(pca_un)

fviz_pca_biplot(pca_un, label = "var")

fviz_screeplot(pca_un, 
               addlabels = TRUE, 
               choice = "variance")

# 


##
## On your own if you're interested: calculating eigen vectors by hand, step by step 

# Suppose we had the following cartesian coordinates based on two features, X1 and X2, for our first observation: `(-1, 0.5)`. To find the linear equation for the line, then, we proceed in three steps. 

# 1. Substitute those point values in the Pythagorean theorem ($a^2 + b^2 = c^2$): `((-1)^2) + ((0.5)^2)`. The reason for this is we want to calculate the linear equation for our line/PC. But we don't have an intercept, so we have to do it on the basis of our input feature values alone.

# --> Doing this, then, gives a value of = `1.25`

# 2. Take the square root to get c (not c^2): 
sqrt(((-1)^2) + ((0.5)^2)) # which gives `1.118034`.

# 2.1 You can double check your math by writing a simple function:

p <- function(a, b){
  sqrt(a^2 + b^2)
}

p(-1, 0.5)

# 3. Then, divide each coordinate value by `c` (1.11) to allow the normalized value of `c` to equal 1, which is the distance to the origin. 

# --> Recall, this is a requirement of PCA, which is that the *PC must pass through the origin*. 

(-1)/1.118034 # a = -0.8944272
(0.5)/1.118034 # b = 0.4472136

# So the shape of our normalized right triangle (allowing side `c` (hypotenuse) to equal a distance of 1 from the origin) is:
# `c = 1`

# This gives the *linear* equation for our line, $-0.8944272 \times X1 + 0.4472136 \times X2$

# To check this, we can do the same thing for some other projected point on this line, where the solution should return the value of the hypotenuse (side c), yet the sign of which will indicate the direction of the line (which, again, is what we are interested in finding when we calculate eigenvectors). 

# Suppose, our second set of points is (2, -0.68)

# Before we start, as a baseline, let's plug in the `X1` and `X2` coordinates from our new set into our formula that we previously got, to give us a baseline:

(-0.8944272)*(2) + (0.4472136)*(-0.68) # -2.09296

# Now, we can just follow the steps:

# 1. Compute $a^2 + b^2$
((2)^2) + ((-0.68)^2) 

# 2. Calculate the hypotenuse (using our simple function above), and this time store this in object `c` to make life a little easier. 
p <- function(a, b){
  sqrt(a^2 + b^2)
}

(c <- p(2, -0.68)) # = 2.11

# 3. Divide each coordinate by c to get the "parameter" values for the line formula.
2/c # a
(-0.68)/c # b

# This then gives the formula: $0.9467727 \times X1 + -0.3219027 \times X2$. 

# Now, we can test it out with our first set of coordinates (-1, 0.5). 

0.9467727*(-1) + (-0.3219027)*0.5 # = -1.11 (which, recall, was the value of c from the first set of points above)

# Great! It worked such that both values from both equations for different projected points on the line indicate the slope of the PC is negative given the two negative values we calculated (-2.10 and -1.11). 

# The take away message here is that we can calculate the slope/direction of the PC by hand (which in this simple case is negative) with the vector of these normalized projected points onto the computed PC. The result is the eigenvector. 
