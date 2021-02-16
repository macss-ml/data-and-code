###
### k-means step-by-step
###
#
# The steps are:
#   1. initialization: randomly assign obs to clusters
#   2. assignment: assign each obs to the clostest cluster in space
#   3. optimization: optimize clusters to calculate new centroids based on the mean of assigned objects 
#
# Iterate until clusters converge and no changes occur, suggesting local minimum is found


# First, create some clustered data and set k
k <- 3 

set.seed(345)

M1 <- matrix(round(runif(100, 1, 5), 1), ncol = 2) # read, total from x to y, split into two columns (& round to 1 decimal place)
M2 <- matrix(round(runif(100, 7, 12), 1), ncol = 2)
M3 <- matrix(round(runif(100, 15, 20), 1), ncol = 2)
M <- rbind(M1, M2, M3)

# Define starting values
C <- M[1:k, ] # define initial centroids (to be assigned/optimized)
n <- length(M) / 2 # searching for coordinates of points; keep in two columns
A <- sample(1:k, n, replace = TRUE) # assign objects to clusters, k, at random
colors <- amerika::amerika_palette(3, 
                                   name = "Dem_Ind_Rep3", 
                                   type = "discrete") # define cluster colors

# Define plotting helper function
kmeans_plot <- function(M, C, title) {
  plot(M, main = title, xlab = "", ylab = "")
  for(i in 1:k) {
    points(C[i, , drop = FALSE], pch = 19, cex = 1.5, col = colors[i])
    points(M[A == i, , drop = FALSE], col = colors[i])    
  }
}
kmeans_plot(M, C, "initialization")

# Now, run the k-means algorithm and repeat until no more changes to cluster assignments
# Be sure to walk back through the several plots this next chunk creates to see this in action - just hit the back arrow in the plot pane viewer
repeat {
  # calculate Euclidean distance between objects and the cluster centroids
  D <- matrix(data = NA, nrow = k, ncol = n)
  for(i in 1:k) {
    for(j in 1:n) {
      D[i, j] <- sqrt((M[j, 1] - C[i, 1])^2 + (M[j, 2] - C[i, 2])^2)
    }
  }
  O <- A

  ## assignment: centroids are fixed, clusters are optimized
  A <- max.col(t(-D)) # assign objects to centroids based on max value
  if(all(O == A)) break # if no change to clusters: stop
  kmeans_plot(M, C, "assignment") # update the plot
  
  ## optimization: clusters are fixed, centroids are optimized
  # determine new centroids based on mean of assigned objects
  for(i in 1:k) {
    C[i, ] <- apply(M[A == i, , drop = FALSE], 2, mean)
  }
  kmeans_plot(M, C, "optimization")
}


# As a check, we can compare our result with the k-means function in Base R
kmeans_base_R <- kmeans(M, 
                        centers = k)

kmeans_plot(M, kmeans_base_R$centers, "Comparison via Base R") # now plot our data, M, but with the base R centers


# numeric comparison of centroid coordinates
(ours <- C[order(C[ , 1]), ])

(base_r <- kmeans_base_R$centers[order(kmeans_base_R$centers[, 1]), ])

round(base_r - ours, 5) # no difference five decimal places out
