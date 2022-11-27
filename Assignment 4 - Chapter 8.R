install.packages("vctrs")
library(caret)
library(ISLR)
library(magrittr)
library(dplyr)
library(tibble)
library(ggplot2)
library(tree)
library(randomForest)
library(gbm)
library(gam)
data(Carseats)

# Problem 8
# a.

set.seed(1)
training = sample(1:nrow(Carseats), nrow(Carseats)/2)

training_set = Carseats[training, ]
test_set = Carseats[-training, ]

# b.
car.tree = tree(Sales ~ ., data = training_set)
plot(car.tree)
text(car.tree, pretty=1, cex=0.7)

# Predicting the MSE
summary(car.tree)
pred = predict(car.tree, newdata = test_set)
mean((pred - test_set$Sales)^2) # Gives us an MSE of 4.922

# c.
set.seed(1)

cv_tree_model <- cv.tree(car.tree, K = 10)

data.frame(n_leaves = cv_tree_model$size,
           CV_RSS = cv_tree_model$dev) %>%
  mutate(min_CV_RSS = as.numeric(min(CV_RSS) == CV_RSS)) %>%
  ggplot(aes(x = n_leaves, y = CV_RSS)) +
  geom_line(col = "grey55") +
  geom_point(size = 2, aes(col = factor(min_CV_RSS))) +
  scale_x_continuous(breaks = seq(1, 17, 2)) +
  scale_y_continuous(labels = scales::comma_format()) +
  scale_color_manual(values = c("deepskyblue3", "green")) +
  theme(legend.position = "none") +
  labs(title = "Carseats Dataset - Regression Tree",
       subtitle = "Selecting the complexity parameter with cross-validation",
       x = "Terminal Nodes",
       y = "CV RSS")

# Given the outcome of the above, the optimal level of tree complexity seems to be 18 terminal nodes.

pruned = prune.tree(car.tree, best=18)
plot(pruned)
text(pruned)

pred = predict(pruned, newdata=test_set)
mean((pred - test_set$Sales)^2)

# Given that our optimal level is 18, the size of a complete tree. The pruned data set does not provide any better MSE (4.922).

# d.
set.seed(1)

bt_model = randomForest(training_set$Sales, data = training_set[, -1], mtry = ncol(training_set) - 1,
                        importance = TRUE)

pred = predict(bt_model, test_set)
mean((pred - test_set$Sales)^2) 

# Returns mean of 1.73, lower than both the standard and pruned trees.

importance(bt_model)

# Our most important variables when determining if a carseat will sell are the Price and Shelving location
# Relative to those two most other predictors are significantly less likely to effect the likelihood of a sale.

# e.

test_MSE = c()
i = 1

for (Mtry in 1:10) {
  set.seed(1)
  
  rf_model = randomForest(y = training_set$Sales, x = training_set[, -1], mtry = Mtry,
                          importance = TRUE)

  pred = predict(rf_model, test_set)
  
  test_MSE[i] = mean((pred - test_set$Sales)^2) 
  i = i + 1
}

data.frame(mtry = 1:10, test_MSE = test_MSE) %>%
  mutate(min_test_MSE = as.numeric(min(test_MSE) == test_MSE)) %>%
  ggplot(aes(x = mtry, y = test_MSE)) +
  geom_line(col = "grey55") +
  geom_point(size = 2, aes(col = factor(min_test_MSE))) +
  scale_x_continuous(breaks = seq(1, 10), minor_breaks = NULL) +
  scale_color_manual(values = c("deepskyblue3", "green")) +
  theme(legend.position = "none") +
  labs(title = "Carseats Dataset - Random Forests",
       subtitle = "Selecting 'mtry' using the test MSE",
       x = "mtry",
       y = "Test MSE")

importance(rf_model) %>%
  as.data.frame() %>%
  rownames_to_column("varname") %>%
  arrange(desc(IncNodePurity))

tail(test_MSE, 1) # returns 2.60

# The test MSE decreases with each iteration of m, intuitively this would make mtry 10 our lowest test MSE, however for this example we
# got 9 as our lowest. There might be some discrepancy between our seeding for randomForest tests as opposed to our previous bagged trees model.
# Again, we find that Price, and shelf location are the two largest predictors for MSE, however with the recent test, we can see that there is 
# a significant increase in  CompPrice as a %IncMSE predictor.

# Problem 10

# a.
data("Hitters")

Hitters_comp = Hitters %>% 
  na.omit() %>% 
  mutate(Salary = log(Salary))

ggplot(Hitters_comp, aes(x = Salary)) + 
  geom_density(fill = "deepskyblue3") + 
  theme(axis.text.y = element_blank(), 
        axis.ticks.y = element_blank(), 
        axis.title.y = element_blank()) + 
  labs(title = "Hitters Dataset - log(Salary) Distribution", 
       x = "log(Salary)")

# b.
training = 1:200
hitters_train = Hitters_comp[training, ]
hitters_test = Hitters_comp[-c(training), ]

# c.
shrink_lambda = 10^seq(-6., 0, 0.1)

train_MSE = c()
test_MSE = c()

for (i in 1:length(shrink_lambda)) {
  boost_TEMP = gbm(Salary ~ ., 
                    data = hitters_train, 
                    distribution = "gaussian", 
                    n.trees = 1000, 
                    interaction.depth = 2, 
                    shrinkage = shrink_lambda[i])
  
  train_MSE[i] = mean((predict(boost_TEMP, hitters_train, n.trees = 1000) - hitters_train$Salary)^2)
  
  test_MSE[i] = mean((predict(boost_TEMP, hitters_test, n.trees = 1000) - hitters_test$Salary)^2)
}

data.frame(lambda = shrink_lambda, train_MSE) %>%
  ggplot(aes(x = lambda, y = train_MSE)) + 
  geom_point(size = 2, col = "deepskyblue3") + 
  geom_line(col = "grey55") + 
  scale_x_continuous(trans = 'log10', breaks = 10^seq(-6, 0), labels = 10^seq(-6, 0), minor_breaks = NULL) + 
  labs(x = "Lambda (Shrinkage)", 
       y = "Training MSE")

# d.
set.seed(1)

data.frame(lambda = shrink_lambda, test_MSE) %>%
  ggplot(aes(x = lambda, y = test_MSE)) + 
  geom_point(size = 2, col = "deepskyblue3") + 
  geom_line(col = "grey55") + 
  scale_x_continuous(trans = 'log10', breaks = 10^seq(-6, 0), labels = 10^seq(-6, 0), minor_breaks = NULL) + 
  labs(x = "Lambda (Shrinkage)", 
       y = "Test MSE")


# e

# Boosting
set.seed(1)

custom_regression_metrics <- function (data, lev = NULL, model = NULL) {
  c(RMSE = sqrt(mean((data$obs-data$pred)^2)),
    Rsquared = summary(lm(pred ~ obs, data))$r.squared,
    MAE = mean(abs(data$obs-data$pred)), 
    MSE = mean((data$obs-data$pred)^2),
    RSS = sum((data$obs-data$pred)^2))
}

ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3, verboseIter = F, summaryFunction = custom_regression_metrics)

gbm_grid <- expand.grid(n.trees = 1000, interaction.depth = 2, n.minobsinnode = 10, shrinkage = 10^seq(-6, 0, 0.1))

boosting_model <- train(Salary ~ ., 
                   data = hitters_train, 
                   method = "gbm",
                   distribution = "gaussian", 
                   verbose = F,
                   metric = "MSE",
                   maximize = F,
                   trControl = ctrl, 
                   tuneGrid = gbm_grid)


boosting_model$results %>%
  rename(CV_MSE = MSE) %>%
  mutate(min_CV_MSE = as.numeric(shrinkage == boosting_model$bestTune$shrinkage)) %>%
  ggplot(aes(x = shrinkage, y = CV_MSE)) + 
  geom_line(col = "grey55") + 
  geom_point(size = 2, aes(col = factor(min_CV_MSE))) + 
  scale_x_continuous(trans = 'log10', breaks = 10^seq(-6, 0), labels = 10^seq(-6, 0), minor_breaks = NULL) + 
  scale_color_manual(values = c("deepskyblue3", "green")) + 
  theme(legend.position = "none") + 
  labs(title = "Hitters Dataset - Boosting", 
       subtitle = "Selecting shrinkage parameter using cross-validation",
       x = "Lambda (Shrinkage)", 
       y = "CV MSE")


boosting_model$bestTune$shrinkage %>% round(4) ## Optimal value equaling 0.0079 or ~0.008
cv_boosting = min(boosting_model$results$MSE) ## 0.2078 cross-validation MSE

mean((predict(boosting_model, hitters_test) - hitters_test$Salary)^2) #MSE equalling ~ 0.2878

# Linear regression:

linear_model = lm(Salary ~ ., data = hitters_train)
lmMSE = mean((predict(linear_model, hitters_test) - hitters_test$Salary)^2) # 0.492 

# Linear regression returns an MSE of 0.492, which is significantly higher than boosting model.

# Partial least squares regression
set.seed(1)

ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 3, verboseIter = F, summaryFunction = custom_regression_metrics)

model_pls <- train(Salary ~ ., 
                   data = hitters_train, 
                   method = "pls", 
                   preProcess = c("center", "scale"), 
                   metric = "MSE", 
                   maximize = F,
                   trControl = ctrl, 
                   tuneGrid = data.frame(ncomp = 1:20))


model_pls$results %>%
  rename(CV_MSE = MSE) %>%
  mutate(min_CV_MSE = as.numeric(ncomp == model_pls$bestTune$ncomp)) %>%
  ggplot(aes(x = ncomp, y = CV_MSE)) + 
  geom_line(col = "grey55") + 
  geom_point(size = 2, aes(col = factor(min_CV_MSE))) + 
  scale_x_continuous(breaks = 1:20, minor_breaks = NULL) +
  scale_color_manual(values = c("deepskyblue3", "green")) + 
  theme(legend.position = "none") + 
  labs(title = "Hitters Dataset - Partial Least Squares", 
       subtitle = "Selecting number of principal components with cross-validation",
       x = "Principal Components", 
       y = "CV MSE")

min(model_pls$results$MSE) %>% round(4) 
# Our optimal # of Principal components is 1, the MSE of this value is .4042 which is again higher than the MSE of the boosting method.
# Below are the listed values of our 3 MSE values.
#
# Boosting: 0.2878
# Linear Regression: 0.492
# PLS: 0.4042

# f.

boosted_model = gbm(Salary ~ ., 
                    data = hitters_train, 
                    distribution = "gaussian", 
                    n.trees = 1000, 
                    interaction.depth = 2, 
                    shrinkage = shrink_lambda[i])

summary(boosted_model)

# Given the above, the most significant predictors of salary are CHits, PutOuts, and CHmRun. The first being the most significant of the 
# # by a factor of 2.

# g.
bagged_hit = randomForest(Salary~., data=hitters_train, mtry = ncol(hitters_train) - 1,
                          importance = TRUE)
bagged_pred = predict(bagged_hit, hitters_test)
mean((bagged_pred - hitters_test$Salary)^2)

# The above gives us an MSE of 0.2279, this performs better than all of the previous models.