library(e1071)
library(scales)
library(ISLR)
library(gridExtra)

# Chapter 9 problem 5

# a.
set.seed(1)
x1 = runif(500) - 0.5
x2 = runif(500) - 0.5
y = 1*(x1^2 - x2^2 > 0)

df = data.frame(x1, x2, y = factor(y))

# b.
plot(x1[y == 0], x2[y == 0], col = c("red", "red"), xlab = "X1", ylab = "X2", pch = 19)
points(x1[y == 1], x2[y == 1], col = c("blue", "blue"), pch = 19)

# c.
glm_model = glm(formula = y ~ ., family = binomial)
summary(glm_model) ## Neither X1 or X2 are significant predictors 

# d.

df = data.frame(x1, x2, y)
linear_prob = predict(glm_model, df, type='response')
linear_pred = ifelse(linear_prob < 0.5, 1, 0)

plot(df[linear_pred == 1, ]$x1, df[linear_pred == 1, ]$x2, col="blue", ylab = 'x2', xlab = 'x1', pch= 19)
points(df[linear_pred == 0, ]$x1, df[linear_pred == 0, ]$x2, col="red", pch= 19)

## We can see with the color labeling above, that linear decision boundaries are not sufficient when applied to the data set.
## We are seeing a linear prediction, which overall does not fit the data as it is displayed in the original graph.

# e.
lrm_model = glm(y ~ x1 + x2 + I(x1^2) + I(x2^2), family = binomial)
df$pll = ifelse(predict(lrm_model, type='response') > 0.50, 1, 0)
summary(lrm_model)

# f
plot(df[df$pll == 1, ]$x1, df[df$pll == 1, ]$x2, col="blue", ylab = 'x2', xlab = 'x1', pch= 19)
points(df[df$pll == 0, ]$x1, df[df$pll == 0, ]$x2, col="red", pch= 19)

sum(df$y == df$pll)/nrow(df)

# The above gives a much more accurate reading of the chart, the decision boundaries are fitting to that of the original model
# According to the accuracy check this is actually a perfect predictor of the data.

# g
svm_model = svm(factor(y) ~ x1 + x2, df, kernal = 'linear', cost = 1)
svm_pred = predict(svm_model, df)

plot(df[svm_pred == 1, ]$x1, df[svm_pred == 1, ]$x2, col="blue", ylab = 'x2', xlab = 'x1', pch= 19)
points(df[svm_pred == 0, ]$x1, df[svm_pred == 0, ]$x2, col="red", pch= 19)

sum(df$y == svm_pred)/nrow(df)

## The svm model is slightly less accurate than the logistic regression model (0.966 .vs. perfect score) seemingly there is a higher
## bias toward the class labels where the x1 label is classified as above.

# h.
svm_model = svm(factor(y) ~ x1 + x2, df, gamma = 1)
svm_pred = predict(svm_model, df)

plot(df[svm_pred == 1, ]$x1, df[svm_pred == 1, ]$x2, col="blue", ylab = 'x2', xlab = 'x1', pch= 19)
points(df[svm_pred == 0, ]$x1, df[svm_pred == 0, ]$x2, col="red", pch= 19)

sum(df$y == svm_pred)/nrow(df)

## Finally, our non-linear kernal data svm is slightly more accurate than the before, however it is not as accurate as the linear regression model (0.972 .vs. 0.966 .vs. perfect score)

# i.
# Much of the commentary is seen above, outside of the predictions using the logistic regression model, the svm application seems to do sufficient
# when predicting the original data set.

