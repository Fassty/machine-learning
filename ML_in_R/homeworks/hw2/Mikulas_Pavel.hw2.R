#!/usr/bin/env Rscript
#################
# Pavel Mikulas #
# NPFL054       #
# HW #2         #
#################

#########
# TASK1 #
#########
library(ISLR)

# Load data
dataset = Auto

# Encode n categorical features into n-1 dummy boolean column
# dataset$american = ifelse(dataset$origin == 1, 0, 0) # count: 245 will be used as default, because it's the most common
dataset$japanese = ifelse(dataset$origin == 2, 1, 0) # count: 68
dataset$european = ifelse(dataset$origin == 3, 1, 0) # count: 79

# Exclude name and origin columns from the training set
trainset = dataset[, c(1:7,10,11)]

# Train the model
lr_model = lm(mpg ~ ., data=trainset)
print(lr_model)


# For some reason I'm getting weird results when acceleration is not sorted
# I suppose that lm requires sorted input, although it's not explicitly
# stated in the documentation
sorted_data = trainset[order(trainset$acceleration),]

# Plot the acceleration and mpg pairs
plot(sorted_data$acceleration, sorted_data$mpg,
	xlab = 'Acceleration',
	ylab = 'Miles per gallon')

# So we can easily distinguish the different lines
colors = c('red', 'blue', 'green', 'purple', 'yellow')
# Perform polynomial regression for degrees 1 to 5 and plot the polynomial fits for each
for(i in 1:5) {
	m = lm(mpg ~ poly(acceleration, i, raw=T),data=sorted_data)
	x = sorted_data$acceleration
	y = m$fitted.values[order(sorted_data$acceleration)]
	lines(x, y, lwd=3, col = colors[i])
	print(sprintf("Deggre: %d, R^2: %.15f", i, summary(m)$r.squared))

}

# Add a legend for easier graph interpretation
legend("topleft", legend = c("model1: linear",
				  "model2: poly x^2",
				  "model3: poly x^3",
				  "model4: poly x^4",
				  "model5: poly x^5"),
	   col=colors,
	   lty=c(1,1,1),
	   bty="n",
	   cex=0.9)

# We could also compare the model accuracies with anova F-test


##########
# TASK 2 #
##########

# Add a feature indicating whether the value of mpg is above or below the mpg median
trainset$mpg01 = ifelse(trainset$mpg > median(trainset$mpg), 1, 0)

# Since we have encoded the mpg values into mpg01 we can now exclude the mpg column
d = trainset[-1]

# Calculate the entropy of mpg01
# Since the mpg01 data are binary I'll use log with base 2
p = prop.table(table(d$mpg01))
H = -sum(p * log2(p))
print(sprintf("Entropy: %.0f", H))

# Set random seed and randomly sample 80% indices
set.seed(666)
train_size = floor(0.8 * nrow(d))
train_ind = sample(seq_len(nrow(d)), size = train_size)

# Split the data into train and test sets
train = d[train_ind, ]
test = d[-train_ind, ]

#########
# TASK3 #
#########

# Trivial classifier
triv_c.mfv = as.numeric(tail(names(sort(table(train$mpg01))),1))
triv_c.acc = sum(test$mpg01 == triv_c.mfv) / nrow(test)
print(sprintf("Trivial classifier accuracy: %.2f%s", triv_c.acc * 100, "%"))

## Logistic regression
log_re = glm(mpg01 ~ .,
		family='binomial',
		data=train)

# Function to calculate model's performance - Precision, Recall, Specificity and F1-measure
perf = function(x) {
	TP = as.numeric(cfm[2,2])
	TN = as.numeric(cfm[1,1])
	FN = as.numeric(cfm[1,2])
	FP = as.numeric(cfm[2,1])

	Accuracy = (TP + TN) / (TP + TN + FP + FN)
	Precision = TP / (TP + FP)
	Recall = TP / (TP + FN)
	Specificity = TN / (TN + FP)
	F1_Measure = 2 * ((Precision * Recall) / (Precision + Recall))

	pf = data.frame(Accuracy, Precision, Recall, Specificity, F1_Measure)

	pf = round(pf, 4)
	pf = pf * 100
	print(sapply(pf, function(x) paste(c(x, "%"), collapse = "")))
}

# First we use 0.5 threshold for cutting the probabilities into two classes
# Calculate train error
log_re.train.prob = predict(log_re, newdata=train, type='response')
log_re.train.pred05 = ifelse(log_re.train.prob >= 0.5, 1, 0)
log_re.train.err = sum(log_re.train.pred05 != train$mpg01) / nrow(train)
print(sprintf("Logistic Regression train error: %.2f%s", log_re.train.err * 100, "%"))

# Calculate test error
log_re.test.prob = predict(log_re, newdata=test, type='response')
log_re.test.pred05 = ifelse(log_re.test.prob >= 0.5, 1, 0)
log_re.test.err = sum(log_re.test.pred05 != test$mpg01) / nrow(test)
print(sprintf("Logistic Regression test error: %.2f%s", log_re.test.err * 100, "%"))

# Print confusion matrix and retrieve false negatives and true positives
cfm = table(log_re.test.pred05, test$mpg01)
print(cfm)
perf(cfm)

# Threshold 0.1
log_re.test.pred01 = ifelse(log_re.test.prob >= 0.1, 1, 0)
cfm = table(log_re.test.pred01, test$mpg01)
print(cfm)
perf(cfm)

# Threshold 0.9
log_re.test.pred09 = ifelse(log_re.test.prob >= 0.9, 1, 0)
cfm = table(log_re.test.pred09, test$mpg01)
print(cfm)
perf(cfm)



# Print model summary
print(summary(log_re))

## Decision tree algorithm
library(rpart)
library(rpart.plot)

# Train the decision tree model with cp=NA so we can pick the best cp later
dst = rpart(mpg01 ~ ., data=train, method="class", cp=NA)

# Compute the train error rate
dst.train.confidence = predict(dst, newdata=train)[,2]
dst.train.pred = ifelse(dst.train.confidence >= 0.5, 1, 0)
dst.train.err = sum(dst.train.pred != train$mpg01) / nrow(train)
print(sprintf("Decision Tree algorithm train error: %.2f%s", dst.train.err * 100, "%"))

# Compute the test error rate
dst.test.confidence = predict(dst, newdata=test)[,2]
dst.test.pred = ifelse(dst.test.confidence >= 0.5, 1, 0)
dst.test.err = sum(dst.test.pred != test$mpg01) / nrow(test)
print(sprintf("Decision Tree algorithm test error: %.2f%s", dst.test.err * 100, "%"))

printcp(dst)
#plotcp(dst)
dst = rpart(mpg01 ~ ., data=train, method="class", cp=0.11)
# Compute the best cp model train error rate
dst.train.confidence = predict(dst, newdata=train)[,2]
dst.train.pred = ifelse(dst.train.confidence >= 0.5, 1, 0)
dst.train.err = sum(dst.train.pred != train$mpg01) / nrow(train)
print(sprintf("Decision Tree algorithm [BEST CP] train error: %.2f%s", dst.train.err * 100, "%"))

# Compute the best cp model test error rate
dst.test.confidence = predict(dst, newdata=test)[,2]
dst.test.pred = ifelse(dst.test.confidence >= 0.5, 1, 0)
dst.test.err = sum(dst.test.pred != test$mpg01) / nrow(test)
print(sprintf("Decision Tree algorithm [BEST CP] test error: %.2f%s", dst.test.err * 100, "%"))
#rpart.plot(dst)
