library(ISLR)
library(caret)
library(rpart)
library(rpart.plot)
library(MLmetrics)
library(gmlnet)
library(ROCR)

set.seed(666)
train_size = 1000
train_ind = sample(seq_len(nrow(Caravan)), size = train_size)

# Split the data into train and test sets
d.train = Caravan[-train_ind, ]
d.test = Caravan[train_ind, ]

mosh = rbind(table(Caravan$MOSHOOFD), round(table(Caravan[, c(86, 5)])[2, ] / table(Caravan$MOSHOOFD) * 100 , 2))
row.names(mosh) = c("#customers", "caravan%")

most = rbind(table(Caravan$MOSTYPE), round(table(Caravan[, c(86, 1)])[2, ] / table(Caravan$MOSTYPE) * 100 , 2))
row.names(most) = c("#customers", "caravan%")

table(Caravan$MOSTYPE, Caravan$MOSHOOFD)

tC = trainControl(method = 'cv', 
                    number = 10,
                    classProbs = TRUE, 
                    summaryFunction = prSummary)

rpart.model = train(Purchase ~ ., data = d.train, 
                method = 'rpart',
                preProcess = c('center', 'scale'),
                tuneLength = 6,
                metric='AUC', 
                trControl=tC)

rf.model = train(Purchase ~ ., data = d.train, 
                method = 'rf',
                preProcess = c('center', 'scale'),
                metric='AUC', 
                trControl=tC)

lambda_grid <- seq(0.001, 0.1, 0.001)
alpha_grid <- seq(0.5,0.75,0.05)
objGrid <- expand.grid(alpha = alpha_grid, lambda = lambda_grid)

lr.model = train(Purchase ~ ., data = d.train, 
                method = 'glmnet',
                preProcess = c('center', 'scale'),
                tuneLength = 6,
                tuneGrid=objGrid,
                metric='AUC', 
                trControl=tC)

rpart.pred.raw = predict(rpart.model, newdata = d.test, type='prob')[, 2]
rpart.pred = prediction(rpart.pred.raw, d.test$Purchase)

rf.pred.raw = predict(rf.model, newdata = d.test, type='prob')[, 2]
rf.pred = prediction(rf.pred.raw, d.test$Purchase)

lr.pred.raw = predict(lr.model, newdata = d.test, type='prob')[, 2]
lr.pred = prediction(lr.pred.raw, d.test$Purchase)

#AUC
rpart.auc = performance(rpart.pred, measure = 'auc', fpr.stop=0.2)
rf.auc = performance(rf.pred, measure = 'auc', fpr.stop=0.2)
lr.auc = performance(lr.pred, measure = 'auc', fpr.stop=0.2)

#ROC curves
rpart.roc = performance(rpart.pred, measure = 'tpr', x.measure='fpr')
rf.roc = performance(rf.pred, measure = 'tpr', x.measure='fpr')
lr.roc = performance(lr.pred, measure = 'tpr', x.measure='fpr')

#Threshold
threshold = sort(lr.pred.raw, TRUE)[100]
best100 = which(lr.pred.raw >= threshold)

## Lasso
x = data.matrix(Purchase ~ ., data = d.train)
y = ifelse(d.train$Purchase == "Yes", 1, 0)

grid <- 10^seq(4, -2, length = 100)
lasso.fit = glmnet(x, y, alpha=1, family='gaussian', lambda=grid)
print(lasso.fit)
coef(lasso.fit)[98,]

# Features from decision tree
labels(rpart.model$finalModel)

# Random forest importance
imp = rf.model$finalModel$importance
best = which(imp %in% head(sort(imp, TRUE), 13))
imp[best,]


# Load blind test dataset and select 100 most promising
Test = read.csv('C:/Users/mikul/Downloads/caravan.test.1000.csv')
names(Test) = names(d.train[-86])
Test$Purchase = 0

Test.pred = predict(lr.model, newdata = Test, type='prob')[, 2]
threshold = sort(Test.pred, TRUE)[100]
best100 = which(Test.pred >= threshold)

Test[best100,]$Purchase = 1

# Print it out to a file
cat(Test$Purchase, file='T.prediction', sep='\n')