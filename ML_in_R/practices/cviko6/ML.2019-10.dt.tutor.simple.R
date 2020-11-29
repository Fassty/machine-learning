ntrees = c()
accs = c()
#for(i in 1:20) {
#	rf = randomForest(profits ~ category + sales + assets + marketvalue, forbes.train, ntree=i*50, mtry=4)
#	pred = predict(rf, forbes.test, type="class")
#	cfm = table(pred, forbes.test$profits)
#	ntrees[i] <- i*50
#	accs[i] <- sum(diag(cfm))/1000
#}
library(randomForest)
rf = randomForest(profits ~ category + sales + assets + marketvalue, forbes.train, ntree=100, cutoff=c(0.2, 0.8))
pred = predict(rf, forbes.test, type="prob", predict.all=T)


