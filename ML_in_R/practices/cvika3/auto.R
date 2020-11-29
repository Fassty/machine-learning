library(ISLR)
summary(auto)

origins = c('USA', 'Europe', 'Japan')
Auto$origin

attach(Auto)

op = par(mfrow=c(2,2))

plot(table(cylinders), ylab = 'count')
plot(table(year), ylab = 'count')
plot(table(origin), ylab = 'count')

message("\n\n####### Correlation of weight and mpg")
cor(Auto[, c('weight', 'displacement', 'horsepower', 'acceleration')])
