f = function(x) {
	1.2 * (x-2)**2 + 3.2
}

grad = function(x) {
	2.4 * (x-2)
}

p = (0:400)/100

plot(p, f(p), type = "l")

theta1 = 0.1
alpha = 0.8
iter = 100

theta1s <- theta1
f.theta1s <- f(theta1)

for (step in 1:iter){
	theta1 <- theta1 - alpha * grad(theta1)
	theta1s <- c(theta1s, theta1)
	f.theta1s <- c(f.theta1s, f(theta1))
}

lines (theta1s, f.theta1s, type = "b", col = "blue")


