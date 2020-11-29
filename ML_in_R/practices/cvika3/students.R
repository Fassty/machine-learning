examples = read.csv("https://stats.idre.ucla.edu/wp-content/uploads/2016/02/sample.csv", header=T)
num.examples = nrow(examples)

examples$hon = factor(examples$hon, levels = c(0,1))

num.train = round(0.9 * num.examples)
num.test = num.examples - num.train

set.seed(123)
s = samples(num.examples)

id.train = s[1:num.train]
train = examples[id.train,]
id.test = s[(num.train)+1:num.examples]
