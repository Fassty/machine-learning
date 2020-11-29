#!/usr/bin/env Rscript
#################
# Pavel Mikulas #
# NPFL054       #
# HW #1         #
#################

# Load source data
source("load-mov-data.R")

#########
# TASK1 #
#########

assignPr <- function(...) {
	# Calculate the probability for each element in the table
	return(prop.table(table(...)))
}

calc_entropy <- function(...) {
	# Calculate the entropy of given set of vectors
	p <- assignPr(...) # To lazy to write another function comupting table(x) / NROW(x) over all arguments
	return( -sum(p * log2(p)) )
}

oc = examples$occupation
ra = examples$rating
H = calc_entropy(ra,oc) - calc_entropy(ra)

##########
# TASK 2 #
##########

wrap_words = function(x) {
	x = paste(strwrap(x, width=18), collapse='\n')
	return(x)
}

# Filter out the movie titles and ratings of the movies that were rated 67 times(columns 3 and 9)
movies_table = table(examples$movie)
movies_id = names(movies_table[movies_table == 67])
movies = examples[examples$movie %in% movies_id, c(3,9)]

# Wrap words so they would fit nicely below the graph
movies$title = sapply(movies$title, wrap_words)

# Group movies by title
movies$title = factor(movies$title,levels=unique(movies$title))

op = par(mar = c(10, 2, 2, 2))

boxplot(rating ~ title, movies, las=2, xlab="", ylab="", main='Movies rated 67 times')
points(x=1:NROW(table(movies$title)),
       y=aggregate(rating ~ title, movies, mean)$rating,
       col='red',
       pch=20)

par(op)


##########
# TASK 3 #
##########


# Add new columns to the data frame and fill them with 0's
users[c('ONE', 'TWO', 'THREE', 'FOUR', 'FIVE')] = 0

# Fill the new features from examples columns 2,3(user, rating)
# For each feature assign given frequency of ratings
apply(examples[,c(2,3)], 1, function(user) {
	offset = 5 # ONE,TWO,etc. features start at index 6
	uid = user[1]
	ra = offset + user[2]
	users[uid, ra] <<- users[uid, ra] + 1
})

# Count relative frequencies for each row and round them to 2 decimal places
users[6:10] = round(prop.table(as.matrix(users[6:10]), 1), 2)

# Perform hierarchical agglomerative clustering using average linkage
hc = hclust(dist(users[c(2, 6:10)]), method='average')
users$cluster = cutree(hc, 20) # Assign to users, for further aggregating

# Create a data frame for easily retrieving desired data
clusters = data.frame(
	population = as.vector(table(users$cluster)),
	age = aggregate(age ~ cluster, users, mean)$age
)

# To get all the duplicates we need to do duplicated(x) | duplicated(x, fromLast = TRUE)
# Otherwise we would just get one of each duplicates, not the full pairs
# Why does R work in such a mysterious ways I have no idea

# Common age, cluster and rating
d_AgeClRa = users[duplicated(users[c(2,6:11)]) | duplicated(users[c(2,6:11)], fromLast=T),]
# Common gender, cluster and rating
d_GenRaCl = users[duplicated(users[c(3,6:11)]) | duplicated(users[c(3,6:11)], fromLast=T),]
# Just common rating(found 3 so this code works, there are just no duplicates inside of clusters)
d_Ra = users[duplicated(users[c(6:10)]) | duplicated(users[c(6:10)], fromLast=T),]
# Common gender and rating
d_GenRa = users[duplicated(users[c(3,6:10)]) | duplicated(users[c(3,6:10)], fromLast=T),]
