#!/usr/bin/env Rscript
fq <- function(x){
	c <- sum(sapply(x, as.numeric))
	return(c)
}
