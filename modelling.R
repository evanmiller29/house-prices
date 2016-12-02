install.packages("devtools")
devtools::install_github("wesm/feather/R")

library(feather)

basepath <- 'C:/Users/evanm_000/Documents/GitHub/house-prices'
setwd(basepath)

df <- read_feather("X_ttl.feather")
