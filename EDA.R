library(ggplot2)
library(tidyverse)
library(scales)
library(ggthemes)
library(GGally)
library(ggcorr)

train <- read_csv('F:/Nerdy Stuff/Kaggle/House prices/train.csv') %>% 
  select(-Id)

test <- read_csv('F:/Nerdy Stuff/Kaggle/House prices/test.csv')

g1 <- ggplot(train, aes(x = YrSold, y = SalePrice)) + 
  geom_jitter() + scale_y_continuous(labels = dollar)

g1 + theme_tufte() + labs(x = 'Year Sold', y = 'Sale price',
                          title = 'Does sale year impact sale price?') +
  geom_smooth(method = 'gam', se = FALSE)

g1 <- ggplot(train, aes(x = MoSold, y = SalePrice)) + 
  geom_jitter() + scale_y_continuous(labels = dollar) +
  scale_x_continuous(breaks = seq(1, 12, 1))

g1 + geom_smooth(method = 'gam', se=  FALSE) +
  theme_tufte() + labs(x = 'Month Sold', y = 'Sale price',
                       title = 'Does sale year impact sale price?')

train %>% 
  group_by(MSSubClass) %>% 
  summarise(avg_sale = mean(SalePrice, na.rm = TRUE),
            n = n())

g1 <- train %>% 
  select_if(is.numeric) %>% 
  select(SalePrice, 1:10) %>% 
  ggcorr(nbreaks = 5)
  
g1 + theme_tufte() + 
  labs(title = 'Correlation of the first ten numeric variables',
       subtitle = 'Including SalePrice')

g1 <- train %>% 
  select_if(is.numeric) %>% 
  select(SalePrice, 11:20) %>% 
  ggcorr(nbreaks = 5)

g1 + theme_tufte() +
  labs(title = 'Correlation of the second ten numeric variables',
       subtitle = 'Including SalePrice')

g1 <- train %>% 
  select_if(is.numeric) %>% 
  select(SalePrice, 21:30) %>% 
  ggcorr(nbreaks = 5)

g1 + theme_tufte() +
  labs(title = 'Correlation of the third ten numeric variables',
       subtitle = 'Including SalePrice')

g1 <- train %>% 
  select_if(is.numeric) %>% 
  select(SalePrice, 31:37) %>% 
  ggcorr(nbreaks = 5)

g1 + theme_tufte() +
  labs(title = 'Correlation of the last batch of numeric variables',
       subtitle = 'Including SalePrice')


train %>% 
  select_if(is.numeric)

glimpse(train)





