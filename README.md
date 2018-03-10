# house-prices

Code for exploring data, building machine learning models and generating submissions for the Kaggle Zestimate challenge.
Using model stacking of results presented here and other scripts resulted in a 274th place (top 8%).

See here for more information: https://www.kaggle.com/c/zillow-prize-1

**Methodology:**

**Feature Engineering**
- Time element (month / year) was largely omitted from this approach
- Only a small amount of feature engineering was utilized in this approach
  - It became apparent quickly in the competition that feature engineeing would be of limited importance

**Model building**
- Utilized Gradient Boosting Machines (broad and specific hyper-params) for initial predictions
- Wider range of low level models included in model stacking with StackNet (https://github.com/kaz-Anova/StackNet)

