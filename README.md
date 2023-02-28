Rossmann Drug Store Sales Prediction 

Machine Learning Project Report 

Soumabha Sarkar


Abstract—Our project attempted to apply various machine learning techniques to a real-world problem of predicting drug store sales. Rossmann operates over 3,000 drug stores in 7 European countries and has provided past sales information of 1115 Rossmann stores located across Europe. We preprocessed, feature-engineered the data, and examined different statistical / machine learning analysis for forecasting sales of each store. Then, we  compared the methods’ predictive power by computing Root Mean Square Percentage Error (RMSPE). 

I. INTRODUCTION 

The objective of this project is to predict 6 weeks of daily sales for 1115 Rossmann stores located across Germany Using  the data which was provided by Rossmann through Almabetter. The motivation of this project is the following:  by reliably predicting daily sales, store managers may  decrease operational costs, and create effective staff  schedules (ex. by assigning more staff during busy days). Also, we would like to identify which type of techniques are  both efficient and effective in a real-world sales prediction task. 

II. DATA 

Data sets are given by Rossmann through Kaggle. There  are three files: “Rossmann Store data.csv”, “Store.csv”, and  Colab notebook containing 1001599 observed daily sales of  different stores from 2013 to 2015. “store.csv” contains  supple- mentary information for each store (1115 lines 
because there are 1115 stores). “Rossmann Store data.csv” and “store.csv” both contain a Store id that can be used to join  the data sets. Finally, “Rossmann Store data.csv” has  1115 lines of daily sales of different stores but the value of Sales column is omitted.We are expected to predict the value of Sales column for the test set (“Rossmann Store data.csv”) and store data (“store.csv”). The following table  describes fields : 

Store: unique id for each store (positive integer ranging from 1 to 1115)

DayOfweek: the day of week (positive integer ranging from 1 to 7 where 1, 2, ..., 7 corresponds to Monday, Tuesday, ...,Sunday)

Date: the date (string of format YYYYMMDD)

Sales:the total amount of revenue for this given day

Customers:the number of customers on the given day

Open:an indicator of whether the store was open on the given day (0 if closed, 1 if open)

Promo:an indicator of whether the store was running a promotion on the given day

StateHoliday:an indicator of whether the given day was a state holiday (0 if not a state holiday, a if public holiday, b if eastern holiday, c if christmas)

SchoolHoliday: an indicator of whether the store was affected by a nearby school holiday on the given day (0 if not affected, 1if affected)



A. Preprocessing 

• We first merged “Rossmann Store data.csv” and  “store.csv” by “Store” because we can predict daily sales better with more data related to the sales. We also  merged “Rossmann Store data.csv” and “store.csv” by “Store”.

• We splitted “Date” field into three fields: “year”, “month”, and “day” to better account for the effects each component of date has.

• We computed average sales for each store to create a new field “AverageSales”. This variable seemed to be an  indicator of how well a store will perform in future. This  makes sense because a store with a strong past performance is likely to perform well in future. 

B. Exploratory Data Analysis 

I. Correlation Heatmap between features. 
II. Stores are mainly open on which day of the week to analyze. 
III. Different year wise sales data analysis. 
IV. Different year wise customer data analysis. 
V. Sales affected by school holidays or not. 
VI. Distribution of different types of stores analysis. VII. How store assortment type is influencing sales. 
VIII. How promotion is influencing sales. 

a) Heat Map To check any distinctive relationships  between variables before applications of prediction  models, we drew a correlation coefficient plot. 

Fig. 1. Correlation Matrix Analysis of the columns mainly revolved around getting its relationship with Sales’. 

b) “DayOfWeek” and “Sales” are negatively correlated (-0.46). This implies that, as “DayOfWeek” approaches on Sunday, sales would decrease because drugstores  would mostly close on Sunday. 

Fig. 2. Day ofWeek 

• As can be seen in the chart above, there are peaks in the average sales and average no. of customers. 

c) Different year wise sales data analysis. 

 Fig. 3. Year wise Analysis 
 
The top image is showing the mean values of sales year wise  and the bottom image is showing the actual sale values year wise. 

d) Different year wise customer data analysis 

Fig. 4 Year  Customer Analysis 

The top image is showing the mean values of number of  customers year wise and the bottom image is showing the actual customer values year wise.

e) Sales Affected by the School Holiday of customers on state holidays is plotted, but only for open stores in the figure below. It is evident that when a store is open on holidays, it attracts more customers. 

g) How store assortment type is influencing sales 

Fig. 5. Sales Affected by School Holiday

• Since most stores are closed on public holidays and 

Fig. 7 . assortment type closed stores can be ignored when making predict- tions, the average sales figures and average number 

f) Distribution of different types of stores analysis 

Fig. 6 store type 
Type A stores are having 54.0% market share while Type B and Type C stores are having market share of 31.2% and 13.3% respectively. 
Assortment is a collection of goods or services that a business provides to customers. B type stores provide maximum service in every  assortment type store. Between them the extended service is the highest. 

C. Feature Selection 

Fig.8 Feature Selection

Among numerical columns, Sales and Customers are heavily  skewed and from correlation heat maps it was seen that the Customers column is causing multicollinearity. So we do not take the Customer column at the time of feature selection. 
As the Sales column is also skewed, we will apply natural logarithm to decrease the skewness. 

❖ LINEAR REGRESSION: 

This model is a rudimentary first look into large-scale trends. We did not expect it to capture the  granular movements of the sales numbers and indeed it  didn’t, reporting an average RMSE of 37.31%. We  observe that there are minimal inter-year trends and therefore can safely disregard them in future considerations. 

We are using a linear regression model so, we need to check 4 basic assumptions of linear regression. 

1. There should be linear relationship between independent and dependent variables.

2. The sum of residuals/error should be near to 0. 3. There should not be multicollinearity. 

3. There should not be heteroscedasticity. 
Linear Regression 

Ridge Regression 

• Ridge Regression is a model tuning method that  is used to analyze any data that suffers from  multicollinearity. This method performs L2 regularization.  When the issue of multicollinearity occurs, least-squares are unbiased, and variances are large, this results in predicted  values being far away from the actual values. 

• Lasso Regression is also called the Penalized regression  method. This method is usually used in machine learning for  the selection of the subset of variables. It provides greater  prediction accuracy as compared to other regression  models. Lasso Regularization helps to increase model  interpretation. 
Lasso Regression

❖ XgBoost 

The main component of XGBoost is its tree ensembles which sum up the predictions of multiple trees. The boost- ing  procedure in XGBoost is an additive process where a trained  model is added to the prediction at each step. In our implementation, we used the RMSE as the loss function for optimization. 

The XGB Regressor gives better RMSE and R Square values compared to other previous models. It is showing higher accuracy compared to other models. 

CONCLUSION: 

The Rossmann store sales prediction is a very engrossing data science  problem to solve. We noticed that the problem is more concentrated  towards the feature engineering and the feature selection part than on  model selection. We had to spend around 60-70% of our time on  analyzing data for trends in order to make our feature selection easier. 

We have also applied regularization techniques like Lasso, Ridge to Avoid Overfitting. We also used XGBRegressor to make predictions that have better performance than any single model. 

Most important feature came out to be customers, where sales are directly related to the number of customers. We performed cross validation using the XGB Regressor model which gave us 0.09593 as a minimum value that is lower than any other models.

Then we trained the XGBRegressor to the whole dataset and predicted the data for the future. That was all about our project.
