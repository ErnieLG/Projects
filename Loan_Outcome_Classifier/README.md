<img src="https://www.picpedia.org/highway-signs/images/loan.jpg" width="700">

# Loan Outcome Classifier

*Loans are one of the most crucial aspects of the American economy, but they come with the potential pitfall of loan recipients being unable to pay off these loans resulting in a loss in profit from the stance of the lender.  In order to minimize this risk and maximize their ROI, lending institutions in the past have approved loans and assigned risk grades (with corresponding interest rates and term lengths) based on a multitude of factors such as credit score, income, race, and value of collateral.  I aim to use modern techniques of data science and machine learning to create a model of classification that would streamline this process and improve upon extant loan grading systems.*


## 1. Data

[Data Import Report](./1.%20Import_Data/Import_Data.ipynb)

I'm using data provided by NathanGeorge on 1.3M observations of Lending Club (https://www.lendingclub.com/) accepted and declined loan requests from the beginning of 2007 to the end of 2018 with 151 features.  (https://www.kaggle.com/wordsforthewise/lending-club)  

Lending Club utilizes a loan risk rating system from A to G, with A having the greatest chance of being paid back in full and also having the lowest interest rate:

<img src="https://www.moneycrashers.com/wp-content/uploads/2015/04/reward-risk.png" width="700">


## 2. Data Cleaning 

[Data Cleaning Report](./2.%20Data_Cleaning/Data_Cleaning.ipynb)

* **Problem 1:** This dataset contains completely different features depending on whether the loan applicants were an individual person or multiple people.  
-**Solution:** These are essentially two datasets with data that can't be compared to each other, therefore we'll focus only on individual applicants.

* **Problem 2:** There are null values for features that describe the number of times an event happened (such as "Number of credit inquiries in past 12 months") or amount of time since a specific event occurred (such as "Months since most recent 90-day or worse rating").
-**Solution:** I interpretted these omissions as evidence of non-occurrence, since they were uncommon events; I filled these with 0 (for number of occurences) or the max value (for amount of time since last occurence).

* **Problem 3:** There were many features that were highly correlated or nearly identical, as well as many categorical features.  
-**Solution:** I was able to remove many features that didn't add any predictive value, and I applied one-hot encoding to the categories.


## 3. EDA

[EDA Report](./3.%20EDA/EDA.ipynb)

* LendingClub's grades aligned well with the charged off rates:

![](./6.%20Readme/outcome_by_grade.png)
![](./6.%20Readme/outcome_by_grade_table.png)

* There were a few strong correlations between features that were mostly expected:

First Feature | Second Feature | Correlation
------------ | ------------- | -------------
upper boundary of borrower’s FICO score | lower boundary of borrower’s FICO score | 1.000
number of open credit lines  | number of satisfactory accounts | .999
amount of loan  | amount of monthly payments | .954
months since oldest revolving account opened  | years between opening of first line of credit and loan issuance | .918
number of revolving accounts  | number of bankcard accounts | .838

* Being fully paid off is most correlated with an A or B grade and FICO score, while an A grade is most correlated with FICO score and a high credit limit:

![](./6.%20Readme/fully_paid_corr.png)![](./6.%20Readme/grade_A_corr.png)

* Being charged off is most correlated with interest rate, term length, and an E or D grade, while a G grade is most correlated interest rate, term length, and amount of monthly payments:

![](./6.%20Readme/charged_off_corr.png)![](./6.%20Readme/grade_G_corr.png)


## 4. Machine Learning & Modeling

[Modeling Report](./4.%20Modeling/Modeling.ipynb)

The most important metric to maximize is the true positive rate; it's uncommon for loans to be charged-off, but they represent a loss in profit, and we want to limit them as much as possible.  After attempting several models such as KNN, Random Forest, and Gradient Boosting, I was not getting TPRs much better than chance.  This led to me applying Bayesian Boosting to Light GBM, optimizing the hyperparameters in order to maximize the AUROC.  Comparing the various methods of LightGBM, Goss had the best preliminary results at 72.4%.  I performed it again for a total of 6 iterations with CV = 5.  

ROC Plot:

![](./6.%20Readme/roc_plot.png)

Adjusting the threshold to .5 allowed us to reach TPR = 68%.

![](./6.%20Readme/conf_matrix0.png)

Adjusting the threshold to .4 allowed us to reach TPR = 83%.

![](./6.%20Readme/conf_matrix1.png)


## 5. Future Improvements

[Slide Deck](./5.%20Presentation/slide_deck.pdf)

* In the future, I would like to analyze the data for loan applicants that were filed by more than one person.

* It would be interesting to try different types of hyper-parameters and more iterations of Bayesian Boosting.


## 6. Credits

Thanks to Tony Paek, my Springboard mentor. 
