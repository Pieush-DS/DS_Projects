#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Filtering out the warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Importing the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


# In[4]:


# To set the Row, Column and Width of the Dataframe to show on Jupyter Notebook
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)


# # <font color = blue> EDA Case Study </font>
# 
# We have two different dataset which having all the informations of the client at the time of application that whether a client has payment difficulties or not and another dataset is related to the previous loan data which contains the data whether the previous application had been approval, cancelled, refused or unused offer.  
# In this assignment, we will try to find some interesting insights if a client has difficulty paying their installments which may be used for taking action related to loan, using Python.

# ##  Task 1: Reading the data

# - ### Subtask 1.1: Read the Previous Application Data.
# 
# Read the EDA Previous Application data file provided and store it in a dataframe `EDA_Prev_App`.
# It contains the data whether the previous application had been approved, cancelled, refused or unused offer.
# 

# In[422]:


# Read the csv file using 'read_csv'.
EDA_Prev_App = pd.read_csv("F:/UpGrade/M_13_Case Study/Credit EDA Case Study-20210323T115611Z-001/Credit EDA Case Study/previous_application.csv")


# In[423]:


EDA_Prev_App.head() #Showing all the EDA Previous Application data with headers


# - ###  Subtask 1.2: Inspect the Dataframe
# 
# Inspect the dataframe for dimensions, null-values, and summary of different numeric columns.

# In[338]:


# Check the number of rows and columns in the dataframe
EDA_Prev_App.shape


# In[339]:


# Check the column-wise info of one dataframe Application
EDA_Prev_App.info(verbose=True)


# In[340]:


# To check Null values of columns
EDA_Prev_App_Null = EDA_Prev_App.isnull().sum()
EDA_Prev_App_Null


# In[424]:


# To reprsent the % or ratio of data imbalance in a dataframe 'Application_data'. 
EDA_Prev_App_Null_Per = round(100*(EDA_Prev_App.isnull().sum()/len(EDA_Prev_App.index)),3)
EDA_Prev_App_Null_Per


# In[425]:


# To check number of columns which has Null values greater than 30%
len(EDA_Prev_App_Null_Per[EDA_Prev_App_Null_Per.values>=(0.3)])


# In[343]:


# List all the columns which values is greater than and equal t0 30%
EDA_Prev_Null_Per_30 = list(EDA_Prev_App_Null_Per[EDA_Prev_App_Null_Per.values>=0.3].index)
EDA_Prev_Null_Per_30


# In[426]:


# Drop all the values which values is greater than equal to 30%.
EDA_Prev_App_Not_Null = EDA_Prev_App.drop(labels = EDA_Prev_Null_Per_30, axis=1, inplace=True)


# In[427]:


# To check rows and columns of a dataframe
EDA_Prev_App.shape


# In[346]:


# Get information of columns of the dataframe
EDA_Prev_App.info()


# In[347]:


# Recheck the Null values of a dataframe with the remaining columns
EDA_Prev_App.isnull().sum()


# In[348]:


# To check % of variations of the updated dataframe columns
round(100*(EDA_Prev_App.isnull().sum()/len(EDA_Prev_App.index)),4)


# In[428]:


# To check the details of the Null column with all the rows
EDA_Prev_App[EDA_Prev_App['PRODUCT_COMBINATION'].isnull()]


# In[429]:


# To count the values in a column
EDA_Prev_App['PRODUCT_COMBINATION'].value_counts()


# In[430]:


# PRODUCT_COMBINATION column has very minimum Nul values, therefore those rows can be removed
EDA_Prev_App = EDA_Prev_App[~EDA_Prev_App.PRODUCT_COMBINATION.isnull()]


# In[431]:


# To recheck Null values for 'PRODUCT_COMBINATION'column
EDA_Prev_App.PRODUCT_COMBINATION.isnull().sum()


# In[432]:


# To count the values in a column
EDA_Prev_App['PRODUCT_COMBINATION'].value_counts()


# In[433]:


# Recheck the dataframe if still there is any null values
EDA_Prev_App.isnull().sum()


# In[355]:


EDA_Prev_App.shape # To check the Shape of the dataframe


# In[356]:


# To check the information of the updated dataframe columns
EDA_Prev_App.info()


# In[434]:


# To count values for the particular column
EDA_Prev_App['NAME_CONTRACT_TYPE'].value_counts()


# In[435]:


# To count values for the particular column
EDA_Prev_App['NAME_PORTFOLIO'].value_counts(normalize=True) 


# In[436]:


# To count values for the particular column
EDA_Prev_App['NAME_PRODUCT_TYPE'].value_counts(normalize=True)


# In[437]:


# Get selected useful columns of 'Application_data' in a new dataframe for analysis
EDA_Prev_App_list = EDA_Prev_App.loc[:,["SK_ID_PREV","SK_ID_CURR", "NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS","AMT_APPLICATION","AMT_CREDIT", "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START", "NAME_CASH_LOAN_PURPOSE","DAYS_DECISION","NAME_PAYMENT_TYPE", "CODE_REJECT_REASON", "NAME_CLIENT_TYPE", "NAME_PORTFOLIO", "NAME_PRODUCT_TYPE","CHANNEL_TYPE","NAME_SELLER_INDUSTRY","NAME_YIELD_GROUP","PRODUCT_COMBINATION"]]


# In[438]:


EDA_Prev_App_list.head() # To show various columns of the New Dataframe


# In[439]:


# To make absolute value for 'DAYS_DECISION' column
EDA_Prev_App_list['DAYS_DECISION'] = EDA_Prev_App_list['HOUR_APPR_PROCESS_START'].abs()
EDA_Prev_App_list['DAYS_DECISION'] = EDA_Prev_App_list['DAYS_DECISION'].abs()


# In[440]:


# To check Null values 
EDA_Prev_App_list.isnull().sum()


# In[441]:


# To get information of the dataframe
EDA_Prev_App_list.info()


# In[442]:


# To count and check values in a column
EDA_Prev_App_list['NAME_CONTRACT_TYPE'].value_counts(normalize=True)


# In[443]:


# To count and check values in a column
EDA_Prev_App_list['NAME_CONTRACT_STATUS'].value_counts(normalize=True)


# In[444]:


# To count and check values in a column
EDA_Prev_App_list['WEEKDAY_APPR_PROCESS_START'].value_counts(normalize=True)


# In[447]:


# To count and check values in a column
EDA_Prev_App_list['NAME_CASH_LOAN_PURPOSE'].value_counts(normalize=True)


# #### The above object column has few annoying values like 'XAP', 'XNA', which doesn't have any meaning. These values has very high % in the table, therefore better to remove both of them.

# In[448]:


# To replace 'XNA', 'XAP' from another highest value of the respective column
EDA_Prev_App_list['NAME_CASH_LOAN_PURPOSE'] = EDA_Prev_App_list['NAME_CASH_LOAN_PURPOSE'].replace(['XAP'],'Repairs')
EDA_Prev_App_list['NAME_CASH_LOAN_PURPOSE'] = EDA_Prev_App_list['NAME_CASH_LOAN_PURPOSE'].replace(['XNA'],'Other')


# In[449]:


# To count and check values in a column
EDA_Prev_App_list['NAME_CASH_LOAN_PURPOSE'].value_counts(normalize=True)


# In[450]:


# To count and check values in a column
EDA_Prev_App_list['NAME_PAYMENT_TYPE'].value_counts(normalize=True)


# In[451]:


# To replace 'XNA' from another highest value of the respective column
EDA_Prev_App_list['NAME_PAYMENT_TYPE'] = EDA_Prev_App_list['NAME_PAYMENT_TYPE'].replace(['XNA'],'Cash through the bank')


# In[452]:


# To count and check values in a column
EDA_Prev_App_list['NAME_PAYMENT_TYPE'].value_counts(normalize=True)


# In[453]:


# To count and check values in a column
EDA_Prev_App_list['CODE_REJECT_REASON'].value_counts(normalize=True)


# In[454]:


# To replace 'XAP' with highest available value after 'XAP' 
EDA_Prev_App_list['CODE_REJECT_REASON'] = EDA_Prev_App_list['CODE_REJECT_REASON'].replace(['XAP'],'HC')


# In[301]:


# To remove those unwanted rows of 'XNA'
EDA_Prev_App_list = EDA_Prev_App_list.drop(EDA_Prev_App_list.loc[EDA_Prev_App_list['CODE_REJECT_REASON']=='XNA'].index)


# In[455]:


# To count and check values in a column
EDA_Prev_App_list['CODE_REJECT_REASON'].value_counts(normalize=True)


# In[456]:


# To count and check values in a column
EDA_Prev_App_list['NAME_CLIENT_TYPE'].value_counts(normalize=True)


# In[457]:


# To drop unwanted rows
EDA_Prev_App_list = EDA_Prev_App_list.drop(EDA_Prev_App_list.loc[EDA_Prev_App_list['NAME_CLIENT_TYPE']=='XNA'].index)


# In[458]:


# To count and check values in a column
EDA_Prev_App_list['NAME_CLIENT_TYPE'].value_counts(normalize=True)


# In[459]:


# To count and check values in a column
EDA_Prev_App_list['NAME_PORTFOLIO'].value_counts(normalize=True)


# In[460]:


# To drop unwanted rows
EDA_Prev_App_list['NAME_PORTFOLIO'] = EDA_Prev_App_list['NAME_PORTFOLIO'].replace(['XNA'],'POS')

#EDA_Prev_App_list = EDA_Prev_App_list.drop(EDA_Prev_App_list.loc[EDA_Prev_App_list['NAME_PORTFOLIO']=='XNA'].index)


# In[461]:


# To count and check values in a column
EDA_Prev_App_list['NAME_PORTFOLIO'].value_counts(normalize=True)


# In[463]:


# To count and check values in a column
EDA_Prev_App_list['NAME_PRODUCT_TYPE'].value_counts(normalize=True)


# In[464]:


# To replace 'XNA' from another highest value of the respective column
EDA_Prev_App_list['NAME_PRODUCT_TYPE'] = EDA_Prev_App_list['NAME_PRODUCT_TYPE'].replace(['XNA'],'x-sell')


# In[465]:


# To count and check values in a column
EDA_Prev_App_list['NAME_PRODUCT_TYPE'].value_counts(normalize=True)


# In[466]:


# To count and check values in a column
EDA_Prev_App_list['CHANNEL_TYPE'].value_counts(normalize=True)


# In[467]:


# To count and check values in a column
EDA_Prev_App_list['NAME_SELLER_INDUSTRY'].value_counts(normalize=True)


# In[468]:


# To replace 'XNA' from another highest value of the respective column
EDA_Prev_App_list['NAME_SELLER_INDUSTRY'] = EDA_Prev_App_list['NAME_SELLER_INDUSTRY'].replace(['XNA'],'Consumer electronics')


# In[469]:


# To count and recheck values in a column
EDA_Prev_App_list['NAME_SELLER_INDUSTRY'].value_counts(normalize=True)


# In[470]:


# To count and check values in a column
EDA_Prev_App_list['NAME_YIELD_GROUP'].value_counts(normalize=True)


# In[471]:


# To replace 'XNA' from another highest value of the respective column
EDA_Prev_App_list['NAME_YIELD_GROUP'] = EDA_Prev_App_list['NAME_YIELD_GROUP'].replace(['XNA'],'middle')


# In[472]:


# To count and check values in a column
EDA_Prev_App_list['NAME_YIELD_GROUP'].value_counts(normalize=True)


# In[473]:


# To count and check values in a column
EDA_Prev_App_list['PRODUCT_COMBINATION'].value_counts(normalize=True)


# In[474]:


EDA_Prev_App_list.head() # To see all the columns again if there is any issue in the column values


# ### Do Univeriate analysis 

# In[604]:


# To analyze 'TARGET' column whether loan defaulters are more or less ('1' = loan defaulter/did late payments, '0'= for all other case )
fig = plt.figure(figsize=(6,5))
EDA_Prev_App_list['NAME_CONTRACT_STATUS'].value_counts().plot.barh()
plt.title("Various Contract Status Counts\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nCount\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
# plt.xticks(rotation=90)
plt.ylabel("Contract Status",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences:
# 
# Inferences 1: According to the above Bargraph 'Approved' loan has highest number among all the four Contract Status whereas 'Unused offer' is very less.

# In[476]:


# To analyze 'TARGET' column whether loan defaulters are more or less ('1' = loan defaulter/did late payments, '0'= for all other case )
fig = plt.figure(figsize=(6,5))
EDA_Prev_App_list['NAME_CONTRACT_TYPE'].value_counts().plot.barh()
plt.title("Contract Types\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nCount\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
# plt.xticks(rotation=90)
plt.ylabel("Contract Types\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences:
# 
# Inferences 1: According to the above Bar-Graph maximum Cash loan and Consumer loan having negligible difference and are highest in number, whereas 'Revolving loans' are very less.

# In[477]:


# To show the Types of Education
fig = plt.figure(figsize=(6,5))
EDA_Prev_App_list['NAME_CLIENT_TYPE'].value_counts().plot.barh()
plt.title("Clinet Types for loan\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nCounts\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.ylabel("Client's Type\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences:
# 
# Inferences 1: According to the above Bargraph maximum client type is 'Repeater for loan, whereas 'Refreshed' are very less.

# ### Bivariate/Multivariate Analysis

# In[478]:


# To analyze various income types in relation with the Gender 
fig = plt.figure(figsize=(7,5))
sns.barplot(data=EDA_Prev_App_list, x='NAME_CONTRACT_TYPE', y='AMT_APPLICATION', hue='NAME_SELLER_INDUSTRY', ci=None)
plt.title("Contract Type Vs Application Amount in terms of Seller Industry\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nContract Type\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Application Amount\n(INR)\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences
# Inferences 1: From above bargraph it is clear that among all the categories of Seller Industry Clothing has highest Application Amount for Cash loan Type Contract.
# 
# Inferences 2: Consumer loans for Tourism Seller Industry has highest Application Amount whereas it is less for Revolving loans.

# In[512]:


EDA_Prev_App_list.head()


# #### To do Bivariate/Multivariate Analysis

# In[603]:


# To show Bar Graph
fig = plt.figure(figsize=(40,35)) 
fig1 = fig.add_subplot(3, 3, 1)    # For adding subplot side by side
sns.barplot(data=EDA_Prev_App_list, x='NAME_PRODUCT_TYPE', y='AMT_APPLICATION', hue='NAME_CONTRACT_STATUS', ci=None, ax=fig1)
plt.title("Product Type Vs Application Amount\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nName Product Type \n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Total Amount of Income\n(INR)\n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})

fig2 = fig.add_subplot(3, 3, 2)    # For adding subplot side by side
sns.barplot(data=EDA_Prev_App_list, x='NAME_PRODUCT_TYPE', y='AMT_CREDIT', hue='NAME_CONTRACT_STATUS', ci=None, ax=fig2)
plt.title("Product Type Vs Credit Amount\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nName Product Type \n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("\n\n\nTotal Credit Amount\n(INR)\n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences
# Inferences 1: Canceled Contracts for 'Walk-in' Product Type is highest for higher Income Amount and Credit Amount, whereas it is very less for X-Sell Product Type.
# 
# Inferences 2: Contract Status as a Unused Offer for 'Walk-in' Product Type is lowest or almost nil for both the amounts.

# In[601]:


# To show Bar Graph between Week days, Hours in terms of Contract Type.
fig = plt.figure(figsize=(10,7)) 
sns.barplot(data=EDA_Prev_App_list, x='WEEKDAY_APPR_PROCESS_START', y='HOUR_APPR_PROCESS_START', hue='NAME_CONTRACT_TYPE', ci=None)
plt.title("Weekdays Vs Hours in terms of Contract Type\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nWeekdays\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Approval Process Hours\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences
# Inferences 1: Approval Process Hours are highest for each day for all kind of Contract Types whether it is Consumer loans, Cash loans or Revolving loans.
# 
# Inferences 2: Approval Process Hours are lowest for Cash loans for any day of the week.

# In[552]:


# To show Bar Graph
fig = plt.figure(figsize=(10,7)) 
sns.barplot(data=EDA_Prev_App_list, x='NAME_YIELD_GROUP', y='AMT_APPLICATION', hue='NAME_CONTRACT_STATUS', ci=None)
plt.title("Yield Group Vs Application Amount interms of Contract Status\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nYield Group\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Application Amount\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences
# Inferences 1: Cancelled Application are higher for Higher Application amount for the three different Yield Groups i.e. Low_Action, Low_Normal, and High whereas for Middle type of Yield Group it is very-2 less.
# 
# Inferences 2: It means Cancel rate is very less minimum amount in Middle Group.

# In[605]:


# Box plot - Amount (Credit/Loan Amount) Vs Contract Status.
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(35,30)) 
fig1 = fig.add_subplot(3, 3, 1)    # For adding subplot side by side
sns.boxplot(data=EDA_Prev_App_list, x='NAME_CONTRACT_STATUS', y='AMT_APPLICATION', ax=fig1) 
plt.title("Contract Status Vs Application Amount\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nContract Status\n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Total Amount of Income\n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})

fig2 = fig.add_subplot(3, 3, 2)    # For adding subplot side by side
sns.boxplot(data=EDA_Prev_App_list, x='NAME_CONTRACT_STATUS', y='AMT_CREDIT', ax=fig2)
plt.title("Contract Status Vs Credit Amount\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nContract Status\n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("\n\n\nTotal Credit Amount\n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences
# Inferences 1: Distribution of values for all kind of Contract Status has almost similar kind of distribution for both the Amounts whether it is Total Amount or Credit Amount.
# 
# Inferencces 2: For Refused Contract, distribution is unequal for both the amount whether it is Total Incomme Amount or Credit Amount.

# In[606]:


# Box plot - Amount (Credit/Loan Amount) Vs Product Type for various yield groups.
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(35,30)) 
fig1 = fig.add_subplot(3, 3, 1)    # For adding subplot side by side
sns.boxplot(data=EDA_Prev_App_list, x='NAME_PRODUCT_TYPE', y='AMT_APPLICATION', hue='NAME_YIELD_GROUP', ax=fig1) 
plt.title("Product Type Vs Application Amount\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nName Product Type \n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Total Amount of Income\n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})

fig2 = fig.add_subplot(3, 3, 2)    # For adding subplot side by side
sns.boxplot(data=EDA_Prev_App_list, x='NAME_PRODUCT_TYPE', y='AMT_CREDIT', hue='NAME_YIELD_GROUP', ax=fig2)
plt.title("Product Type Vs Credit Amount\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nName Product Type \n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("\n\n\nTotal Credit Amount\n",fontdict={'fontsize':18,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences
# Inferences 1: Distribution of values for both the Product Types (X-Sell, Walk-in) has almost similar for all kind of Yiled Groups whether it is Total Amount or Credit Amount.
# 
# Inferences 2: For low_action Yield Group distribution is equal for product type 'Walk-in' whether it is Total income Amount or Total Credit Amount.

# In[506]:


# To find out Correlation among various variables of dataset
EDA_Prev_App_list_corr = EDA_Prev_App_list.iloc[:,2:]
EDA_Prev_App_list_2 = EDA_Prev_App_list_corr.corr(method='pearson', min_periods=1)


# In[589]:


# To show Correlation data table with heads
EDA_Prev_App_list_corr.head()


# In[508]:


# To show Correlation Chart
EDA_Prev_App_list_2


# In[607]:


# Heatmap to show coorelation among all the numeric variables.
fig = plt.figure(figsize=(7,5))
sns.heatmap(EDA_Prev_App_list_2, cmap="RdYlGn", annot=True)
plt.title("Correlation between Various Amounts & Durations\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nVarious Amount's & Durations\n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Various Amount's & Durations\n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.show()


# ### Inferences
# 
# Inference 1: Application Amount is inversaly proportional to the no. of Days Decision and required total Approval hours. It means higher Application Amount is sanctioned in less no. of days. and Application Process start in less no. of hours.
# 
# Inference 2: Credit Amount is also inversaly proportional to the no. of Days Decision and required total Approval hours. It means higher Credited Amount is sanctioned in less no. of days. and Application Process start in less no. of hours.

# In[610]:


# Heatmap to show relation among various parameters.
fig = plt.figure(figsize=(25,20)) 
fig1 = fig.add_subplot(3, 3, 1)    # For adding subplot side by side
Prev_Income_Bucket = EDA_Prev_App_list.groupby('NAME_CONTRACT_TYPE')['AMT_APPLICATION','AMT_CREDIT'].mean()
sns.heatmap(Prev_Income_Bucket, cmap="RdYlGn", annot=True, ax=fig1)
plt.title("Contract Type Vs Various Amounts\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nVarious Amount's \n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Contract Type\n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})

fig2 = fig.add_subplot(3, 3, 2)    # For adding subplot side by side
Prev_Income_Bucket = EDA_Prev_App_list.groupby('NAME_CONTRACT_TYPE')['DAYS_DECISION','HOUR_APPR_PROCESS_START'].mean()
sns.heatmap(Prev_Income_Bucket, cmap="RdYlGn", annot=True, ax=fig2)
plt.title("Contract Type Vs Approval Hours, Days\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nHours & Days for Approval",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("\nContract Type\n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.show() 


# ### Inferences
# 
# Inference 1: Cash loan are highest for Amount Credited and Application Amount. 
# 
# Inference 2: Revolving loans has very less value for Amount Credited.
# 
# Inference 3: Revolving loans approved in very less no. of days and it's Process also started in minimum hours, whereas Cash loans are taking more no. of days to approve and more hours to start the process.
# 
# Inference 4: Overall Revolving loans are taking less hours, minimum no. of days in decision but their loan amount and Credited amount is also less.

# ### End of the Analysis for the 'Previous Application-Data' set

# 
# 
# 
