#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Filtering out the warnings
import warnings
warnings.filterwarnings('ignore')


# In[5]:


# Importing the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


# In[6]:


# To set the Row, Column and Width of the Dataframe to show on Hupyter Notebook
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)


# # <font color = blue> EDA Case Study </font>
# 
# We have two different dataset which having all the informations of the client at the time of application that whether a client has payment difficulties or not and another dataset is related to the previous loan data which contains the data whether the previous application had been approval, cancelled, refused or unused offer.  
# In this assignment, we will try to find some interesting insights if a client has difficulty paying their installments which may be used for taking action related to loan, using Python.

# ##  Task 1: Reading the data

# - ### Subtask 1.1: Read the Application Data.
# 
# Read the EDA Application data file provided and store it in a dataframe `EDA_App`.
# It is about whether a client has payment difficulties.
# 
# Read the EDA Previous Application data file provided and store it in a dataframe `EDA_Prev_App`.
# It contains the data whether the previous application had been approved, cancelled, refused or unused offer.
# 

# In[7]:


# Read the csv file using 'read_csv'.
EDA_App = pd.read_csv("F:/UpGrade/M_13_Case Study/Credit EDA Case Study-20210323T115611Z-001/Credit EDA Case Study/application_data.csv")


# In[8]:


EDA_App.head() #Showing all the EDA Application data with headers


# - ###  Subtask 1.2: Inspect the Dataframe
# 
# Inspect the dataframe for dimensions, null-values, and summary of different numeric columns.

# In[9]:


# Check the number of rows and columns in the dataframe
EDA_App.shape


# In[10]:


# Check the column-wise info of one dataframe Application
EDA_App.info(verbose=True)


# In[11]:


# To check Null values of columns
EDA_App_Null = EDA_App.isnull().sum()
EDA_App_Null


# In[12]:


# To reprsent the % or ratio of data imbalance in a dataframe 'Application_data'. 
EDA_App_Null_Per = round(100*(EDA_App.isnull().sum()/len(EDA_App.index)),3)
EDA_App_Null_Per


# In[13]:


# To check number of columns which has Null values greater than 30%
len(EDA_App_Null_Per[EDA_App_Null_Per.values>=(0.3)])


# In[14]:


# List all the columns which values is greater than and equal t0 30%
EDA_Null_Per_30 = list(EDA_App_Null_Per[EDA_App_Null_Per.values>=0.3].index)
EDA_Null_Per_30


# In[15]:


# Drop all the values which values is greater than equal to 30%.
EDA_App_Not_Null = EDA_App.drop(labels = EDA_Null_Per_30, axis=1, inplace=True)


# In[16]:


# To check rows and columns of a dataframe
EDA_App.shape


# In[17]:


# Recheck the Null values of a dataframe with the remaining columns
EDA_App.isnull().sum()


# In[857]:


# Get information of columns of the dataframe
EDA_App.info()


# In[18]:


# To check % of variations of the updated dataframe columns
round(100*(EDA_App.isnull().sum()/len(EDA_App.index)),4)


# In[19]:


# To check null values of a column
EDA_App.AMT_ANNUITY.isnull().sum()


# In[20]:


# To check null values of a column
EDA_App.AMT_GOODS_PRICE.isnull().sum()


# In[21]:


# To check null values of a column
EDA_App.CNT_FAM_MEMBERS.isnull().sum()


# In[22]:


# To check null values of a column
EDA_App.EXT_SOURCE_2.isnull().sum()


# In[23]:


# To check null values of a column
EDA_App.DAYS_LAST_PHONE_CHANGE.isnull().sum()


# In[24]:


# To check the details of the Null column with all the rows
EDA_App[EDA_App['AMT_ANNUITY'].isnull()]


# In[25]:


# To count the values in a column
EDA_App['AMT_ANNUITY'].value_counts()


# In[26]:


# To describe the column
round(EDA_App.AMT_ANNUITY.describe(),2)


# In[27]:


# To find outliers for the column 'AMT_ANNUITY'
plt.figure(figsize = [10,3])
sns.boxplot(EDA_App["AMT_ANNUITY"])
plt.show()


# In[28]:


# Find out the quantile (0.25, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99, 1) of 'AMT_ANNUITY' column
EDA_App["AMT_ANNUITY"].quantile([0.25, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99, 1])


# #### The above result is showing that 'AMT_ANNUITY' column has outliers and the maximum outliers lies between 0.99 and 1 quantile. Therefore we should impute the value of NaN values by taking median of this column.

# In[29]:


# To find Median of 'AMT_ANNUITY' Column
AMT_median = round(EDA_App['AMT_ANNUITY'].median(), 2)
AMT_median


# In[30]:


# To replace null value of 'AMT_ANNUITY' with median
EDA_App['AMT_ANNUITY'] = EDA_App['AMT_ANNUITY'].fillna(AMT_median)


# In[31]:


# To recheck Null values for 'AMT_ANNUITY'column
EDA_App.AMT_ANNUITY.isnull().sum() 


# In[32]:


# CNT_FAM_MEMBERS column has very minimum nul values, therefore those rows can be removed
EDA_App = EDA_App[~EDA_App.CNT_FAM_MEMBERS.isnull()]


# In[33]:


# To recheck Null values for 'CNT_FAM_MEMBERS'column
EDA_App.CNT_FAM_MEMBERS.isnull().sum()


# In[34]:


# DAYS_LAST_PHONE_CHANGE column has very minimum nul values, therefore those rows can be removed
EDA_App = EDA_App[~EDA_App.DAYS_LAST_PHONE_CHANGE.isnull()]


# In[35]:


# To recheck Null values for 'DAYS_LAST_PHONE_CHANGE'column
EDA_App.DAYS_LAST_PHONE_CHANGE.isnull().sum()


# In[36]:


# To find outliers for the column 'AMT_GOOD_PRICE'
plt.figure(figsize = [10,3])
sns.boxplot(EDA_App["AMT_GOODS_PRICE"])
plt.show()


# In[37]:


# Find out the quantile (0.5, 0.7,00.9, 0.95, 0.99, 1) of 'AMT_ANNUITY' column
EDA_App["AMT_GOODS_PRICE"].quantile([0.25, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99, 1])


# In[38]:


# To find mean of 'AMT_GOODS_PRICE' 
Goods_mean = round(EDA_App['AMT_GOODS_PRICE'].mean(), 2)
Goods_mean


# In[39]:


# Repalce null values from mean value 
EDA_App['AMT_GOODS_PRICE'] = EDA_App['AMT_GOODS_PRICE'].fillna(Goods_mean)


# In[40]:


# To recheck Null values for 'AMT_GOODS_PRICE'column
EDA_App.AMT_GOODS_PRICE.isnull().sum()


# In[41]:


# To find outliers for the column 'EXT_SOURCE_2'
plt.figure(figsize = [10,3])
sns.boxplot(EDA_App["EXT_SOURCE_2"])
plt.show()


# In[42]:


# Find out the quantile (0.5, 0.7,00.9, 0.95, 0.99, 1) of 'EXT_SOURCE_2' column
EDA_App["EXT_SOURCE_2"].quantile([0.25, 0.5, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99, 1])


# In[43]:


# Find out mean for "EXT_SOURCE_2"
Ext_mean = round(EDA_App['EXT_SOURCE_2'].mean(), 2)
Ext_mean


# In[44]:


# Repalce null values from mean value 
EDA_App['EXT_SOURCE_2'] = EDA_App['EXT_SOURCE_2'].fillna(Ext_mean)


# In[45]:


# To recheck Null values for 'EXT_SOURCE_2'column
EDA_App.EXT_SOURCE_2.isnull().sum()


# In[46]:


# Recheck the dataframe if still there is any null values
EDA_App.isnull().sum()


# In[47]:


EDA_App.shape


# In[48]:


# To check the information of the updated dataframe columns
EDA_App.info()


# In[52]:


# Get selected useful columns of 'Application_data' in a new dataframe for analysis
EDA_App_list = EDA_App.loc[:,["SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE","CODE_GENDER", "CNT_CHILDREN", "AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY", "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE","DAYS_BIRTH","DAYS_EMPLOYED","CNT_FAM_MEMBERS","ORGANIZATION_TYPE"]]


# In[53]:


EDA_App_list.head() # To show various columns of the New Dataframe


# In[54]:


# To check the shape of the new dataframe
EDA_App_list.shape   


# In[55]:


# To get column-wise information of the new dataframe
EDA_App_list.info()


# In[56]:


# To check Null values in new dataframe's column
EDA_App_list.isnull().sum()


# In[57]:


# For column wise description of new dataframe
round(EDA_App_list.describe(), 2)


# In[58]:


# To count and check values in a column
EDA_App_list['NAME_CONTRACT_TYPE'].value_counts(normalize=True)


# In[59]:


# To count and check values in a column
EDA_App_list['CODE_GENDER'].value_counts(normalize=True)


# In[60]:


# To find the most occuring object in the column Gender Column
Code_mode = EDA_App_list['CODE_GENDER'].mode()[0]   # To get string value
Code_mode


# In[61]:


# Replace old values ('XNA') with maximumm occuring value ('F') of the respective column
EDA_App_list['CODE_GENDER'] = EDA_App_list['CODE_GENDER'].replace(['XNA'],'F')


# In[62]:


# To recount and check values in a column
EDA_App_list['CODE_GENDER'].value_counts(normalize=True)


# In[63]:


# To analyze 'ORGANIZATION_TYPE' column values
EDA_App_list['ORGANIZATION_TYPE'].value_counts(normalize=True)


# ##### As 'XNA' in the 'ORGANIZATION_TYPE' column is up tp 18% therefore we can replace 'XNA' with the Maximum occuring value in the Organization Types 

# In[64]:


# Find out mode for "ORGANIZATION_TYPE"
Org_mode = EDA_App['ORGANIZATION_TYPE'].mode()[0]
Org_mode


# In[65]:


# Replace old values ('XNA') with maximumm occuring value ('Business Entity Type 3') of the respective column 
EDA_App_list['ORGANIZATION_TYPE'] = EDA_App_list['ORGANIZATION_TYPE'].replace(['XNA'],'Business Entity Type 3')


# In[66]:


# To recount and check values in a column
EDA_App_list['ORGANIZATION_TYPE'].value_counts(normalize=True)


# In[67]:


# To analyze 'NAME_INCOME_TYPE' column
EDA_App_list['NAME_INCOME_TYPE'].value_counts(normalize=True)


# In[68]:


# To analyze 'NAME_EDUCATION_TYPE' column
EDA_App_list['NAME_EDUCATION_TYPE'].value_counts(normalize=True)


# In[69]:


# To replace 'Secondary / secondary special with Secondary.
EDA_App_list.NAME_EDUCATION_TYPE = EDA_App_list.NAME_EDUCATION_TYPE.replace(['Secondary / secondary special'], "Secondary")


# In[70]:


# To analyze 'NAME_FAMILY_STATUS' column
EDA_App_list['NAME_FAMILY_STATUS'].value_counts(normalize=True)


# In[71]:


# To replace 'Civil marriage', 'Seperated' as 'Married' and 'Single', 'Widow' as 'Not Married' just to make 2 categories.
EDA_App_list.NAME_FAMILY_STATUS = EDA_App_list.NAME_FAMILY_STATUS.replace(['Civil marriage','Separated'], "Married")
EDA_App_list.NAME_FAMILY_STATUS = EDA_App_list.NAME_FAMILY_STATUS.replace(['Single / not married','Widow'], "Not Married")


# In[72]:


# To analyze 'NAME_FAMILY_STATUS' column
EDA_App_list['NAME_FAMILY_STATUS'].value_counts(normalize=True)


# In[73]:


# To analyze 'NAME_HOUSING_TYPE' column
EDA_App_list['NAME_HOUSING_TYPE'].value_counts(normalize=True)


# In[74]:


# To make absolute value for 'DAYS_BIRTH' AND 'DAYS_EMPLOYED' column
EDA_App_list['DAYS_BIRTH'] = EDA_App_list['DAYS_BIRTH'].abs()
EDA_App_list['DAYS_EMPLOYED'] = EDA_App_list['DAYS_EMPLOYED'].abs()


# In[75]:


# To convert 'CNT_FAM_MEMBERS' column data type from float to int as number of members can't be float
EDA_App_list.CNT_FAM_MEMBERS = EDA_App_list.CNT_FAM_MEMBERS.astype("int64")


# In[76]:


# To recheck the detail of columns
EDA_App_list.head()


# In[ ]:





# In[91]:


# Dividing the dataset into two datasets to make 2 different categories of column 'TARGET' TARGET=1(client with payment difficulties) and TARGET=0(all other)
EDA_Targt0 = EDA_App_list.loc[EDA_App_list["TARGET"]==0]
EDA_Targt1 = EDA_App_list.loc[EDA_App_list["TARGET"]==1]


# In[92]:


# To find out the length
len(EDA_Targt0)


# In[93]:


#To find out the length
len(EDA_Targt1)


# In[94]:


# To check % imbalance between these two categories of Client with payment difficulties and all other
round(len(EDA_Targt0)/len(EDA_Targt1),2)


# ### The above result shows that there is 11.39% imbalnce between the two categories of loan (Defaulters/Delayed Payment and Others)

# In[943]:


EDA_Targt = EDA_Targt0.append(EDA_Targt1, ignore_index = True) # To append both the dataframes and make a new dataframe


# In[944]:


EDA_Targt.head() # To show various columns


# In[947]:


EDA_Targt.shape  # To check the shape of the dataframe


# ### Do Univeriate analysis 

# In[126]:


# To analyze 'TARGET' column whether loan defaulters are more or less ('1' = loan defaulter/did late payments, '0'= for all other case )
fig = plt.figure(figsize=(6,5))
EDA_App_list['TARGET'].value_counts().plot.barh()
plt.title("Defaulters/delayed payment, Other Payments Vs Count\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nCounts\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
#plt.xticks(rotation=90)
plt.ylabel("Target Values (Defaulter,Others)\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences
# 
# Inferences : As per the above bargraph loan defaulters are very less than all other cases.

# In[123]:


# To show value counts for different Income Types
fig = plt.figure(figsize=(6,5))
EDA_App_list['NAME_INCOME_TYPE'].value_counts().plot.barh()
plt.title("Client's Income Type Vs Count\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nCounts\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.ylabel("Client's Income Types\n(Defaulter/Others)\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences 
# 
# Inferences: Above bargraph shows that maximum people's source of income is job/working and others are Commercial Associate, Pensioner, and State Servant whereas very few are Businessman, Students, Unemployed and are on Meternity leave.

# In[124]:


# To show the Types of Education
fig = plt.figure(figsize=(6,5))
EDA_App_list['NAME_EDUCATION_TYPE'].value_counts().plot.barh()
plt.title("Education Type Vs Count\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nCounts\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.ylabel("Client's Education Types\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences:
# 
# Inferences: According to the above Bargraph maximum people have done upto Seconday education, whereas very less have done Academic degree.

# ### Bivariate/Multivariate Analysis

# In[80]:


# To analyze various income types in relation with the Gender 
fig = plt.figure(figsize=(7,5))
sns.barplot(data=EDA_App_list, x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', hue='CODE_GENDER', ci=None)
plt.title("Income Type Vs Total Income in terms of Gender\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nIncome Type\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Total Amount of Income\n(INR)\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences:
# 
# Inferences: From above bargraph it is clear that among all the categories of Income Type (axcept 'Unemployed and 'Student') 'Male' has higher Total income than 'Female'.
# 

# In[81]:


# To analyze Total Income for Various Income Types in terms of Target
fig = plt.figure(figsize=(8,5)) 
sns.barplot(data=EDA_App_list, x='NAME_INCOME_TYPE', y='AMT_INCOME_TOTAL', hue='TARGET', ci=None)
plt.title("Income Type Vs Total Income in terms of Target\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nIncome Type\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Total Amount of Income\n(INR)\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# #### Inferences
# 
# Inferences 1: With the help of above graph it can be seen that there are no people who are businessman and having average income more than INR 600000 are not the loan defaulter / did delayed payment, whereas the people who are Working/Pensioner has equal amount of loan defaulter and other cases.
# 
# Inferences 2: It means Businessman are less Defaulters / did delayed payment.

# In[82]:


# Bar Chart of 'Married, Not Married' people with respect to Total income and are loan defaulter or not
fig = plt.figure(figsize=(8,5)) # ORGANIZATION_TYPE, estimator=np.max
sns.barplot(data=EDA_App_list, x='NAME_FAMILY_STATUS', y='AMT_INCOME_TOTAL', hue='TARGET', ci=None)
plt.title("Family Status Vs Total Income in terms of Target\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nFamily Status\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.ylabel("Total Amount of Income\n(INR)\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences from the above Bar plot:
# 
# Inference 1: The above Bar Graph is showing that 'Married' people who has good income having less marginal difference between loan defaulter and all other cases.
# 
# Inference 2: People who are 'Not Married' and having higher income has little less defaulter than Married people.  
# 
# Inferences 3: It means 'Not Married' people can be preferred over 'Married' people during loan approvals.

# In[83]:


# To Count no. of childerns 
EDA_App_list['CNT_CHILDREN'].value_counts()


# In[84]:


# To analyze number of childern for Various Income Types in terms of Target
fig = plt.figure(figsize=(7,5)) 
sns.barplot(data=EDA_App_list, x='CNT_CHILDREN', y='AMT_INCOME_TOTAL', hue='TARGET', ci=None)
plt.title("Number of Childern Vs Total Income in terms of Target\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nNumber of Childern\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Total Amount of Income\n(INR)\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.show()


# ### Inferences from the above Bar plot:
# Inference 1: In most of the cases as number of children increases loan defaulters/delayed payment increases.
# 
# Inference 2: As Total Income increses number of children increases.
# 
# Inferences 3: Therefore, if number of children are more loan defaulters/ delayed payment are more, hence for up to 4 children loan defaulters are less. 
# 
# Inferences 4: Bank should avoid giving loan to the people who having more children.  

# In[85]:


# Box plot - Amount (Credit/Annuity) Vs Target in terms of Contract Type.
get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(20, 20))
fig1 = fig.add_subplot(3, 3, 1)    # For adding subplot side by side
sns.boxplot(data=EDA_App_list, x='TARGET', y='AMT_CREDIT', hue="NAME_CONTRACT_TYPE", ax=fig1)
plt.title("Amount Credit Vs Target\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nTARGET\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.ylabel("AMT_CREDIT\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})

fig2 = fig.add_subplot(3, 3, 2)    # For adding subplot side by side
sns.boxplot(data=EDA_App_list, x='TARGET', y='AMT_ANNUITY', hue="NAME_CONTRACT_TYPE", ax=fig2)
plt.title("Amount Annuity Vs Target (Defaulter/Others)\n", fontdict={'fontsize':18,'fontweight':5,'color':'Green'})
plt.xlabel("\nTARGET\n",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
plt.ylabel("\nAMT_ANNUITY",fontdict={'fontsize':14,'fontweight':5,'color':'Brown'})
fig.tight_layout()
plt.show()


# ### There are following Inferences from the above Box plots
# Inference 1: For the client whether they are loan defaulter/ did delayed payment or for all other cases having higher Amount Annuity for Cash loans than Revolving loans.
# 
# Inference 2: Amount Annuity is equally distributed for Cash loans whether it is loan defaulter/ did delayed payment or for all other cases.
# 
# Inference 3: For both the Targets Revolving loans maximum is less than the minimum of Cash loans.
# 
# Inference 4: For both the cases Credit amount and Amount Annuity is equally divided for Cash loans whereas, Revolving loan's upper half is higher in both the Targets. 

# In[86]:


## Amount types Heatmap plot.....
Income_Bucket = EDA_App_list.groupby('ORGANIZATION_TYPE')['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY'].mean()
fig = plt.figure(figsize=(40,40))
fig1 = fig.add_subplot(3, 3, 1)    # For adding subplot side by side
sns.heatmap(Income_Bucket, cmap="RdYlGn", annot=True, ax=fig1)
plt.title("Organization Types Vs Different Amounts\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nAmount Totals\n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Organization Types",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})


## Days Types Heatmap plot.....
Income_Bucket = EDA_App_list.groupby('ORGANIZATION_TYPE')['DAYS_BIRTH','DAYS_EMPLOYED'].mean()
fig2 = fig.add_subplot(3, 3, 2)    # For adding subplot side by side
sns.heatmap(Income_Bucket, cmap="RdYlGn", annot=True, ax=fig2)
plt.title("Organization Types Vs Different Days\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nNumber of Days\n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("\n\nOrganization Types",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.show()


# ### There are following Inferences from the above Heatmap
# 
# Inference 1: Although Maximum Organization Type for the 'legal services' is in medicore range still maximum Credit amount of the loan is for the same Organization Type ('Legal Services') whereas Amount Annuity is less than 100000 for all the Organization Types. 
# 
# Inference 2: Maximum number of days employed at the time of loan are for the Organiztion type 'Business Entity Type 3'.
# 

# In[117]:


# To find out Correlation among various variables of dataset for Target 0 (All other Cases)
EDA_Targt0_corr = EDA_Targt0.iloc[:,2:]
EDA_Targt0_corr_2 = EDA_Targt0_corr.corr(method='pearson', min_periods=1)


# In[118]:


# To find out Correlation among various variables of dataset for Target 1 (loan defaulter/ Delayed Payment)
EDA_Targt1_corr = EDA_Targt1.iloc[:,2:]
EDA_Targt1_corr_2 = EDA_Targt1_corr.corr(method='pearson', min_periods=1)


# In[119]:


# Heatmap to show coorelation among all the numeric variables for both the cases.
fig = plt.figure(figsize=(40,30))
fig1 = fig.add_subplot(3, 3, 1)    # For adding subplot side by side
sns.heatmap(EDA_Targt1_corr_2, cmap="RdYlGn", annot=True, ax=fig1)
plt.title("Correlation among Various Numerical Data for loan defaulters\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nVarious Numerical Variables\n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("Various Numerical Variables\n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
#plt.show()

fig2 = fig.add_subplot(3, 3, 2)    # For adding subplot side by side
sns.heatmap(EDA_Targt0_corr_2, cmap="RdYlGn", annot=True, ax=fig2)
plt.title("Correlation among Various Numerical Data for All other cases\n", fontdict={'fontsize':22,'fontweight':5,'color':'Green'})
plt.xlabel("\nVarious Numerical Variables\n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.xticks(rotation=90)
plt.ylabel("\nVarious Numerical Variables\n",fontdict={'fontsize':18,'fontweight':8,'color':'Brown'})
plt.show()


# ### Inferences:
# 
# Target 1: For loan defaulters/ delayed payments
# 
# Target 0: For all other cases.
# 
# Inferences 1: No. of Children and total family members count are porportionate with each other for both the cases, therefore more no. of Children means more no. of family members and less total income.
# 
# Inferences 2: Total income amount and Credit income is inversaly proportional to no. of children, therefore less total income, credit amount for more number of children but this ratio is little better for 'Target 0' instead of 'Target 1'.
# 
# Inferences 3: For both the cases 'Amount Annuity' is inversaly proportional to Days Employed, therefore Amount Annuity is less for more no. of days employed. 
# 
# Inferences 4: For maximum variable like Total Income, Credit Amount, Amount Annuity value is higher for 'Target0' with respect to 'Target1'. 

# ### End of the Analysis for the 'Application-Data' set

# 
# 
# 
