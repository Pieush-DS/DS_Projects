# -*- coding: utf-8 -*-
"""UOA_P_Vyas_Data_Ethics_Encryption_file.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yLshlZx1gJ6jMN9vDTM5xbHUEzVrOrej

# <font color = blue> Data Ethics - Data Anonymization  </font>

Data anonymization is a process of modifying or transforming data in such a way that it no longer contains personally identifiable information (PII) or other sensitive details that could identify individuals. The goal of data anonymization is to protect individual privacy while still allowing the data to be used for legitimate purposes such as analysis, research, and testing.

There are multiple techniques  of doing Data Anonymization.Few of them are:

 **STEP 1:**

- De-identification

- Data Masking :
    - Partial Data Masking
    - Full Data Masking
    
- Pseudonymization    

**STEP 2:**
- Decrypting our data
"""

#import important libraries

import pandas as pd

# Generating or manipulating random integers
import random

# helps us encrypt and decrypt data.
from cryptography.fernet import Fernet

# To remove warnings
import warnings
warnings.filterwarnings('ignore')

# To set the Row, Column and Width of the Dataframe to show on Jupyter Notebook
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Loading the dtaset
churn  = pd.read_excel("Data+ethics+assignment+-+dataset.xlsx")

# To show first 5 rows of dataframe
churn.head()

"""<div class="alert alert-block alert-info">In Attribute Suppression, entire columns containing sensitive attributes can be removed from the dataset. This ensures that no information related to those attributes is present.</div>

# STEP 1

# 1. De-identification
"""

#create a copy of original dataset, as you dont want to remove coloumns from original dataset
df_copy = churn.copy(deep=True)
df_copy

# Here we are changing column name
column_mapping = {column: column.replace("Member's ","") for column in churn.columns if "Member's " in column}
df_copy = df_copy.rename(columns=column_mapping)

# To show variables of dataframe
df_copy.head()

# Define the columns to remove
columns_to_remove = ['Full Name', 'Address', 'Email', 'Phone Number', 'Credit Card Number']

# Drop the specified columns from the DataFrame
df_copy.drop(columns_to_remove, axis=1, inplace=True)

# To show varaiables
df_copy.head()

"""**So here we have deleted all the sensitive records. Now we can use this new dataframe for further data processing.**

# 2. Data Masking

Data masking, also known as data obfuscation or data anonymization, is a technique used to protect sensitive information by replacing, hiding, or scrambling original data with fake or altered data while maintaining the data's overall format and structure. The primary goal of data masking is to ensure that sensitive data remains confidential and secure, especially when it needs to be shared or used in non-production environments for testing, development, or analysis.
"""

# Returning the first 5 rows
churn.head()

# Here we are changing column name
column_mapping = {column: column.replace("Member's ","") for column in churn.columns if "Member's " in column}
churn = churn.rename(columns = column_mapping)

# Returning the first row to check the updated column names
churn.head(1)

# To make a copy of original datset
df_copy = churn.copy(deep=True)

"""###  Partial data masking of numerical values

**Creating a function for masking only the  "PHONE NUMBER"  column in our dataframe. Other numerical columns remain the same.**
"""

# creating a function "mask_phone_number" for masking phone numbers.

def mask_phone_number(Phone_Number):
    str1 = str(Phone_Number)
    str_output = ''
    count = 0
    val = int(len(str1) * 0.6)

    for x in str1:
        if random.randint(1, 10) % 2 == 0 and count <= (len(str1) - val):
            s = x
            count += 1
        else:
            s = "*"

        str_output += s

    return str_output

# Rename the Phone Number Column
df_copy.rename(columns={'Phone Number': 'Phone_Number'}, inplace=True)

# To Mask the phone numbers

for i in range(0,len(df_copy['Phone_Number'])):
    df_copy['Phone_Number'][i]=mask_phone_number(df_copy['Phone_Number'][i])

# To verify wheather the phone numbers are masked or not.
df_copy.head(1)

"""# 3 .Pseudonymization

The importance of encrypted data lies in the protection and security it provides to sensitive information. Encryption is a crucial tool in safeguarding data from unauthorized access, ensuring privacy, and preventing data breaches. Here are some key reasons why encrypted data is important:

1)Data Confidentiality

2)Data Integrity

3)Privacy Protection

4)Mitigating Data Breaches

5)Secure Communication

6)Compliance and Legal Requirements

7)Protecting Intellectual Property

8)Cloud Security

To encrypt the specified columns using Fernet encryption, you can use the cryptography library in Python.

First, you'll need to generate a secret key for encryption.

**Here we are Encryting Multipe  column  in our dataset.**
"""

# To return the first 3 rows of our dataframe
churn.head(3)

# To view all the columns names
churn.columns

# Generate a secret key
key = Fernet.generate_key()

# Create an instance of the Fernet cipher using the secret key
fernet = Fernet(key)

# Save the secret key for future decryption (keep it secure)
with open('secret.key', 'wb') as key_file:
    key_file.write(key)

# Columns needed to be encrypted
Encrypted_columns = ['Full Name','Address', 'Email', 'Phone Number' , 'Credit Card Number']

# Encrypt the data in each  selected columns
for column in Encrypted_columns:
    churn[column] = churn[column].apply(lambda x: fernet.encrypt(str(x).encode()))

# Save the encrypted DataFrame to a new CSV file
churn.to_csv('Encrypted_data.csv', index=False)

# Reading the Encrypted data
pd.read_csv("Encrypted_data.csv")

"""- As we can see all the sensitive data has been encryted. Now if we share this encryted data with anyone then they will not be able to misuse our senstive information.
- All the sensitive information has been encrypted.

#  STEP 2

# 2. Decrypting our data

To decrypt data using the Fernet secret key stored in a text file, you can follow these steps:

Read the Fernet secret key from the text file.

Initialize the Fernet object with the secret key.

Decrypt the data in each cell of the specified columns using the Fernet object.
"""

# Read the secret key from the file (make sure it's secure and not accessible to unauthorized users)
with open('secret.key', 'rb') as key_file:
    key = key_file.read()

# Create an instance of the Fernet cipher using the secret key
fernet = Fernet(key)

#creating a list of columns which are encryted
Encrypted_columns = ['Full Name','Address', 'Email', 'Phone Number' , 'Credit Card Number']

# Create a new DataFrame to store the decrypted data
decrypted_df = churn.copy()

# To generate the key
fernet_key = Fernet.generate_key()

# List of columns to decrypt
Encrypted_columns = ['Full Name', 'Address', 'Email', 'Phone Number', 'Credit Card Number']

# Decrypt the data in each column
for column in Encrypted_columns:
    decrypted_df[column] = decrypted_df[column].apply(lambda x: fernet.decrypt(x).decode())

"""- Now in this step **Decrypted the previously Encrypted data**. Which can be read by anyone."""

# Reading top 5 rows of decrypted dataframe.
decrypted_df.head()

"""### Important points to remember

If you are getting an "invalid token" error, it usually means that the Fernet key you are using is not valid.
Here are some possible reasons for this error:

@Incorrect Fernet Key: Make sure that the Fernet key you are using is correct and in the right format.
The Fernet key must be 32 url-safe base64-encoded bytes. If the key is not in the correct format, you will get an
"invalid token" error.

@Key Mismatch: Ensure that you are using the correct Fernet key that was used to encrypt the data.
If you use a different key for decryption than the one used for encryption, you will get an "invalid token" error.

@Data Corruption: If the encrypted data has been corrupted or modified, you might encounter an "invalid token" error
when trying to decrypt it.

To resolve this issue, double-check the Fernet key and ensure that it is correct. If you are still facing the problem,
consider re-encrypting the data with a new Fernet key and try the decryption process again. Additionally, ensure that the data has not been corrupted or tampered with during storage or transfer.

# CONCLUSION:

It's important to understand  that while data anonymization can significantly reduce the risk of individual identification, re-identification attacks are still possible in some cases, especially when combined with other available data sources. Therefore, it's crucial to carefully plan and execute anonymization strategies based on the specific data and privacy requirements.

## <font color = blue> Thanks for your pateince  </font>
"""