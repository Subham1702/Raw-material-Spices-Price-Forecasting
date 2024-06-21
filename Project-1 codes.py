# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:24:20 2024

@author: mital
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(r"C:\Users\mital\Documents\Project-1 (Spices price forecasting) files\ind_spices1.csv")
help(df.duplicated)


# % of missing values present in Grade column-
total_rows = len(data)
null_count = data['Grade'].isna().sum()
missing_percentage = (null_count/total_rows) * 100;

print(f"Percentage of missing values in 'Grade': {missing_percentage:.2f}%")


# percentage of missing values present in Location column-
null_count = data['Location'].isna().sum()
missing_percentage = (null_count/total_rows)*100
print(f"Percentage of missing values in Location : {missing_percentage:.2f}%")


# dropping the Grade column #
data.drop(columns=['Grade'], inplace=True)

data.info()
data.describe()



                """ First Moment Business Decision """
# Evaluation of average Price #
data.Price.mean()                

# Evaluation of median Price #
data.Price.median()


# Evaluation of most-frequent Price #
data.Price.mode() 

# Evaluation of most-frequent spice #
data.Spices.mode()

# Evaluation of most-frequent Month-year #
data.Mon_Year.mode()

# Evaluation of most-frequent Location #
data.Location.mode()



            """ Second Moment Business Decision """
# Evaluation of range in Price #            
range = max(data.Price) - min(data.Price)
range            

# Evaluation of variance in Price #
data.Price.var()

# Evaluation of Standard Deviation in Price #
data.Price.std()



            """ Third Moment Business Decision """
# Evaluation of Skewness in Price column #
data.Price.skew()



            """ Fourth Moment Business Decision """
# Evaluation of Kurtosis in Price column #
data.Price.kurt()




                " Pre-processing of data"
data.dtypes

# removal of duplicates from the rows #
data = df.drop_duplicates(keep = False)
duplicate = df.duplicated()


# looking for outliers in Price column #
import seaborn as sns
sns.boxplot(data.Price)

sns.histplot(data['Price_replaced'], kde=True)


# detection of outliers(finding limits for Price based on IQR)
IQR = data['Price'].quantile(0.75) - data['Price'].quantile(0.25)
lower_limit = data['Price'].quantile(0.25) - 1.5*IQR
upper_limit = data['Price'].quantile(0.75) + 1.5*IQR

# flagging the outliers #
outliers_df = np.where(data.Price > upper_limit, True, np.where(data.Price < lower_limit, True, False))


# Replacing the outlier values with the upper and lower limits #
data['Price_replaced'] = pd.DataFrame(np.where(data['Price'] > upper_limit, upper_limit, np.where(data['Price'] < lower_limit, lower_limit, data['Price'])))

sns.boxplot(data.Price_replaced)



# Discretization of Price column #
data['Price_new'] = pd.cut(data['Price_replaced'], bins = [min(data['Price_replaced']), data['Price_replaced'].quantile(0.25), data['Price_replaced'].mean(), max(data['Price_replaced'])],
                           labels = ["low", "medium", "high"])


# replacing nan values #
median_val = data['Price'].median()

data['Price'].fillna(median_val, inplace = True)

average_val = data['Price_replaced'].mean()
data['Price_replaced'].fillna(average_val, inplace = True)




# check for count of NAN values in each column #
data.isna().sum()


# imputation of missing values #
from sklearn.impute import SimpleImputer

mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
data.Price_new = pd.DataFrame(mode_imputer.fit_transform(data[['Price_new']]))
data.Location = pd.DataFrame(mode_imputer.fit_transform(data[['Location']]))




                 #----- Transformation -----#
import scipy.stats as stats
import pylab

# checking whether Price column is normally distributed
stats.probplot(data.Price_replaced, dist = 'norm', plot = pylab)    

#  Function Transformation
import numpy as np           
stats.probplot(np.log(data.Price_replaced), dist = 'norm', plot = pylab)       


# Power Transformation
# (a) Box-cox Transformation-
import matplotlib.pyplot as plt
import seaborn as sns

fitted_data, fitted_lambda = stats.boxcox(data.Price_replaced)

# creating axes to draw plots
fig, ax = plt.subplots(1, 2)

# Plotting the original data (non-normal) and fitted data (normal)
sns.histplot(data["Price_replaced"], kde = True, stat = "density", kde_kws = dict(cut=3))

sns.histplot(fitted_data, kde = True, stat = "density", kde_kws = dict(cut=3))

# adding legends to the subplots
plt.legend(loc = "upper right")

# rescaling the subplots
fig.set_figheight(5)
fig.set_figwidth(10)

print(f"Lambda value used for Transformation: {fitted_lambda}")

# Transformed data
prob = stats.probplot(fitted_data, dist = stats.norm, plot = pylab)


# (b) Yeo-Johnson Transformation - 
from feature_engine import transformation

# Set up the variable transformer
tf = transformation.YeoJohnsonTransformer(variables = 'Price_replaced')

data_tf = tf.fit_transform(data)

# Transformed data
prob = stats.probplot(data_tf.Price_replaced, dist = stats.norm, plot = pylab)                              




                    #---- Encoding categorical data ----#
import pandas as pd

data_new = pd.get_dummies(data)
data_new_1 = pd.get_dummies(data, drop_first = True)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc_df = pd.DataFrame(enc.fit_transform(data.iloc[:, 1:4]).toarray())




                    #----- Normalisation -----#
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

data['Price_norm'] = norm_func(data['Price_replaced'])


# pushing the data into MySQL 
from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://root:password@Localhost/Forecasting_db')
data.to_sql('spices_data', con=engine, if_exists='replace', index=False)

