# Raw-material Price Forecasting.
## Business Problem: 
The business problem at hand is the unpredictable fluctuation in the prices of raw spice materials, negatively impacting the cost structure and inventory management.
## Business Solution: 
First, I collected research papers related to our problem and after that I tried to obtain any relevant dataset from the collected research articles. Using a secondary dataset (provided by our client) I performed Exploratory Data Analysis (EDA) using MySQL and Python for deriving preliminary insights from the data. After EDA, I jumped into the Data Preprocessing section and cleaned the raw data accordingly using MySQL and Python. Then, I pushed the cleaned data from MySQL to Power BI for visualizing the data. I also used MS Excel as well as an open-source online platform called Looker Studio for creating interactive dashboards.
### EDA and presprocessing code
<details>
  <summary>Click to expand Python code</summary>
	
```MySQL
CREATE DATABASE IF NOT EXISTS Forecasting_db;
USE Forecasting_db;

#TRUNCATE TABLE Spices;
DROP TABLE IF EXISTS spices;

select Price from Spices oRDER BY Price DESC limit 5;


CREATE TABLE IF NOT EXISTS spices (
month_yr VARCHAR(20) NOT NULL,
spices VARCHAR(30) NOT NULL,
Location VARCHAR(20) NOT NULL,
Grade VARCHAR(10) NOT NULL,
Price FLOAT NOT NULL
);

select * from spices;

# to find number of blank values for Grade and Location column #
select count(*) from spices where Price = '';    
select count(*) from spices WHERE Grade = '';
select count(*) from spices WHERE Location = '';
select count(*) from spices WHERE month_yr = '';
select count(*) from spices WHERE spices = '';

-- Dropping unnecessary columns --
ALTER TABLE Spices
DROP COLUMN Grade;

select distinct(count(*)) from spices_data
where Location = 'India';

select count(distinct(spices)) from spices_data;

# number of outliers present in Price column #
SELECT 
    COUNT(*) AS outlier_count
FROM 
    Spices
WHERE 
    Price < (SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY Price) FROM Spices) - (1.5 * (SELECT percentile_cont(0.75) WITHIN GROUP (ORDER BY Price) FROM Spices) - (SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY Price) FROM Spices))
    OR Price > (SELECT percentile_cont(0.75) WITHIN GROUP (ORDER BY Price) FROM Spices) + (1.5 * (SELECT percentile_cont(0.75) WITHIN GROUP (ORDER BY Price) FROM Spices) - (SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY Price) FROM Spices));


                  #-- 1st Moment business decision --#
-- MEAN --
SELECT avg(Price) AS Average_Price FROM spices;


-- MEDIAN--
WITH cte AS (
    SELECT Price,
           ROW_NUMBER() OVER (ORDER BY Price) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM spices
)
SELECT AVG(Price) AS median_Price
FROM cte
WHERE row_num IN (FLOOR((total_count + 1) / 2), CEIL((total_count + 1) / 2));

-- MODE --
# Modal price #
with cte as( select *, count(Price) over(partition by Price) 
as ranked  from Spices)

select distinct(Price) from cte 
where ranked=(select max(ranked) from cte);

-- Modal year --
with cte as( select *, count(right(month_yr, 2)) over(partition by month_yr) 
as ranked  from Spices)

select distinct(right(month_yr, 2)) from cte 
where ranked=(select max(ranked) from cte);

-- Modal month and year --
SELECT month_yr AS mode_value, COUNT(*) AS frequency
FROM spices GROUP BY month_yr
ORDER BY frequency DESC LIMIT 1;

# Most-frequent Location-
SELECT Location AS mode_value, COUNT(*) AS frequency
FROM spices GROUP BY Location
ORDER BY frequency DESC
LIMIT 1;

                        #- 2nd Moment Business Decision --#
                        
-- Standard Deviation --
select round((stddev(Price)), 4) AS std_Price FROM Spices;

-- Range --
select round(MAX(Price) - MIN(Price), 4) as price_range FROM Spices;

-- Variance --
select ROUND((variance(Price)), 4) as price_variance from spices;


                      #-- 3rd Moment Business Decision --#
-- Skewness --
SELECT
(
SUM(POWER(Price- (SELECT AVG(Price) FROM Spices), 3)) /
(COUNT(*) * POWER((SELECT STDDEV(Price) FROM Spices), 3))
) AS skewness

FROM Spices;      

                    #-- 4th Moment Business Decision --#
-- Kurtosis --
SELECT
(
(SUM(POWER(Price- (SELECT AVG(Price) FROM Spices), 4)) /
(COUNT(*) * POWER((SELECT STDDEV(Price) FROM Spices), 4))) - 3
) AS kurtosis
FROM Spices;

                  #- Duplicate Handling -#
-- counting the duplicates --                  
SELECT Price, COUNT(*) as duplicate_count
FROM Spices
GROUP BY Price
HAVING COUNT(*) > 1;

SELECT spices, COUNT(*) AS duplicate_count
FROM Spices
GROUP BY spices HAVING COUNT(*) < 30;

SELECT Location, count(*) AS duplicate_count
from Spices
GROUP BY Location HAVING count(*) > 1;


set sql_safe_updates = 0;

-- dropping the duplicates --
DELETE FROM Spices
WHERE spices IN (
    SELECT spices
    FROM (
        SELECT spices, COUNT(*) AS cnt
        FROM (
            SELECT * FROM Spices
        ) AS inner_query
        GROUP BY spices
        HAVING cnt < 30
    ) AS subquery
);

                        #-- Outlier Treatment --#
-- Inter-Quartile Range method --
-- Viewing the outlier values in Price column
WITH orderedList AS (
    SELECT Price, ROW_NUMBER() OVER (ORDER BY Price) AS row_n
    FROM Spices
),
iqr AS (
    SELECT
        Price,
        q3_value AS q_three,
        q1_value AS q_one,
        q3_value - q1_value AS outlier_range
    FROM orderedList
    JOIN (SELECT Price AS q3_value FROM orderedList WHERE row_n = FLOOR((SELECT COUNT(*) FROM Spices) * 0.75)) q3 ON 1=1
    JOIN (SELECT Price AS q1_value FROM orderedList WHERE row_n = FLOOR((SELECT COUNT(*) FROM Spices) * 0.25)) q1 ON 1=1
)
SELECT Price AS outlier_value
FROM iqr
WHERE Price >= q_three + outlier_range
   OR Price <= q_one - outlier_range;

-- using Z-score --
select spices,
(Price - avg(Price) OVER()) / STDDEV(Price) over() AS Z_score
FROM Spices;

-- obtaining extreme outliers (less/more than 3 standard deviations)--
select * from (
select spices,
(Price - avg(Price) OVER()) / STDDEV(Price) over() AS Z_score
FROM Spices) AS extreme_outliers
WHERE Z_score > 2.576 OR Z_score < -2.576;

select count(*) from (
select spices,
(Price - avg(Price) OVER()) / STDDEV(Price) over() AS Z_score
FROM Spices) AS extreme_outliers
WHERE Z_score > 2.576 OR Z_score < -2.576;

-- outliers beyond two standard deviation range --
select count(*) from (
select spices,
(Price - avg(Price) OVER()) / STDDEV(Price) over() AS Z_score
FROM Spices) AS 2stdev_outliers
WHERE Z_score > 1.960 OR Z_score < -1.960;

-- outliers byond one standard deviation range --
select count(*) from (
select spices,
(Price - avg(Price) OVER()) / STDDEV(Price) over() AS Z_score
FROM Spices) AS stdev_outliers
WHERE Z_score > 1.645 OR Z_score < -1.645;


select count(*) from Spices;
                         
                         #-- Zero & Near-Zero Variance --#
SELECT
VARIANCE(month_yr) AS v1,
VARIANCE(spices) AS v2,
VARIANCE(Location) AS v3,
VARIANCE(Grade) AS v4,
VARIANCE(Price) AS v5
FROM Spices;        


                            #-- Missing values --#                 
SET SQL_SAFE_UPDATES = 0;

-- Imputing missing values with Mode--
SELECT Grade
FROM Spices
WHERE Grade IS NOT NULL
GROUP BY Grade
ORDER BY count(*) DESC LIMIT 1
INTO @mode_value;

UPDATE Spices
SET Grade = @mode_value
WHERE Grade = '';

SELECT * FROM Spices;


-- Median imputation (Price column) --
SELECT (WITH cte AS (
    SELECT Price,
           ROW_NUMBER() OVER (ORDER BY Price) AS row_num,
           COUNT(*) OVER () AS total_count
    FROM Spices
)
SELECT AVG(Price) AS median_value
FROM cte
WHERE row_num IN (FLOOR((total_count + 1) / 2), CEIL((total_count + 1) / 2)))
FROM Spices
ORDER BY cte.median_value DESC LIMIT 1
INTO @median_price;
     
                            #-- Discretization --#
-- Price column --
SELECT
month_yr,
spices,
Location,
Price,
CASE
WHEN Price < 100 THEN 'Low'
WHEN Price >= 100 AND Price < 150 THEN 'Medium'
WHEN Price >= 150 THEN 'High'
ELSE 'Unknown'
END AS Price_group
FROM Spices;

							#- Normalization/min-max scaling -#
select * from Spices;
CREATE TABLE Spices_scaled AS
SELECT
month_yr,
spices,
Location,
Grade,
Price,
(Price - min_Price) / (max_Price - min_Price) AS scaled_Price
FROM (
SELECT
Month_yr,
spices,
Location,
Grade,
Price,
(SELECT MIN(Price) FROM Spices) AS min_Price,
(SELECT MAX(Price) FROM Spices) AS max_Price
FROM Spices
) AS scaled_data;

select * from Spices_scaled;


									#- Standardization -#
CREATE TABLE Spices_standardized AS 
SELECT month_yr, spices, Location, Grade, Price,
(Price - mean_price / stdd_price) AS std_Price
FROM (
SELECT month_yr, spices, Location, Grade, Price, (select avg(Price) FROM Spices) AS mean_price, (select stddev(Price) FROM Spices)
AS stdd_price FROM Spices
) 
AS standardized_data;

select * from Spices_standardized;
			
# DQL #
DELETE FROM Spices WHERE Location IS NULL;

select distinct(spices) FROM Spices;

DELETE FROM Spices WHERE month_yr IS NULL;


 #-  computing statistical measures for each Price at different locations -#
SELECT 
    spices,
    Location,
    AVG(Price) AS mean,
    MAX(Price) - MIN(Price) AS `range`,
    STDDEV(Price) AS standard_deviation,
    VARIANCE(Price) AS variance,
(
SUM(POWER(Price- (SELECT AVG(Price) FROM Spices), 3)) /
(COUNT(*) * POWER((SELECT STDDEV(Price) FROM Spices), 3))
) AS skewness,
(
(SUM(POWER(Price- (SELECT AVG(Price) FROM Spices), 4)) /
(COUNT(*) * POWER((SELECT STDDEV(Price) FROM Spices), 4))) - 3
) AS kurtosis
FROM 
    Spices
GROUP BY 
    spices, Location;



SELECT * FROM spices_data;

ALTER TABLE spices_data ADD date DATE;
set sql_safe_updates = 0;
UPDATE spices_data SET date = STR_TO_DATE(Mon_Year, '%d-%m-%Y');



SELECT host FROM mysql.user WHERE user = 'root';


select distinct(Location) from spices_data WHERE Price_new = 'medium';

select max(Price_replaced) from spices_data;
```
<details>
  <summary>Click to expand Python code</summary>
	
```python
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
```
 
### Using Power BI.
![Alt text](https://github.com/Subham1702/Raw-material-Spices-Price-Forecasting/raw/main/Screenshot%20(328).png)

### Using Looker Studio.
![Alt text](https://github.com/Subham1702/Raw-material-Spices-Price-Forecasting/raw/main/IMG-20240408-WA0001.jpg)

## Technology stack:
### 1. For EDA and Data Preprocessing:
• Python: NumPy, Pandas, Matplotlib, Seaborn.
• MySQL
### 2. For Data Visualization:
• Power Bi.
• Looker Studio.
• MS Excel.
## Business Benefits:
• Optimize procurement strategies and reduce production costs by 10%.
• Achieve cost savings in raw material procurement and inventory management at least by 20%.
