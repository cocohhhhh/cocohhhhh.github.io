---
title: Python Cheat sheet
date: 2024-12-26
categories: [Notes, Class Notes]
tags: [python]     # TAG names should always be lowercase
---


This cheat sheet is adapted from the ADA course materials, with special thanks to [Mehdi](https://github.com/medimed66), [Jiaming](https://github.com/jiaming-jiang), [Yanzi](https://github.com/llooyee7) and [Davide](https://github.com/davromano) for their contributions.

## Panda basics

### Initialize a dataframe

```python
data = pd.DataFrame({'value':[632, 1638, 569, 115, 433, 1130, 754, 555],
                   'patient':[1, 1, 1, 1, 2, 2, 2, 2],
                   'phylum':['Firmicutes', 'Proteobacteria', 'Actinobacteria', 
  'Bacteroidetes', 'Firmicutes', 'Proteobacteria', 'Actinobacteria', 'Bacteroidetes']})
```

### Rename the column names

```python
# Rename the 'old_name' column to 'new_name'
df.rename(columns={'old_name': 'new_name'}, inplace=True)
```

### Set indexes

- Set one column as index

    ```python
    # Create a sample DataFrame
    data = {'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'San Francisco', 'Los Angeles']}
    
    df = pd.DataFrame(data)
    
    # Set the 'Name' column as the index
    df.set_index('Name', inplace=True)
    ```

- Reset indexes

    ```python
    # Reset the index
    df.reset_index(inplace=True)
    
    # After resetting the index, DataFrame will have the default integer index
    ```

### Reshape (Concat and join)

- Concat (along the rows or cols)

    ```python
    # Sample DataFrames
    df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                        'B': ['B0', 'B1', 'B2']})

    df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5'],
                        'B': ['B3', 'B4', 'B5']})

    # Concatenate along rows (axis=0)
    result = pd.concat([df1, df2], axis=0)

    # Concatenate along cols (axis=1)
    result = pd.concat([df1, df2], axis=1)
    ```

- Join (the cols)

    ```python
    # Sample DataFrames
    df1 = pd.DataFrame({'key1': ['A', 'B', 'C', 'D'],
                        'key2': ['X', 'Y', 'Z', 'X'],
                        'value1': [1, 2, 3, 4]})

    df2 = pd.DataFrame({'key1': ['B', 'D', 'E', 'F'],
                        'key2': ['Y', 'X', 'Z', 'W'],
                        'value2': [5, 6, 7, 8]})

    # Join based on multiple columns ('key1' and 'key2')
    result = pd.merge(df1, df2, on=['key1', 'key2'], how='inner')
    ```

- **Inner Join:** Returns rows with common values in both dataframes.
- **Outer Join (Full Outer Join):** Returns all rows and fills in missing values with NaN.
- **Left Join (Left Outer Join):** Returns all rows from the left dataframe and matching rows from the right dataframe.
- **Right Join (Right Outer Join):** Returns all rows from the right dataframe and matching rows from the left dataframe.

### Sort

- Sort values

    ```python
    # Sort by 'Age' in ascending order, then by 'Salary' in descending order
    df_sorted = df.sort_values(by=['Age', 'Salary'], ascending=[True, False])

    # numpy sorted
    numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

    # Sorting in ascending order (default)
    sorted_numbers_asc = sorted(numbers)
    print(sorted_numbers_asc)  # Output: [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]

    # Sorting in descending order
    sorted_numbers_desc = sorted(numbers, reverse=True)
    print(sorted_numbers_desc)  # Output: [9, 6, 5, 5, 5, 4, 3, 3, 2, 1, 1]
    ```

- Sort indexes

    ```python
    # Sort the index in descending order
    df_sorted_index_desc = df.sort_index(ascending=False)
    ```

### Find, Replace, Drop

- Loc

    ```python
    # Select columns in positions 1, 2 and 5 (first column is 0).
    df.iloc[10:20] Select rows 10-20. df.iloc[:, [1, 2, 5]]
    # Select all columns between x2 and x4 (inclusive).
    df.loc[:, 'x2':'x4']
    # Select rows meeting logical condition, and only the specific columns .
    df.loc[df['a'] > 10, ['a’, 'c']]

    # loc a series item with index
    tmp_a.loc[2008]
    ```

- Filtering rows

    ```python
    # filter numbers
    filtered_df = df[df['Age'] > 30]
    # filter string
    male_df = df[df['Gender'] == 'Male']
    filtered_df = df[df['column_1'].str.endswith("ple")]
    # multiple criteria
    filtered_df = df[(df['Column1'] >= 20) & (df['Column2'] < 40)]
    ```

- Replacing data

    ```python
    # replace string
    df['Gender'].replace('Male', 'M', inplace=True)
    # Replace the end of strings in 'column_1' from "ple" to "abc"
    df['column_1'] = df['column_1'].str.replace(r'ple$', 'abc')

    # Replace numbers larger than 100 in the 'Value' column with 100
    df.loc[df['Value'] > 100, 'Value'] = 100

    # replace multiple data
    cdystonia.treat.replace({0:'Placebo', 1:'5000U', 2:'10000U'})

    # Fill na
    # Fill missing values with a specific value (e.g., 0)
    df_filled = df.fillna(0)
    # Fill missing values with the mean of the column
    df_filled_mean = df.fillna(df.mean())

    # Removing duplicates
    df.drop_duplicates(inplace=True)

    # Changing data types
    df['column_name'] = df['column_name'].astype('desired_data_type')
    ```

- Drop

    ```python
    # Drop rows
    df.drop(df[df['Salary'] < 50000].index, inplace=True)
    # Drop cols
    df.drop(columns=['Column1', 'Column2'], inplace=True)
    # Drop rows with missing values
    df_dropped = df.dropna()
    # Remove duplicate rows
    df_no_duplicates = df.drop_duplicates()
    ```

### Numerical features

- Value counts

    ```python
    # Count number of rows with each unique value of variable
    df['w'].value_counts()

    # number of distinct values in a column.
    df['w'].nunique()
    ```

- Mean, min, max, std, median

    ```python
    # per col, same for mean, max, std, median
    min_value = df['Column1'].min()
    # or use this
    df['Column1'].agg(['mean', 'median', 'min', 'max'])

    # per row
    std_values_per_row = df.std(axis=1)
    ```

- Percentiles

    ```python
    df['Column1'].quantile(0.25)  # 25th percentile

    # cut data into percentiles
    quintiles = pd.qcut([v for _, v in sorted(scores_dict.items())], q=5, labels=False)
    ```

- Correlation and covariance

    ```python
    df[['Column1', 'Column2']].corr()
    df[['Column1', 'Column2']].cov()
    ```

### Groupby

- Group and get the mean, count, median, etc

	After group-by, the group-by column will be the index of the output.

    ```python
    df.groupby('Category')['Value'].mean()
  
    # Calculate counts in column A for each unique value in column B
    counts_by_category = df.groupby('column_B')['column_A'].count()
  
    # Calculate the mean of corresponding values in column A for each unique value in column B
    mean_by_category = df.groupby('column_B')['column_A'].mean()
  
    # Calculate the median of column D for each unique combination of columns A and B
    median_by_combination = df.groupby(['column_A', 'column_B'])['column_D'].median()
  
    # Calculate the maximum value in column E for each year in column F
    max_by_year = df.groupby('column_F')['column_E'].max()
    ```

- Calculate anything after group-by

    ```python
    # any customized function
    # get the mean of the positive numbers
    df.groupby(['YEA','TGT'])['VOT'].apply(lambda x: np.mean(x>0))
    
    # get the range
    def price_range(x):
    		range = x.max() - x.min()
    		# range = x.quantile(0.975) - x.quantile(0.025)
        return range
    
    result = df.groupby('Category')['Price'].apply(price_range)
    
    # Calculate the fraction of a certain category X in column C
    # for each unique value in column B
    def calculate_fraction(data, category_x):
        total_count = data['column_C'].count()
        x_count = data[data['column_C'] == category_x]['column_C'].count()
        return x_count / total_count
    
    fraction_by_category = df.groupby('column_B').apply(lambda x: calculate_fraction(x, 'category_X'))
    ```

- Reset the index of the group-by result

    ```python
    # reset the index ('YEA', 'TGT') of the groupby result
    # and set the new index ('YEA')
    tmp = df.groupby(["YEA", "TGT"]).VOT
    tmp_a = tmp.count().reset_index().groupby("YEA").VOT.count()
    ```

- Aggregate functions

    ```python
    import pandas as pd
    
    data = {'Category': ['A', 'B', 'A', 'B', 'A'],
            'Value': [10, 20, 15, 25, 30]}
    
    df = pd.DataFrame(data)
    
    # Group by 'Category' and aggregate 'Value'
    grouped = df.groupby('Category')
    result = grouped['Value'].agg(['sum', 'mean', 'max']).reset_index()
    
    print(result)
    
    >>>
    Category  sum  mean  max
    0        A   55  18.333333   30
    1        B   45  22.500000   25
    ```

### Math operations

```python
# for one col
def custom_function(x):
  return x ** 2
df['Squared'] = df['Column1'].apply(custom_function)

# for multiple cols
df['Result'] = df['Column1'] + df['Column2']
baseball['obp']=baseball.apply(lambda p: (p.h+p.bb+p.hbp)/(p.ab+p.bb+p.hbp+p.sf) if (p.ab+p.bb+p.hbp+p.sf) != 0.0 else 0.0, axis=1)
```

## Visualization

### Basic charts

- Use Case of different plots

    1. Bar Charts:
       - **Use Case:** Comparing values across categories.
       - **Example:** Comparing sales performance for different products.
    2. Histograms:
       - **Use Case:** Showing the distribution of a continuous variable.
       - **Example:** Displaying the distribution of ages in a population.
    3. Line Charts:
       - **Use Case:** Visualizing trends over a continuous variable, often time.
       - **Example:** Showing the stock prices over a period of time.
    4. Scatter Plots:
       - **Use Case:** Examining the relationship between two continuous variables.
       - **Example:** Plotting the relationship between height and weight.
    5. Box Plots (Box-and-Whisker Plots):
       - **Use Case:** Displaying the distribution of a dataset and highlighting outliers.
       - **Example:** Comparing the distribution of exam scores across different classes.
    6. Pie Charts:
       - **Use Case:** Showing the proportion of each category in a whole.
       - **Example:** Displaying the percentage distribution of expenses in a budget.
    7. Heatmaps:
       - **Use Case:** Visualizing the magnitude of a phenomenon across two categorical variables.
       - **Example:** Displaying the correlation matrix between variables.
    8. Violin Plots:
       - **Use Case:** Combining the benefits of a box plot and a kernel density plot.
       - **Example:** Visualizing the distribution of a variable across different groups.
    9. Radar Charts:
       - **Use Case:** Comparing multiple quantitative variables across different categories.
       - **Example:** Comparing the skill levels of individuals in different sports.
    10. Treemaps:
        - **Use Case:** Displaying hierarchical data as nested rectangles.
        - **Example:** Visualizing the composition of expenses in a budget hierarchy.
    11. Choropleth Maps:
        - **Use Case:** Showing spatial variations in a variable across regions.
        - **Example:** Displaying population density across different countries.

- Bar chart

    ```python
    import matplotlib.pyplot as plt
    
    plt.bar(df['categories'], df['values'], color='blue')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Bar Chart')
    plt.show()
    ```

- Histogram

    ```python
    plt.hist(df['variable'], bins=20, color='green', alpha=0.7)
    plt.xlabel('Variable')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()plt.hist(df['variable'], bins=20, color='green', alpha=0.7)
    plt.xlabel('Variable')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()
    ```

- Line chart

    ```python
    plt.plot(df['time'], df['values'], marker='o', linestyle='-', color='red')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.title('Line Chart')
    plt.show()
    ```

- Scatter plot

    ```python
    plt.scatter(df['x'], df['y'], color='purple', alpha=0.5)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot')
    plt.show()
    ```

- Box plot

    ```python
    import seaborn as sns
    
    sns.boxplot(x=df['category'], y=df['values'], palette='Set3')
    plt.xlabel('Category')
    plt.ylabel('Values')
    plt.title('Box Plot')
    plt.show()
    ```

- Pie chart

    ```python
    plt.pie(df['values'], labels=df['categories'], autopct='%1.1f%%', colors=['gold', 'lightcoral'])
    plt.title('Pie Chart')
    plt.show()
    ```

- Heatmap

    ```python
    import seaborn as sns
    
    heatmap_data = df.pivot(index='row_variable', columns='column_variable', values='values')
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
    plt.title('Heatmap')
    plt.show()
    
    # normalize the heatmap
    # Your original array
    data = np.array([[  63,    5,    4,    0,    1],
                     [  61,   68,   16,   11,    4],
                     [ 123,   87,  107,   35,    9],
                     [ 136,  145,  192,  192,   56],
                     [ 212,  306,  493,  681, 1381]])
    
    # Normalize rows to sum up to 1
    normalized_data = data / data.sum(axis=1, keepdims=True)
    
    # Create a heatmap
    sns.heatmap(normalized_data, annot=True, fmt=".2f")  # Format to 2 decimal places
    plt.ylabel("Row Index")
    plt.xlabel("Column Index")
    plt.title("Normalized Heatmap (Rows sum to 1)")
    plt.show()
    ```

- Violin plot

    ```python
    sns.violinplot(x=df['category'], y=df['values'], palette='viridis')
    plt.xlabel('Category')
    plt.ylabel('Values')
    plt.title('Violin Plot')
    plt.show()
    ```

- Radar chart

    ```python
    from math import pi
    
    categories = list(df.columns[1:])
    values = df.iloc[0].tolist()[1:]
    values += values[:1]  # To close the circular graph
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    plt.polar(angles, values, 'o-', color='orange', linewidth=2)
    plt.fill(angles, values, color='orange', alpha=0.25)
    plt.title('Radar Chart')
    plt.show()
    ```

- Treemap

    ```python
    import squarify
    
    squarify.plot(sizes=df['values'], label=df['categories'], color=['skyblue', 'salmon', 'lightgreen'])
    plt.title('Treemap')
    plt.axis('off')
    plt.show()
    ```

- Choropleth map

    ```python
    import geopandas as gpd
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Assuming 'gdf' is a GeoDataFrame with geometry and value columns
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    
    gdf.plot(column='values', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True, cax=cax)
    ax.set_title('Choropleth Map')
    plt.show()
    ```

- Q-Q plot

    ```python
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import numpy as np
    
    # Assuming 'sample_data' is your sample data
    sample_data = np.random.normal(loc=0, scale=1, size=1000)  # Replace this with your actual sample data
    
    # Create a Q-Q plot
    stats.probplot(sample_data, dist='norm', plot=plt)
    plt.title('Q-Q Plot')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.show()
    ```

- Pair Plot

    ```python
    import seaborn as sns
    sns.pairplot(data)
    ```

### Dimension reduction (TSNE/PCA)

```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# dimension reduction using tsne
X_reduced_tsne = TSNE(n_components=2, init='random', learning_rate='auto', random_state=0).fit_transform(X)

# dimension reduction using pca
X_reduced_pca = PCA(n_components=2).fit(X10d).transform(X)
```

### CDF and CCDF

  The *Cumulative Distribution Function* (CDF) plot is a lin-lin plot with data overlay and confidence limits. It shows the cumulative density of any data set over time (i.e., Probability vs. size).

```python
import seaborn as sns
import warnings

# Creates complementary CDF with log scale
# FOR CDF: complementary=False
sns.ecdfplot(df[df["throws"] == "L"].salary, label="Left-handed", complementary=True)
sns.ecdfplot(df[df["throws"] == "R"].salary, label="Right-handed", complementary=True)
plt.xscale("log")
plt.legend()
plt.title("CCDF Salary")
```

A bit more about the calculation of CDF and CCDF

  ```python
  def get_ccdf(var_list):
      """
      Get the complementary culmulative distribution function of a list of values
      """
      var_count = Counter(var_list)
      var_count = sorted(var_count.items(), key=lambda x: x[0])
      ccdf = 1 - np.cumsum([x[1] for x in var_count]) / sum([x[1] for x in var_count])
      return [x[0] for x in var_count], ccdf
  
  def get_cdf(var_list):
      """
      Get the culmulative distribution function of a list of values
      """
      var_count = Counter(var_list)
      var_count = sorted(var_count.items(), key=lambda x: x[0])
      cdf = np.cumsum([x[1] for x in var_count]) / sum([x[1] for x in var_count])
      return [x[0] for x in var_count], cdf
  ```

### PDF (probability density function)

  The term *Probability* is used in this instance to describe the size of the total population that will fail (failure data or any other data) by size.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting the PDF
sns.histplot(df[df["throws"] == "L"].salary, kde=True, stat="density", label="Left-handed")

# Adding labels and title for clarity
plt.xlabel
```

### Layout

```python
# Creating Multiple Subplots

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

axes[0, 0].plot(df['x'], df['y'], marker='o', linestyle='-', color='red')
axes[0, 0].set_xlabel('X-axis')
axes[0, 0].set_ylabel('Y-axis')
axes[0, 0].set_title('Subplot 1')

# Add more subplots as needed

plt.tight_layout()
plt.show()
```

## Simple statistics

### Sample data

```python
#make 10 samples with replacement
sample1_counties = df.sample(n = 10, replace = True)

#make 10 samples without replacement
sample1_counties = df.sample(n = 10, replace = False)

#sometimes we want to sample in an ublanaced way, so that we upsample datapoints of certain characteristic,
#and downsample the others. this can be acieved with weights parameter
#here we sample by upsampling counties with large population
sample2_counties = df.sample(n = 10, replace = False, weights = df['TotalPop'])

# sample a fraction
# Calculate the desired sample size as 0.1% of the total rows
percentage = 0.1  # Change this to the desired percentage
sample_size = int(len(df) * (percentage / 100))

# Perform the sample
sampled_df = df.sample(n=sample_size, replace=False)
```

### Distribution test

```python
from statsmodels.stats import diagnostic
# get statistical desription for every column of df
df.describe()

# test to verify if the data come from a normal distrbution
diagnostic.kstest_normal(df['IncomePerCap'].values, dist = 'norm')
#output: (statistic, p-value)
#p_value < 0.05 -> not a normal distribution!

# test to verify if the data come from an exponential distribution
diagnostic.kstest_normal(df['IncomePerCap'].values, dist = 'exp')
#output: (statistic, p-value)
#p_value < 0.05 -> not exponential distribution!
```

### 95% CI of the mean

- Calculate and point plot

    ```python
    salaries = df.groupby(["throws"]).salary.agg(["mean", "sem"])
    salaries["low_ci"] = salaries["mean"] - 1.96 * salaries["sem"]
    salaries["high_ci"] = salaries["mean"] + 1.96 * salaries["sem"]

    # shows confidence intervals
    display(salaries)

    # simple plot
    sns.pointplot(x="throws", y="salary", data=df_pitching)
    plt.title("Average salary for left-\\nand right-handed throwers")
    ```

- Line plot with 95% confidence intervals

    ```python
    fig, axs = plt.subplots(1, 2, figsize=(14,4))
    
    for idx, col_agg in enumerate(["salary", "BAOpp"]):
        df_col_agg = df.groupby(["yearID", "throws"])[col_agg].agg(["mean", "sem"]).reset_index()
        df_col_agg_L = df_col_agg[df_col_agg.throws == "L"]
        df_col_agg_R = df_col_agg[df_col_agg.throws == "R"]
    
        axs[idx].plot(df_col_agg_L["yearID"], df_col_agg_L["mean"], color="tab:red", label="lefties")
        axs[idx].fill_between(df_col_agg_L["yearID"], df_col_agg_L["mean"] - 1.96 * df_col_agg_L["sem"], 
                         df_col_agg_L["mean"] + 1.96 * df_col_agg_L["sem"], alpha=0.25
                         , color="tab:red")
        axs[idx].plot(df_col_agg_R["yearID"], df_col_agg_R["mean"], color="tab:blue", label="righties")
        axs[idx].fill_between(df_col_agg_R["yearID"], df_col_agg_R["mean"] - 1.96 * df_col_agg_R["sem"] , 
                         df_col_agg_R["mean"] + 1.96 * df_col_agg_R["sem"], alpha=0.25
                         , color="tab:blue")
        
        print("avg", col_agg, "in 1999 for lefties:", df_col_agg_L[df_col_agg_L["yearID"] == 1999]["mean"].values[0])
        print("avg", col_agg, "in 1999 for righties:", df_col_agg_R[df_col_agg_R["yearID"] == 1999]["mean"].values[0])
    
    axs[0].set_title("A) Average salary per year")
    axs[1].set_title("B) Avg. opponents' batting average per year")
    axs[0].set_xlabel("Year")
    axs[1].set_xlabel("Year")
    plt.legend()
    ```

### T-test (of two means)

```python
# t-test for the null hypothesis that the two independent samples have identical means
# dropna if needed!
x1 = df[(df["throws"] == "L")].salary.dropna().values
x2 = df[(df["throws"] == "R")].salary.dropna().values
print(np.mean(x1), np.mean(x2))
display(scipy.stats.ttest_ind(x1, x2))
# output: (statistic, p_value)
#p_value < 0.05 : we reject null hypothesis
```

### Correlation test

```python
# pearson's correlation : amount of linear dependence
stats.pearsonr(df['IncomePerCap'],df['Employed'])
# output: (person correlation, p_value)
# p_value < 0.05 : significant correlation

# spearman's rank correlation
stats.spearmanr(df['IncomePerCap'],df['Employed'])
# output: (spearman correlation, p_value)
# p_value < 0.05, significant correlation
```

## Machine learning

### Pre-processing

- Train test split

    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=123, 
                                                    stratify=y)
    ```

- Change categorial data to numerical

    ```python
    import pandas as pd
    
    data = {'Color': ['Red', 'Green', 'Blue', 'Green', 'Red']}
    df = pd.DataFrame(data)
    
    dummies = pd.get_dummies(df['Color'], prefix='Color')
    
    # or
    (df["gender"] == "F").astype("int").values
    ```

### Machine learns

#### Regression

- logistic regression using smf
  
    ```python
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    
    mod = smf.logit(formula='y ~  x + C(discrete_x)', data=df)
    
    res = mod.fit()
    
    print(res.summary())
    ```
  
- Linear regression using smf
  
    ```python
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    
    mod = smf.ols(formula='y ~  x + C(discrete_x)', data=df)
    
    res = mod.fit()
    
    print(res.summary())
    ```

#### Clustering

- K-Means for multiple values of K
  
    - Overall example

        ```python
        from sklearn.cluster import KMeans

        MIN_CLUSTERS = 2
        MAX_CLUSTERS = 10

        # Compute number of row and columns
        COLUMNS = 3
        ROWS = math.ceil((MAX_CLUSTERS-MIN_CLUSTERS)/COLUMNS)
        fig, axs = plt.subplots(ROWS, COLUMNS, figsize=(10,8), sharey=True, sharex=True)

        # Plot the clusters
        for n_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS+1):
        current_column = (n_clusters-MIN_CLUSTERS)%COLUMNS
        current_row = (n_clusters-MIN_CLUSTERS)//COLUMNS
        # Get the axis where to add the plot
        ax = axs[current_row, current_column]
        # Cluster the data with the current number of clusters
        kmean = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        # Plot the data by using the labels as color
        ax.scatter(X[:,0], X[:,1], c=kmean.labels_, alpha=0.6)
        ax.set_title("%s clusters"%n_clusters)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        # Plot the centroids
        for c in kmean.cluster_centers_:
            ax.scatter(c[0], c[1], marker="+", color="red")

        plt.tight_layout()
        ```

    - silhouette score to find optimal k for k-means
    
        ```python
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        
        silhouettes = []
        
        # Try multiple k
        for k in range(2, 11):
            # Cluster the data and assigne the labels
            labels = KMeans(n_clusters=k, random_state=10).fit_predict(X)
            # Get the Silhouette score
            score = silhouette_score(X, labels)
            silhouettes.append({"k": k, "score": score})
        
        # Convert to dataframe
        silhouettes = pd.DataFrame(silhouettes)
        
        # Plot the data
        plt.plot(silhouettes.k, silhouettes.score)
        plt.xlabel("K")
        plt.ylabel("Silhouette score")
        ```

    - elbow method to find optimal k for k-means
    
        ```python
        from sklearn.cluster import KMeans
        
        def plot_sse(features_X, start=2, end=11):
            sse = []
            for k in range(start, end):
                # Assign the labels to the clusters
                kmeans = KMeans(n_clusters=k, random_state=10).fit(features_X)
                sse.append({"k": k, "sse": kmeans.inertia_})
        
            sse = pd.DataFrame(sse)
            # Plot the data
            plt.plot(sse.k, sse.sse)
            plt.xlabel("K")
            plt.ylabel("Sum of Squared Errors")
        
        plot_sse(X)
        ```

    - DBSCAN for multiple values of epsilon
    
        ```python
        from sklearn.cluster import DBSCAN
        
        # Create a list of eps
        eps_list = np.linspace(0.05, 0.15, 14)
        
        # Compute number of row and columns
        COLUMNS = 7
        ROWS = math.ceil(len(eps_list)/COLUMNS)
        
        fig, axs = plt.subplots(ROWS, COLUMNS, figsize=(12, 4), sharey=True, sharex=True)
        
        for i in range(0, len(eps_list)):
          eps = eps_list[i]
        
          current_column = i%COLUMNS
          current_row = i//COLUMNS
        
          ax = axs[current_row, current_column]
          labels = DBSCAN(eps=eps).fit_predict(X_moons)
          ax.scatter(X_moons[:,0], X_moons[:,1], c=labels, alpha=0.6)
          ax.set_title("eps = {:.3f}".format(eps))
        
        plt.tight_layout()
        ```

### Finetune parameters (Grid search)

```python
# 2. Import libraries and modules
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib 

# 3. Load red wine data.
dataset_url = '<https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv>'
data = pd.read_csv(dataset_url, sep=';')

# 4. Split data into training and test sets
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                  test_size=0.2, 
                                                  random_state=123, 
                                                  stratify=y)

# 5. Declare data preprocessing steps
pipeline = make_pipeline(preprocessing.StandardScaler(),
                       RandomForestRegressor(n_estimators=100,
                                             random_state=123))

# 6. Declare hyperparameters to tune
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                'randomforestregressor__max_depth': [None, 5, 3, 1]}

# 7. Tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train, y_train)

# 8. Refit on the entire training set
# No additional code needed if clf.refit == True (default is True)

# 9. Evaluate model pipeline on test data
pred = clf.predict(X_test)
print( r2_score(y_test, pred) )
print( mean_squared_error(y_test, pred) )
print("Accuracy:", accuracy_score(y_test, pred))

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# 10. Save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
# To load: clf2 = joblib.load('rf_regressor.pkl')
```

### Evaluation

- ROC

    ```python
    from sklearn.metrics import roc_auc_score
    print("roc score", roc_auc_score(y, y_pred))
    ```

- Confusion matrix

    ```python
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # true_labels: the true labels of the test set
    # predicted_labels: the labels predicted by your model
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    ```

## Text Data

- Remove special characters like \n and \t

    ```python
    text = [" ".join(b.split()) for b in text]
    ```

- Entities extraction

    ```python
    for ent in doc.ents:
      print(ent.text, ent.label_)
    ```

- Removing stop words

    ```python
    import spacy
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    non_stop_words= [token.text for token in doc if not token.is_stop]
    ```

- Noun Chunks

    ```python
    for chunk in doc.noun_chunks:
      print(chunk.text)
    ```

- Counting word occurences

    ```python
    from collections import Counter

    words = [token.text for token in doc]
    word_freq = Counter(words)
    common_words = word_freq.most_common()
    ```

- Sentiment Analysis

    ```python
    import vaderSentiment
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(example)
    #The sentiment score consits of four values. Neutral, positive and negative sum to one. The final score is obtained by thresholding the compound value (e.g. +/-0.05)

    print('Negative sentiment:',vs['neg'])
    print('Neutral sentiment:',vs['neu'])
    print('Positive sentiment:',vs['pos'])
    print('Compound sentiment:',vs['compound'])
    ```

- Bag of words representation

    ```python
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()

    #initialize and specify minumum number of occurences to avoid untractable number of features
    #vectorizer = CountVectorizer(min_df = 2) if we want high frequency

    #create bag of words features
    X = vectorizer.fit_transform(chunks)
    ```

- Topic detection

    ```python
    import pyLDAvis.gensim_models
    STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
    
    processed_docs = list()
    for doc in nlp.pipe(chunks, n_process=5, batch_size=10):
    
      # Process document using Spacy NLP pipeline.
      ents = doc.ents  # Named entities
    
      # Keep only words (no numbers, no punctuation).
      # Lemmatize tokens, remove punctuation and remove stopwords.
      doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    
      # Remove common words from a stopword list and keep only words of length 3 or more.
      doc = [token for token in doc if token not in STOPWORDS and len(token) > 2]
    
      # Add named entities, but only if they are a compound of more than word.
      doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
      processed_docs.append(doc)
    docs = processed_docs
    del processed_docs
    
    # Add bigrams too
    from gensim.models.phrases import Phrases
    
    # Add bigrams to docs (only ones that appear 15 times or more).
    bigram = Phrases(docs, min_count=15)
    
    for idx in range(len(docs)):
      for token in bigram[docs[idx]]:
          if '_' in token:
              # Token is a bigram, add to document.
              docs[idx].append(token)
    
    # models
    from gensim.models import LdaMulticore
    params = {'passes': 10, 'random_state': seed}
    base_models = dict()
    model = LdaMulticore(corpus=corpus, num_topics=4, id2word=dictionary, workers=6,
                  passes=params['passes'], random_state=params['random_state'])
    
    # plot topics
    data =  pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
    pyLDAvis.display(data)
    
    # assignment
    sent_to_cluster = list()
    for n,doc in enumerate(corpus):
      if doc:
          cluster = max(model[doc],key=lambda x:x[1])
          sent_to_cluster.append(cluster[0])
    ```
### Regex (Regular Expressions)

- Simply use pandas to match strings

    ```python
    # find a pattern
    df['Column'].str.contains('pattern')
    # Start of a String
    df['Column'].str.startswith('start_pattern')
    # end of a string
    df['Column'].str.endswith('end_pattern')
    # match with any character (except a new line)
    df['Column'].str.contains('a.b')
    # match a set. [abc]: Matches any single character 'a', 'b', or 'c'.
    df['Column'].str.contains('[aeiou]')
    # match a range. [0-9]: Matches any digit from 0 to 9.
    df['Column'].str.contains('[0-9]')
    # multiple chars
    df['Column'].str.contains('a{2}')  # Matches 'aa'
    df['Column'].str.contains('a{2,4}')  # Matches 'aa', 'aaa', or 'aaaa'
    # match a word boundary
    df['Column'].str.contains(r'\\bword\\b')
    # exluding this pattern
    df['Column'].str.contains(r'^(?!exclude_pattern).*$')
    ```

- Re

```python
import re

def extract_matched_strings(text):
  pattern = r'\\d+'  # Example: Match one or more digits
  matches = re.findall(pattern, text)
  return ', '.join(matches) if matches else None

data = {'Column1': ['abc123', 'def456', 'xyz789']}
df = pd.DataFrame(data)

df['Matched'] = df['Column1'].apply(extract_matched_strings)
```

- Regex examples

  Search and test here:

  [regex101: build, test, and debug regex](https://regex101.com/)  
  [pyrexp: test, and visualize regex](https://pythonium.net/regex)  

## Network analysis

### Generate the network

```python
import networkx as nx

# Assuming 'nodes_df' is the DataFrame for nodes and 'edges_df' is the DataFrame for edges
G = nx.Graph()

# Adding nodes with attributes
for index, node_data in nodes_df.iterrows():
  G.add_node(node_data['node_id'], attr1=node_data['attribute1'], attr2=node_data['attribute2'])

# Adding edges
for index, edge_data in edges_df.iterrows():
  G.add_edge(edge_data['source'], edge_data['target'], weight=edge_data['weight'])

======================================================================
# or
G = nx.from_pandas_edgelist(pd.read_csv("./to_push_as_is/wiki-RfA.csv.gz"), 
                          'SRC', 'TGT', ['VOT', 'RES', 'YEA', 'DAT'], create_using=nx.Graph)
```

### Describe the network

```python
# Basic information about the network
print(nx.info(G))

# List of nodes with attributes
for node, data in G.nodes(data=True):
  print(f"Node {node}: {data}")

# List of edges with attributes
for edge, data in G.edges(data=True):
  print(f"Edge {edge}: {data}")

# in-degree and out-degree (for DiGraph)
print(sorted(dict(G_.out_degree()).values()))

# Helper function for printing various graph properties
def describe_graph(G):
  print(G)
  if nx.is_connected(G):
      print("Avg. Shortest Path Length: %.4f" %nx.average_shortest_path_length(G))
      print("Diameter: %.4f" %nx.diameter(G)) # Longest shortest path
  else:
      print("Graph is not connected")
      print("Diameter and Avg shortest path length are not defined!")
  print("Sparsity: %.4f" %nx.density(G))  # #edges/#edges-complete-graph
  # #closed-triplets(3*#triangles)/#all-triplets
  print("Global clustering coefficient aka Transitivity: %.4f" %nx.transitivity(G))
```

### Get attributes

```python
# iterate over attributes
for node, attr in G.nodes(data=True):
  # use node and attr
for u, v, attr in G.edges(data=True):
  # use u, v, and attr

# !! for multiDiGraph, G.edges funtion return triples (u,v,k)
# !! k represents the k-th time of the multi egde

# filter network based on attributes
edges_2004 = [i for i, v in nx.get_edge_attributes(G, "YEA").items() if v == 2004]
```

- Subgraph

      ```python
      # edge subgraph
      edges_2004 = [i for i, v in nx.get_edge_attributes(G, "YEA").items() if v == 2004]
      G_2004 = G.edge_subgraph(edges_2004)
      
      # node subgraph
      nodes_2004 = [n for n, attr in G.nodes(data=True) if attr.get('Year') == 2004]
      G_2004 = G.subgraph(nodes_2004)
      ```

### Visualize the network

- Network itself

    ```python
    import matplotlib.pyplot as plt

    # Basic visualization
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()

    # More customized visualization
    pos = nx.spring_layout(G)  # You can use other layout algorithms
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8, edge_color='gray', linewidths=0.5)
    plt.show()
    ```

- Plot the degree distribution (CDF or CCDF)

    ```python
    sns.ecdfplot(list(dict(G.degree()).values()), complementary=True)
    plt.xscale("log")
    # plt.axvline(10)
    # plt.axhline(0.4)
    plt.title("Complementary CDF")
    plt.xlabel("Degree centrality")
    ```

- Degree distribution, but the X-axis is the n-th data point

    ```python
    indegree = sorted(dict(G.in_degree()).values(), reverse=True)
    outdegree = sorted(dict(G.out_degree()).values(), reverse=True)
    indegree = np.array(indegree)
    outdegree = np.array(outdegree)

    hired_percentage = indegree / sum(indegree)
    output_percentage = outdegree / sum(outdegree)

    plt.plot(hired_percentage.cumsum(), label='Percentage of students hired by the N universities that hire most')
    plt.plot(output_percentage.cumsum(), label='Percentage of students output by the N universities that output most')
    plt.legend()
    plt.show()
    ```

- Calculate metrics

    ```python
    # Sparsity of the network
    sparsity = nx.density(G)
    print(f"Network Sparsity: {sparsity}")
    
    # Node degree centrality
    degree_centrality = nx.degree_centrality(G)
    print(f"Node Degree Centrality: {degree_centrality}")
    
    # Edge betweenness centrality
    edge_betweenness = nx.edge_betweenness_centrality(G)
    print(f"Edge Betweenness Centrality: {edge_betweenness}")
    
    # Clustering coefficient
    clustering_coefficient = nx.average_clustering(G)
    print(f"Average Clustering Coefficient: {clustering_coefficient}")
    ```
