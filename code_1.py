
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_csv(filename):
    """
    reads a CSV file, skips the first four rows, and drops some unnecessary columns. 
    It then transforms the data from wide to long format using the melt function, pivots it 
    using the pivot_table function, and drops null values. The resulting dataframe is then split into two dataframes - 
    one with years as columns and the other with countries as columns. 
    return: <pd.DataFrame>, <pd.DataFrame> 
    """

    # read CSV file and skip first four rows
    df = pd.read_csv(filename, skiprows=4)

    # drop unnecessary columns
    df = df.drop(columns=['Country Code', 'Indicator Code', 'Unnamed: 66'])

    # melt the dataframe to long format
    df = df.melt(id_vars=['Country Name', 'Indicator Name'], var_name='Year', value_name='Value')
    df = df.pivot_table(values='Value', columns='Indicator Name', index=['Country Name', 'Year']).reset_index()

    # drop null values
    df = df.dropna(thresh=int(0.25 * df.shape[0]), axis=1)
    df = df.dropna(thresh=int(0.25 * df.shape[1]))
   
    # convert years to columns
    df_years = df.set_index(['Year', 'Country Name']).unstack(level=0).swaplevel(axis=1).sort_index(axis=1, level=0)

    # convert countries to columns
    df_countries = df.set_index(['Year', 'Country Name']).unstack(level=1).swaplevel(axis=1).sort_index(axis=1, level=0)
        
    return df_years, df_countries


df_years, df_countries = read_csv(r"C:\Users\Harsha V\Downloads\DataViz\world-bank-data-analysis\wbcc.csv")

# create a dataframe with country and year as columns for data cleaning
df = df_countries.unstack().unstack(level=1).reset_index()

# describe the dataframe and save in a csv format
df.describe().to_csv("desc.csv")

# let's randomly choose 6 countries and 10 indicators for our analysis.
# we can repeat this step until desired combination is obtained
countries = random.sample(df['Country Name'].unique().tolist(), 6)
indicators = random.sample(df.columns.to_list(), 10)
print("We have chosen the following countries for our analysis: ", countries)
print("We have chosen the following indicators for our analysis: ", indicators)

# plot a heatmap using seaborn chart for Canada
xticks=yticks=[i.split('(')[0].strip() for i in indicators]
ax = sns.heatmap(df_countries['Canada'].corr(), annot=True)
ax.set_title(f"Correlation Matrix for Canada")
ax.set_xlabel(''), ax.set_ylabel(''), ax.set_xticklabels(xticks, fontsize=8), ax.set_yticklabels(yticks, fontsize=8)
plt.savefig('corrCanada.png')

# plot a heatmap using seaborn chart for India
ax = sns.heatmap(df_countries['India'].corr(), annot=True)
ax.set_title(f"Correlation Matrix for India")
ax.set_xlabel(''), ax.set_ylabel(''), ax.set_xticklabels(xticks, fontsize=8), ax.set_yticklabels(yticks, fontsize=8)
plt.savefig('corrIndia.png') # save the plot in png format

# bar plot to describe population growth of all countries
plt.figure(figsize=(6, 4))
ax = sns.barplot(x = 'Country Name', y = 'Population, total', hue = 'Year',
                data = df_years[list(range(1990, 2016, 5))].unstack().unstack(level=1).reset_index()
                )
ax.set_ylabel(''), ax.set_xlabel(''), ax.set_title('Population, total', fontsize=6)
ax.set_xticklabels(countries, fontsize=8)
plt.savefig("populationBarPlot.png")


# bar plot to visualize Agricultural land area in different countries
plt.figure(figsize=(6,4))
ax = sns.barplot(x = 'Country Name', y = "Agricultural land (sq. km)", hue = 'Year',
                data = df_years[list(range(1990, 2016, 5))].unstack().unstack(level=1).reset_index()
                )
ax.set_ylabel(''), ax.set_xlabel(''), ax.set_title("Agricultural land (sq. km)", fontsize=6)
ax.set_xticklabels(countries, fontsize=8)
plt.legend(loc='upper left', fontsize=6)
plt.savefig("agriBarPlot.png")

# create a new column for electricity production categories
df[f'catElectricityProduction'] = pd.cut(df['Electricity production from oil sources (% of total)'], bins=[0, 25, 50, 75, 100], labels=['Very Low', 'Low', 'Medium', 'High'])

# create a horizontal bar plot
plt.figure(figsize=(4,3))
ax = sns.barplot(x='Nitrous oxide emissions (thousand metric tons of CO2 equivalent)', y="catElectricityProduction", data=df, hue='Country Name')
plt.legend(loc='upper right', fontsize=6)


# create a multiple line plot
plt.figure(figsize=(4,3))
ax = sns.lineplot(x='Year', y='CO2 emissions (metric tons per capita)', hue='Country Name', data=df, dashes=True)
for i in ax.lines:
    i.set_linestyle("--")
ax.set_ylabel(ylabel='')
ax.set_title('CO2 emissions (metric tons per capita)', fontsize=8)
plt.legend(loc='upper left', fontsize='6')
plt.show()

# create a dataframe to show nitrous oxide emissions by dropping the indicator name column level
newDf = df_years[[(i, "Nitrous oxide emissions (thousand metric tons of CO2 equivalent)") for i in [1990, 2005, 2015]]]
newDf.columns = newDf.columns.droplevel(1)
newDf.to_csv('nitrousOxide.csv')

# Use crosstab to plot a bar graph (stacked) for Urban population of different countries
pd.crosstab(pd.cut(df["Urban population (% of total population)"], 10), df['Country Name']).plot.bar(stacked=True)