

 #In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m1_survey_data.csv')


# In[4]:


data.head()


# In[4]:


data.dtypes


# In[5]:


len(data)


# In[5]:


data.shape


# In[73]:


ty = data[data.duplicated()]
print (ty)


# In[27]:


data_removedup = data.drop_duplicates()
print (data_removedup)


# In[26]:


data_removedup.isnull().sum()


# In[28]:


num_vars = data_removedup.columns[data.dtypes != 'object']

print(num_vars)


# In[126]:


data_removedup[num_vars].head(10)


# In[7]:


data_removedup.describe()


# In[10]:


data_removedup['EdLevel'].isnull().sum()


# In[11]:


data_removedup['WorkLoc'].value_counts()


# In[12]:


df2 = data_removedup['WorkLoc'].replace('np.nan', 'Office', regex=True)


# In[13]:


df2.isnull().sum()


# In[10]:


df2 = data_removedup['WorkLoc'].replace ( 'NaN', 'Office')
df2.isnull().sum()


# In[9]:


data_removedup['WorkLoc'].fillna('Office', inplace = True)


# In[7]:


data_removedup['WorkLoc'].isnull().sum()


# In[8]:


data_removedup['Employment'].value_counts()


# In[18]:


data_removedup['UndergradMajor'].value_counts()


# In[19]:


data_removedup['ConvertedComp'].isnull().sum()


# In[20]:


data_removedup['ConvertedComp'].value_counts()


# In[21]:


data_removedup['CompFreq'].value_counts()


# In[22]:


data_removedup['CompTotal'].value_counts()


# In[121]:


def f(row):
    if row['CompFreq'] == 'Yearly':
        val = row['CompTotal']
    elif row['CompFreq'] == 'Monthly':
        val = row['CompTotal'] * 12
    else:
        val = row['CompTotal'] * 52
    return val
data_removedup['NormalizedAnnualCompensation'] = data_removedup.apply(f, axis=1)


# In[123]:


data_removedup[['NormalizedAnnualCompensation','CompTotal', 'CompFreq']]


# In[13]:


import matplotlib.pyplot as plt
from scipy.stats import norm


# In[14]:


dt = data_removedup[['ConvertedComp']]
plt.plot(dt, norm.pdf(dt, 1.315967e+05	, 2.947865e+05), color ='red')


# In[29]:


data_removedup['Gender'].value_counts()


# In[19]:


data_removedup.hist(column='ConvertedComp',
        grid = False,
        figsize=(14, 6),
        legend=True,
        orientation='vertical',
        color='red');


# In[20]:


import pandas as pd
import seaborn as sns
sns.set(style="darkgrid", font_scale=1.2)


# In[27]:


sns.displot(
  data=data_removedup,
  x="ConvertedComp",
  kind="hist",
    bins = 20
)


# In[64]:


def g(row):
    if row['Gender'] == 'Woman':
        vall = row['ConvertedComp']
    else:
        vall = 0
    return vall
data_ = data_removedup.apply(g, axis=1)


# In[70]:


data_.isnull().sum()


# In[55]:


data_removedup['Woman'].value_counts()


# In[38]:


data_removedup['Age'].isnull().sum()


# In[39]:


data_removedup['Respondent'].isnull().sum()


# In[37]:


data_removedup['Age'].fillna(30, inplace = True)


# In[49]:


data_removedup['CompTotal'].mean()


# In[54]:


data_removedup['ConvertedComp'].mean()


# In[59]:


data_removedup['ConvertedComp'].fillna(131596.7, inplace = True)


# In[58]:


data_removedup['CompTotal'].fillna(757047.7, inplace = True)


# In[35]:


data_removedup['Age'].mean()


# In[64]:


data_removedup['WorkWeekHrs'].isnull().sum()


# In[62]:


data_removedup['WorkWeekHrs'].mean()


# In[63]:


data_removedup['WorkWeekHrs'].fillna(data_removedup['WorkWeekHrs'].mean(), inplace = True)


# In[66]:


data_removedup['CodeRevHrs'].fillna(data_removedup['CodeRevHrs'].mean(), inplace = True)


# In[56]:


sns.displot(
  data=data_removedup,
  x="Age",
  kind="hist",
    bins = 20
)


# In[93]:


plt.figure(figsize=(16,7))
data_removedup['ConvertedComp'].plot(kind='box')
plt.semilogy();


# In[96]:


data_removedup['ConvertedComp'].median()


# In[97]:


lower_limit = data_removedup['ConvertedComp'].quantile(0.25)
upper_limit  = data_removedup['ConvertedComp'].quantile(0.75)


# In[101]:


upper_limit = data_removedup['ConvertedComp'].mean() + 3*data_removedup['ConvertedComp'].std()
lower_limit = data_removedup['ConvertedComp'].mean() - 3*data_removedup['ConvertedComp'].std()



# In[87]:


print("Highest allowed",data_removedup['ConvertedComp'].mean() + 3*data_removedup['ConvertedComp'].std())
print("Lowest allowed",data_removedup['ConvertedComp'].mean() - 3*data_removedup['ConvertedComp'].std())


# In[90]:


data_removedup[(data_removedup['ConvertedComp'] > 173429.90802493144) | (data_removedup['ConvertedComp'] < 35771.53725813025)]


# In[104]:


data_removedup['ConvertedComp'].describe()


# In[145]:


pd.set_option("display.max_columns", None)
# display the dataframe head
data_removedup.head()


# In[268]:


data_removedup['DatabaseWorkedWith'].value_counts()


# In[103]:


data_removedup['ConvertedComp'] = np.where(
    data_removedup['ConvertedComp']>upper_limit,
    upper_limit,
    np.where(
        data_removedup['ConvertedComp']<lower_limit,
        lower_limit,
        data_removedup['ConvertedComp']
    )
)


# In[119]:


plt.figure(figsize=(16,7))
data_removedup['Age'].plot(kind='box')



# In[130]:


print(np.where(data_removedup['Age']>60))


# In[128]:


fig, ax = plt.subplots(figsize = (18,10))
ax.scatter(data_removedup['Age'], data_removedup['WorkWeekHrs'])
 

plt.show()


# In[135]:


# scatter plot with scatter() function
# transparency with "alpha"
# bubble size with "s"
plt.figure(figsize=(16,7))
plt.scatter('WorkWeekHrs', 'CodeRevHrs', 
             s='Age',
             alpha=1, 
             data=data_removedup)
plt.xlabel("WorkWeekHrs", size=16)
plt.ylabel("CodeRevHrs", size=16)
plt.title("Bubble Plot with Matplotlib", size=18)


# In[30]:


from scipy.stats.stats import pearsonr


# In[ ]:


Index(['Respondent', 'CompTotal', 'ConvertedComp', 'WorkWeekHrs', 'CodeRevHrs',
       'Age']


# In[281]:


a = data_removedup['Age']
b = data_removedup['Respondent']
c = data_removedup['CompTotal']
d = data_removedup['ConvertedComp']
e = data_removedup['WorkWeekHrs']
f = data_removedup['CodeRevHrs']

pearsonr(a, b)



# In[283]:


pearsonr(a, c)


# In[282]:


pearsonr(a, d)


# In[284]:


pearsonr(a, e)


# In[67]:


pearsonr(a, f)


# In[105]:


get_ipython().system('wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m4_survey_data.sqlite')


# In[106]:


import sqlite3
conn = sqlite3.connect("m4_survey_data.sqlite")


# In[114]:


QUERY = "SELECT COUNT(*) FROM master"


# In[115]:


df = pd.read_sql_query(QUERY,conn)
df.head()


# In[116]:


QUERY = """
SELECT name as Table_Name FROM
sqlite_master WHERE
type = 'table'
"""
# the read_sql_query runs the sql query and returns the data as a dataframe
pd.read_sql_query(QUERY,conn)


# In[207]:


QUERY = """
SELECT ConvertedComp FROM master

"""
by = pd.read_sql_query(QUERY,conn)


# In[208]:


by


# In[213]:


by.plot(kind = 'hist',
        grid = True,
        figsize=(14, 6),
        legend=True,
        orientation='vertical',
        color='red');


# In[269]:


QUERY = """
SELECT LanguageDesireNextYear, COUNT(Respondent) as count
FROM LanguageDesireNextYear
group by LanguageDesireNextYear

order by count desc limit 10
"""
fg = pd.read_sql_query(QUERY,conn)
fg


# In[247]:


QUERY = """
SELECT DatabaseDesireNextYear, COUNT(Respondent) as count
FROM DatabaseDesireNextYear
group by DatabaseDesireNextYear
order by count desc limit 10
"""
bk = pd.read_sql_query(QUERY,conn)


# In[248]:


bk


# In[220]:


my_labels = ('PostgreSQL','MongoDB','Redis','MySQL','Elasticsearch')
bk.plot(kind = 'pie', y = 'count', labels=my_labels,autopct='%1.1f%%',figsize=(20, 10))


plt.title('Next Year')

plt.show()


# In[246]:


QUERY = 'SELECT WorkWeekHrs, CodeRevHrs from master GROUP BY Age HAVING SUM(SIGN(1-SIGN(WorkWeekHrs-CodeRevHrs)))/COUNT(*) > .5'

bb = pd.read_sql_query(QUERY,conn)
bb


# In[238]:


QUERY = """
SELECT WorkWeekHrs, CodeRevHrs, Age
FROM master where Age between 30 and 35 group by Age 
"""
tk = pd.read_sql_query(QUERY,conn)
tk


# In[273]:


QUERY = """
SELECT ConvertedComp
FROM master where Age between 45 and 60 group by Age 
"""
ck = pd.read_sql_query(QUERY,conn)
ck


# In[277]:


data_removedup['MainBranch'].value_counts()


# In[275]:


ck.plot.line(figsize=(20, 10))


# In[279]:


QUERY = """
SELECT MainBranch, count(*) as count
FROM master  group by MainBranch
"""
cyk = pd.read_sql_query(QUERY,conn)
cyk


# In[280]:


cyk.plot.barh(x='MainBranch', y='count', figsize=(20, 10))


# In[240]:


tk.plot.bar(x='Age', stacked=True, title='The number of Hrs', figsize=(20, 10))


# In[175]:


df2 = dff[:5].copy()
df2


# In[285]:


conn.close()

