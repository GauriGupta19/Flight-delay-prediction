#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime, warnings, scipy 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


# In[2]:


data = pd.read_csv("flights.csv", low_memory=False)


# In[3]:


airport = pd.read_csv('airports.csv')
airlines = pd.read_csv('airlines.csv')
data_jan = data.iloc[0:469967] #To extract the data of January


# In[4]:


data_jan.info()


# In[5]:


data_jan.shape


# In[6]:


data_jan.describe()


# In[7]:


data_jan


# In[8]:


airport = airport.dropna(subset = ['LATITUDE','LONGITUDE'])


# In[9]:


airlines


# In[10]:


Data_NULL = data_jan.isnull().sum()*100/data_jan.shape[0]
Data_NULL


# In[11]:


data_jan.shape


# In[12]:


data1 = data_jan.dropna(subset = ["TAIL_NUMBER",'DEPARTURE_TIME','DEPARTURE_DELAY','TAXI_OUT','WHEELS_OFF','SCHEDULED_TIME',
             'ELAPSED_TIME','AIR_TIME','WHEELS_ON','TAXI_IN','ARRIVAL_TIME','ARRIVAL_DELAY'])


# In[13]:


data_jan.shape


# In[14]:


data3 = data1.drop(['CANCELLATION_REASON','AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY',
                    'LATE_AIRCRAFT_DELAY','WEATHER_DELAY'],axis = 1)


# In[15]:


data3.shape


# In[16]:


data3.info


# In[17]:


def Format_Hourmin(hours):
        if hours == 2400:
            hours = 0
        else:
            min = hours%100
            hours = (hours-min)/100
            time = int(60*hours+min)
            return time 


# In[18]:


# Applying the function to required variables in the dataset
data3['Actual_Departure'] =data1['DEPARTURE_TIME'].apply(Format_Hourmin)
data3['Actual_Departure'] =data1['DEPARTURE_TIME'].apply(Format_Hourmin)
data3['Scheduled_Arrival'] =data1['SCHEDULED_ARRIVAL'].apply(Format_Hourmin)
data3['Scheduled_Departure'] =data1['SCHEDULED_DEPARTURE'].apply(Format_Hourmin)
data3['Actual_Arrival'] =data1['ARRIVAL_TIME'].apply(Format_Hourmin)


# In[19]:


# Merging on AIRLINE and IATA_CODE
data3 = data3.merge(airlines, left_on='AIRLINE', right_on='IATA_CODE', how='inner')


# In[20]:


data3 = data3.drop(['AIRLINE_x','IATA_CODE'], axis=1)


# In[21]:


data3 = data3.rename(columns={"AIRLINE_y":"AIRLINE"})
data3 = data3.merge(airport, left_on='ORIGIN_AIRPORT', right_on='IATA_CODE', how='inner')
data3 = data3.merge(airport, left_on='DESTINATION_AIRPORT', right_on='IATA_CODE', how='inner')


# In[22]:


data3 = data3.drop(['LATITUDE_x', 'LONGITUDE_x',
       'STATE_y', 'COUNTRY_y', 'LATITUDE_y', 'LONGITUDE_y','STATE_x', 'COUNTRY_x'], axis=1)


# In[23]:


data3 = data3.rename(columns={'IATA_CODE_x':'Org_Airport_Code','AIRPORT_x':'Org_Airport_Name','CITY_x':'Origin_city',
                             'IATA_CODE_y':'Dest_Airport_Code','AIRPORT_y':'Dest_Airport_Name','CITY_y':'Destination_city'})


# In[24]:


data3


# In[26]:


# we are taking the required data into Account for visualization and the Analysis
# Creating Date in the Datetime format
data3['Date'] = pd.to_datetime(data3[['YEAR','MONTH','DAY']])
data3.Date
#data3['Day'] = data3['Date'].dt.weekday_name
ReqdData = pd.DataFrame(data3[['AIRLINE','Org_Airport_Name','Origin_city',
                               'Dest_Airport_Name','Destination_city','ORIGIN_AIRPORT',
                               'DESTINATION_AIRPORT','DISTANCE','Actual_Departure','Date','DAY_OF_WEEK',
                               'Scheduled_Departure','DEPARTURE_DELAY','Actual_Arrival','Scheduled_Arrival','ARRIVAL_DELAY',
                              'SCHEDULED_TIME','ELAPSED_TIME','AIR_TIME','TAXI_IN','TAXI_OUT','DIVERTED',]])


# In[27]:


data3.DEPARTURE_TIME.dtype


# In[28]:


ReqdData = ReqdData.dropna(subset = ['Actual_Departure','Actual_Arrival'])


# In[29]:


ReqdData = ReqdData.dropna(subset = ['Actual_Departure','Actual_Arrival'])
ReqdData.info()


# In[30]:


# Cleaned Dataset for visualization and Analysis
Flights = ReqdData
Flights


# In[31]:


axis = plt.subplots(figsize=(20,14))
sns.heatmap(ReqdData.corr(),annot = True)
plt.show()


# In[32]:


pca = PCA(n_components=13)

pca.fit(Flights[['DISTANCE','Actual_Departure','DAY_OF_WEEK',
                               'Scheduled_Departure','DEPARTURE_DELAY','Actual_Arrival','Scheduled_Arrival','ARRIVAL_DELAY',
                              'SCHEDULED_TIME','ELAPSED_TIME','AIR_TIME','TAXI_IN','TAXI_OUT']])

print(pca.explained_variance_ratio_)
pca_df = pd.DataFrame()


# In[33]:


covariance_df = pd.DataFrame(pca.get_covariance())
covariance_df.to_csv('covariance.csv')

correlation_df = pd.DataFrame(Flights[['DISTANCE','Actual_Departure','DAY_OF_WEEK',
                               'Scheduled_Departure','DEPARTURE_DELAY','Actual_Arrival','Scheduled_Arrival','ARRIVAL_DELAY',
                              'SCHEDULED_TIME','ELAPSED_TIME','AIR_TIME','TAXI_IN','TAXI_OUT']])
correlation_df.to_csv('correlation_df_original.csv')

pca_df['Variance_Explained'] = pca.explained_variance_ratio_
pca_df['Singular_Values']  = pca.singular_values_

cumsum  = np.cumsum(pca_df['Variance_Explained'],axis=0) 
pca_df['Cumulative Sum of Variance'] = cumsum

pca_df.to_csv('PCA_Dataframe.csv')


# In[72]:


pca_transformed_data = pca.transform(Flights[['DISTANCE','Actual_Departure','DAY_OF_WEEK',
                               'Scheduled_Departure','DEPARTURE_DELAY','Actual_Arrival','Scheduled_Arrival','ARRIVAL_DELAY',
                              'SCHEDULED_TIME','ELAPSED_TIME','AIR_TIME','TAXI_IN','TAXI_OUT']])

plt.plot(np.arange(13) + 1 ,pca_df['Cumulative Sum of Variance'])
plt.title('Variance Explained by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumultive Variance')
plt.show()

pca_transformed_df = pd.DataFrame(pca_transformed_data , index=np.arange(pca_transformed_data.shape[0]), columns=np.arange(pca_transformed_data.shape[1]))
pca_correlation_df = pd.DataFrame(np.corrcoef(pca_transformed_df))
pca_correlation_df.to_csv('pca_correlation_df.csv')
pca_transformed_df['class'] = np.asarray(df['class'])

# ScatterPlot .. do after PCA
sns.scatterplot(x=0, y=1,hue = 'class',data= pca_transformed_df)
sns.distplot( df['class'], kde=False)
sns.catplot(x="x1", y="y1", hue="gender",col="class", kind = 'bar',data=df, palette = "rainbow")


# In[73]:


X = pca_transformed_data
wcss = []
for i in range(1,20):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,20),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[77]:


num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters,init='k-means++',max_iter=300,n_init=10,random_state=0) 
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.show()
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()
print(y_kmeans)
labels = np.zeros_like(y_kmeans)
for i in range(num_clusters):
    mask = (y_kmeans == i)
    labels[mask] = mode(df['class'].loc[mask])[0]

print(accuracy_score(df['class'], labels) )

mat = confusion_matrix(df['class'], labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=np.arange(5),
            yticklabels=np.arange(5))
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[34]:


df_flights_jan=ReqdData
df_flights_jan['DELAYED'] = df_flights_jan.loc[:,'ARRIVAL_DELAY'].values > 0
# choosing the predictors
feature_list = [
    'AIRLINE'
    ,'ELAPSED_TIME'
    ,'DEPARTURE_DELAY'
    ,'SCHEDULED_TIME'
    ,'AIR_TIME'
    ,'DISTANCE'
    ,'DAY_OF_WEEK'
    ,'TAXI_IN'
    ,'TAXI_OUT'
]

X = df_flights_jan[feature_list]


# In[35]:


from sklearn.preprocessing import LabelEncoder
labelenc = LabelEncoder()

# Converting "category" airline to integer values
X.iloc[:,feature_list.index('AIRLINE')] = labelenc.fit_transform(X.iloc[:,feature_list.index('AIRLINE')])


# In[36]:


df_flights_jan['DELAYED'] = df_flights_jan.loc[:,'ARRIVAL_DELAY'].values > 0
y = df_flights_jan.DELAYED


clf = RandomForestClassifier(n_estimators=9, random_state=32) 
clf.fit(X, y)


# In[37]:


i=0
df_feature_selection = pd.DataFrame(columns=['FEATURE','IMPORTANCE'])
for val in (clf.feature_importances_):
    df_feature_selection.loc[i] = [feature_list[i],val]
    i = i + 1
    

df_feature_selection.sort_values('IMPORTANCE', ascending=False)


# In[78]:


ReqdData['Delay_Difference'] = ReqdData['DEPARTURE_DELAY'] - ReqdData['ARRIVAL_DELAY']
plt.scatter(ReqdData['Delay_Difference'],ReqdData['AIR_TIME'],s=0.9)
plt.xlabel('Delay Difference')
plt.ylabel('Air Time')
plt.show()


# In[83]:


ReqdData['Average_Air'] = ReqdData['Actual_Arrival'] + ReqdData['Actual_Departure']
plt.scatter(ReqdData['Average_Air'],ReqdData['Delay_Difference'],s=0.9)
plt.xlabel('Arrival Time + Departure Time')
plt.ylabel('Delay difference')
plt.show()


# In[84]:


plt.scatter(data3['ARRIVAL_DELAY'],data3['DEPARTURE_DELAY'])
plt.xlabel('Arrival Delay')
plt.ylabel('Departure Delay')
plt.show()
plt.scatter(data3['Actual_Departure'],data3['DEPARTURE_DELAY'], s=0.9)
plt.xlabel('Actual Departure')
plt.ylabel('Departure Delay')
plt.show()
plt.hist(ReqdData['Actual_Departure'],24)
plt.xlabel('Time of Departure')
plt.ylabel('Number of Flights')
plt.show()


# In[86]:


plt.scatter(data3['DAY_OF_WEEK'],data3['DEPARTURE_DELAY'], s=0.9)
plt.xlabel('Day of Week')
plt.ylabel('Departure Delay')
plt.show()

plt.scatter(data3['DAY_OF_WEEK'],data3['ARRIVAL_DELAY'], s=0.9)
plt.xlabel('Day of Week')
plt.ylabel('Arrival Delay')
plt.show()
plt.scatter(ReqdData['DAY_OF_WEEK'],ReqdData['Delay_Difference'], s=0.9)
plt.show()
plt.xlabel('Day of Week')
plt.ylabel('Number of Flights')
plt.hist(ReqdData['DAY_OF_WEEK'],24)
plt.show()


# In[57]:


F=ReqdData.groupby('AIRLINE').Delay_Difference.mean().to_frame().sort_values(by='Delay_Difference', ascending=False).round(3)
G=ReqdData.groupby('DAY_OF_WEEK').ARRIVAL_DELAY.mean().to_frame().sort_values(by='ARRIVAL_DELAY', ascending=False).round(3)
print(G)
H=ReqdData.groupby('DAY_OF_WEEK').DEPARTURE_DELAY.mean().to_frame().sort_values(by='DEPARTURE_DELAY', ascending=False).round(3)
print(H)


# In[58]:


# Cleaned Dataset for visualization and Analysis
Flights = ReqdData
Flights


# In[68]:


#cities with the most flights
F=ReqdData.Origin_city.value_counts().sort_values(ascending=False)[:15]
print(F)
plt.figure(figsize=(10, 10))
axis = sns.countplot(x=Flights['Origin_city'], data = Flights,
              order=Flights['Origin_city'].value_counts().iloc[:20].index)
axis.set_xticklabels(axis.get_xticklabels(), rotation=90, ha="right")
plt.tight_layout()
plt.show()


# In[70]:



F=ReqdData.Org_Airport_Name.value_counts().sort_values(ascending=False)[:20]
print(F)
plt.figure(figsize=(10, 14))
axis = sns.countplot(x=Flights['Org_Airport_Name'], data = Flights,
              order=Flights['Org_Airport_Name'].value_counts().iloc[:20].index)
axis.set_xticklabels(axis.get_xticklabels(), rotation=90, ha="right")
plt.tight_layout()
plt.show()


# In[62]:


axis = plt.subplots(figsize=(10,10))
Name = Flights["AIRLINE"].unique()
size = Flights["AIRLINE"].value_counts()
plt.pie(size,labels=Name,autopct='%5.0f%%')
plt.show()


# In[76]:


axis = plt.subplots(figsize=(10,14))
sns.despine(bottom=True, left=True)
# Observations with Scatter Plot
sns.stripplot(x="DEPARTURE_DELAY", y="AIRLINE",
              data = Flights, dodge=True, jitter=True
            )
plt.show()


# In[64]:


get_ipython().run_cell_magic('html', '', "<div class='tableauPlaceholder' id='viz1569936847474' style='position: relative'><noscript><a href='#'><img alt=' ' \nsrc='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;9B&#47;9BD44NRJ9&#47;1_rss.png' style='border: none'\n/></a></noscript><object class='tableauViz'  \nstyle='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' \n/> <param name='embed_code_version' value='3' /> <param name='path' value='shared&#47;9BD44NRJ9' \n/> <param name='toolbar' value='yes' \n/><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;9B&#47;9BD44NRJ9&#47;1.png' \n/> <param name='animate_transition' value='yes'\n/><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' \n/><param name='display_overlay' value='yes' /><param name='display_count' value='yes' \n/><param name='filter' value='publish=yes' /></object></div>                \n<script type='text/javascript'>                    \nvar divElement = document.getElementById('viz1569936847474');                    \nvar vizElement = divElement.getElementsByTagName('object')[0];                    \nvizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';                    \nvar scriptElement = document.createElement('script');                    \nscriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    \nvizElement.parentNode.insertBefore(scriptElement, vizElement);               \n</script>")


# In[65]:


# Plot to show the Taxi In and Taxi Out Time
axis = plt.subplots(figsize=(10,14))
sns.set_color_codes("pastel")
sns.set_context("notebook", font_scale=1.5)
axis = sns.barplot(x="TAXI_OUT", y="AIRLINE", data=Flights, color="g")
axis = sns.barplot(x="TAXI_IN", y="AIRLINE", data=Flights, color="r")
axis.set(xlabel="TAXI_TIME (TAXI_OUT: green, TAXI_IN: blue)")
axis = plt.subplots(figsize=(20,14))
sns.heatmap(Flights.corr(),annot = True)
plt.show()


# In[ ]:




