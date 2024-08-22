import streamlit as st
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/deeplearning (1)/deeplearning/online_shoppers_intention.csv')

visitor_type_counts = df['VisitorType'].value_counts()
st.bar_chart(visitor_type_counts)

import plotly.express as px

os_counts = df['OperatingSystems'].value_counts()
fig = px.bar(os_counts, x=os_counts.index, y=os_counts.values, 
             labels={'x': 'Operating System', 'y': 'Count'},
             title='Distribution of Operating Systems')
st.plotly_chart(fig)

import matplotlib.pyplot as plt

st.subheader("Comparison of Traffic Types")
traffic_counts = df['TrafficType'].value_counts()
fig, ax = plt.subplots()
ax.bar(traffic_counts.index, traffic_counts.values)
plt.xlabel("Traffic Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)


import streamlit as st
import pandas as pd
import plotly.express as px

df_traffic = df.groupby('Month')['PageValues'].sum().reset_index()

fig = px.line(df_traffic, x='Month', y='PageValues', title='Website Traffic Over Time')

st.plotly_chart(fig)

monthly_activity = df.groupby('Month').sum()

fig = px.line(monthly_activity, 
              x=monthly_activity.index, 
              y=['Administrative', 'Informational', 'ProductRelated'],
              title='Monthly Website Activity')

st.plotly_chart(fig)

import seaborn as sns

st.subheader("Relationship between Time Spent and Revenue")
fig, ax = plt.subplots()
sns.scatterplot(x="Administrative_Duration", y="Informational_Duration", hue="Revenue", data=df, ax=ax)
plt.xlabel("Administrative Duration")
plt.ylabel("Informational Duration")
st.pyplot(fig)

st.write("**Correlation:**")
corr_admin = df["Administrative_Duration"].corr(df["Revenue"])
corr_info = df["Informational_Duration"].corr(df["Revenue"])
st.write(f"Administrative Duration vs. Revenue: {corr_admin:.2f}")
st.write(f"Informational Duration vs. Revenue: {corr_info:.2f}")

weekend_visits = df['Weekend'].sum()
total_visits = len(df)
weekend_proportion = weekend_visits / total_visits

fig = px.pie(
    values=[weekend_proportion, 1 - weekend_proportion],
    names=['Weekend', 'Weekday'],
    title='Proportion of Visits on Weekends vs. Weekdays'
)

st.plotly_chart(fig)

import numpy as np
import plotly.graph_objects as go

regions = ['North', 'South', 'East', 'West']
products = ['Product A', 'Product B', 'Product C']
sales_data = np.random.rand(len(regions), len(products))

fig = go.Figure(data=go.Heatmap(
                   z=sales_data,
                   x=products,
                   y=regions,
                   hoverongaps = False))
fig.update_layout(title='Regional Preferences for Products',
                  xaxis_title='Products',
                  yaxis_title='Regions')


st.plotly_chart(fig)

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('/content/drive/MyDrive/deeplearning (1)/deeplearning/online_shoppers_intention.csv')

y = df['Revenue']
X = df.drop('Revenue', axis=1)
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

st.title("Online Shopper Intention Prediction")

user_input = {}
for feature in X.columns:
    user_input[feature] = st.number_input(feature, value=0)

user_df = pd.DataFrame([user_input])
prediction = model.predict(user_df)[0]

if prediction == 1:
    st.write("This shopper is likely to generate revenue.")
else:
    st.write("This shopper is unlikely to generate revenue.")

























