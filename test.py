import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import networkx as nx

# Load the dataset
file_path = r"crop_yield.csv"  # Adjust this path to your file location
data = pd.read_csv(file_path)

# Data Preprocessing
data['Rainfall_mm'] = pd.cut(data['Rainfall_mm'], bins=[0, 300, 600, 1000], labels=['Low', 'Medium', 'High'])
data['Temperature_Celsius'] = pd.cut(data['Temperature_Celsius'], bins=[0, 15, 25, 35], labels=['Low', 'Medium', 'High'])
data['Yield_tons_per_hectare'] = pd.cut(data['Yield_tons_per_hectare'], bins=[0, 3, 7, 10], labels=['Low', 'Medium', 'High'])

# Encode categorical variables as one-hot encoding
encoded_data = pd.get_dummies(data, columns=['Region', 'Soil_Type', 'Crop', 'Rainfall_mm',
                                             'Temperature_Celsius', 'Weather_Condition',
                                             'Fertilizer_Used', 'Irrigation_Used', 'Yield_tons_per_hectare'])

# Drop irrelevant column and handle missing values
transaction_data = encoded_data.drop(columns=['Days_to_Harvest'])
transaction_data = transaction_data.dropna()
transaction_data = transaction_data.sample(n=100, random_state=42)

# Streamlit Sidebar for user inputs
st.sidebar.header('Algorithm and Parameters')
algorithm_choice = st.sidebar.selectbox('Choose Algorithm', ['Apriori', 'FP-Growth'])
min_support = st.sidebar.slider('Min Support', 0.01, 0.1, 0.01, 0.01)
min_confidence = st.sidebar.slider('Min Confidence', 0.5, 1.0, 0.6, 0.05)
metric = st.sidebar.selectbox('Metric', ['confidence', 'lift'])

# Apply Apriori or FP-Growth Algorithm
st.title('Association Rules for Crop Planning')
if algorithm_choice == 'Apriori':
    st.subheader('Using Apriori Algorithm')
    frequent_itemsets = apriori(transaction_data, min_support=min_support, use_colnames=True)
else:
    st.subheader('Using FP-Growth Algorithm')
    frequent_itemsets = fpgrowth(transaction_data, min_support=min_support, use_colnames=True)

# Generate Association Rules
rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))

# Convert frozenset to readable format
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Display Association Rules in Streamlit
st.write("Top 10 Association Rules:")
st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Save rules to a CSV file and provide download option
csv_file = f"association_rules_{algorithm_choice.lower()}.csv"
rules.to_csv(csv_file, index=False)

st.download_button(
    label=f"Download {algorithm_choice} Association Rules CSV",
    data=open(csv_file, 'rb').read(),
    file_name=csv_file,
    mime="text/csv"
)

# Plot 1: Directed Graph for Association Rules
st.subheader('Association Rules Graph')
G = nx.DiGraph()
for _, row in rules.iterrows():
    antecedents = row['antecedents'].split(", ")
    consequents = row['consequents'].split(", ")
    for antecedent in antecedents:
        for consequent in consequents:
            G.add_edge(antecedent, consequent, weight=row['lift'])

fig, ax = plt.subplots(figsize=(20, 20))
pos = nx.spring_layout(G, k=0.5, iterations=20)
nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', 
        edge_color='gray', node_size=2000, font_size=10, ax=ax)
ax.set_title(f'Association Rules for Crop Planning ({algorithm_choice})')
st.pyplot(fig)  # Display the graph

# Plot 2: Pie Chart of Antecedents Frequency
st.subheader('Distribution of Antecedents')
antecedents_flat = [item for sublist in rules['antecedents'].str.split(', ') for item in sublist]
antecedents_counts = pd.Series(antecedents_flat).value_counts()

fig, ax = plt.subplots(figsize=(14, 14))
antecedents_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='Pastel1', ax=ax)
ax.set_title('Distribution of Antecedents')
ax.set_ylabel('')
st.pyplot(fig)  # Display the pie chart

# Plot 3: Lift vs Confidence Scatter Plot
st.subheader('Lift vs Confidence Scatter Plot')
fig, ax = plt.subplots(figsize=(16, 10))
scatter = ax.scatter(rules['confidence'], rules['lift'], s=rules['support'] * 1000, alpha=0.6, c='blue')
ax.set_title('Lift vs Confidence')
ax.set_xlabel('Confidence')
ax.set_ylabel('Lift')
ax.grid(True)
st.pyplot(fig)  # Display scatter plot

# Plot 4: Bar Plot of Top 10 Rules by Lift
st.subheader('Top 10 Rules by Lift')
top_rules = rules.nlargest(10, 'lift')
fig, ax = plt.subplots(figsize=(16, 6))
sns.barplot(x=top_rules['lift'].round(4), y=top_rules.index, palette='viridis', ax=ax)
ax.set_title('Top 10 Rules by Lift')
ax.set_xlabel('Lift')
ax.set_ylabel('Rules')
st.pyplot(fig)  # Display bar plot
