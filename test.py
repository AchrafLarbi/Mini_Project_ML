import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import networkx as nx

# Load the dataset
file_path = r"D:\Users\pc\OneDrive\Documents\4cp\S1\ML\mini_projet\crop_yield.csv"
data = pd.read_csv(file_path)

# Data Preprocessing
data['Rainfall_mm'] = pd.cut(data['Rainfall_mm'], bins=[0, 300, 600, 1000], labels=['Low', 'Medium', 'High'])
data['Temperature_Celsius'] = pd.cut(data['Temperature_Celsius'], bins=[0, 15, 25, 35], labels=['Low', 'Medium', 'High'])
data['Yield_tons_per_hectare'] = pd.cut(data['Yield_tons_per_hectare'], bins=[0, 3, 7, 10], labels=['Low', 'Medium', 'High'])

encoded_data = pd.get_dummies(data, columns=['Region', 'Soil_Type', 'Crop', 'Rainfall_mm',
                                             'Temperature_Celsius', 'Weather_Condition',
                                             'Fertilizer_Used', 'Irrigation_Used', 'Yield_tons_per_hectare'])

transaction_data = encoded_data.drop(columns=['Days_to_Harvest'])
transaction_data = transaction_data.dropna()
transaction_data = transaction_data.sample(n=100, random_state=42)

# Sidebar for user inputs
st.sidebar.header('Apriori Algorithm Parameters')
min_support = st.sidebar.slider('Min Support', 0.01, 0.1, 0.01, 0.01)
min_confidence = st.sidebar.slider('Min Confidence', 0.5, 1.0, 0.6, 0.05)
metric = st.sidebar.selectbox('Metric', ['confidence', 'lift'])
# Sidebar for chart options
st.sidebar.header('Chart Options')
chart_option = st.sidebar.selectbox('Select Chart', 
    ['Lift vs Confidence Scatter Plot', 
     'Association Rules Graph', 
     'Top 10 Rules by Lift Bar Plot', 
     'Antecedents Frequency Pie Chart', 
     'Antecedent-Consequence Heatmap'])
# Apply Apriori Algorithm
frequent_itemsets = apriori(transaction_data, min_support=min_support, use_colnames=True)

# Generate association rules with the correct number of itemsets
rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))

# Convert frozenset to a more readable string format
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Display rules in Streamlit
st.title('Association Rules')
st.write("Top 10 Association Rules:")
st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Save the rules to a CSV file
csv_file = "association_rules_with_cof0.6.csv"
rules.to_csv(csv_file, index=False)

# Provide a download button in Streamlit
st.download_button(
    label="Download Association Rules CSV",
    data=open(csv_file, 'rb').read(),
    file_name=csv_file,
    mime="text/csv"
)



# Show the selected chart
if chart_option == 'Lift vs Confidence Scatter Plot':
    fig, ax = plt.subplots()
    sns.scatterplot(x=rules['confidence'], y=rules['lift'], size=rules['support'], alpha=0.6, ax=ax)
    ax.set_title('Lift vs Confidence')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Lift')
    st.pyplot(fig)
 
elif chart_option == 'Association Rules Graph':
    # Create a graph from the association rules
    G = nx.DiGraph()

    for index, row in rules.iterrows():
        for antecedent in row['antecedents']:
            for consequent in row['consequents']:
                G.add_edge(antecedent, consequent, weight=row['lift'])

    # Plot the graph
    fig, ax = plt.subplots(figsize=(20, 20))
    pos = nx.spring_layout(G, k=0.5, iterations=20)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='lightblue', edge_color='gray', node_size=2000, ax=ax)
    ax.set_title('Association Rules for Crop Planning')
    st.pyplot(fig)
elif chart_option == 'Top 10 Rules by Lift Bar Plot':
    # Bar plot of top 10 rules by lift
    top_rules = rules.nlargest(10, 'lift')
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(x=top_rules['lift'].round(4), y=top_rules.index, palette='viridis', ax=ax)
    ax.set_title('Top 10 Rules by Lift')
    ax.set_xlabel('Lift')
    ax.set_ylabel('Rules')
    st.pyplot(fig)

elif chart_option == 'Antecedents Frequency Pie Chart':
    # Pie chart of antecedents frequency
    antecedents_flat = [item for sublist in rules['antecedents'] for item in sublist]
    antecedents_counts = pd.Series(antecedents_flat).value_counts()

    fig, ax = plt.subplots(figsize=(16, 16))
    antecedents_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='Pastel1', ax=ax)
    ax.set_title('Distribution of Antecedents')
    ax.set_ylabel('')
    st.pyplot(fig)

elif chart_option == 'Antecedent-Consequence Heatmap':
    # Create a pivot table of antecedents vs consequents with support as values
    heatmap_data = rules.pivot_table(index='antecedents', columns='consequents', values='support', fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f", ax=ax)
    ax.set_title('Heatmap of Antecedent-Consequence Support')
    ax.set_xlabel('Consequents')
    ax.set_ylabel('Antecedents')
    st.pyplot(fig)