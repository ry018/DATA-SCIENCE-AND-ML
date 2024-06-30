import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read data from CSV file
df = pd.read_csv('products_data.csv')

# Convert components to a dictionary format expected by the original code
def convert_row_to_components(row):
    components = {}
    if not pd.isna(row['Baking Mixture']):
        components['Baking Mixture'] = row['Baking Mixture']
    if not pd.isna(row['Oven']):
        components['Oven'] = row['Oven']
    if not pd.isna(row['Warehouse']):
        components['Warehouse'] = row['Warehouse']
    if not pd.isna(row['Chocolate']):
        components['Chocolate'] = row['Chocolate']
    if not pd.isna(row['Cooling System']):
        components['Cooling System'] = row['Cooling System']
    if not pd.isna(row['Assembly']):
        components['Assembly'] = row['Assembly']
    if not pd.isna(row['Strawberries']):
        components['Strawberries'] = row['Strawberries']
    return components

df['Components'] = df.apply(convert_row_to_components, axis=1)

# Calculate CO2e values for each product
df['CO2e'] = df['Components'].apply(lambda x: sum(x.values()))

# Extract features and target
X = df['Components'].apply(pd.Series)  # Features: components contributing to CO2e
y = df['CO2e']  # Target: CO2e emissions

# Initialize Random Forest Regressor
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
random_forest.fit(X, y)

# Get predictions and feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': random_forest.feature_importances_})

# Sort DataFrame by CO2e in descending order
df = df.sort_values(by='CO2e', ascending=False).reset_index(drop=True)

# Print major CO2e contributors
print("MAJOR CO2e CONTRIBUTORS:")
df['Rank'] = df.index + 1  # Assign ranks starting from 1
df['Rank'] = df['Rank'].apply(lambda x: f"#{x}")
print(df[['Rank', 'Product', 'Components', 'CO2e']])

# Print recommendations
print("\nSTRATEGIC INSIGHTS:")
for index, row in df.iterrows():
    product = row['Product']
    max_component = max(row['Components'], key=row['Components'].get)
    if max_component == 'Baking Mixture':
        print(f"For {product}, the highest CO2e component is {max_component}. Recommendation: Switch to a baking mixture supplier with lower CO2e value.")
    elif max_component == 'Oven':
        print(f"For {product}, the highest CO2e component is {max_component}. Recommendation: Switch to a Natural gas supplier with lower CO2e value or find an alternative energy source to supplement electricity consumption.")
    elif max_component == 'Warehouse':
        print(f"For {product}, the highest CO2e component is {max_component}. Recommendation: Employ usage of renewable sources of energy (solar, wind, hydro etc) to supplement electricity consumption for warehouse.")
    elif max_component == 'Chocolate':
        print(f"For {product}, the highest CO2e component is {max_component}. Recommendation: Switch to a supplier with lower CO2e for chocolate.")
    elif max_component == 'Cooling System':
        print(f"For {product}, the highest CO2e component is {max_component}. Recommendation: Optimize your cooling system by reducing its duty cycle or supplementing its energy supply with alternative eco-friendly and renewable energy sources.")
    elif max_component == 'Assembly':
        print(f"For {product}, the highest CO2e component is {max_component}. Recommendation: Employ usage of renewable sources of energy (solar, wind, hydro etc) to supplement electricity consumption for assembly section.")
    elif max_component == 'Strawberries':
        print(f"For {product}, the highest CO2e component is {max_component}. Recommendation: Switch to a strawberry supplier with lower CO2e value .")
