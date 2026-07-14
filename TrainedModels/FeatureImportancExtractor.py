import pandas as pd
import joblib
import os
import json
if os.path.exists("feature_importance.json"):
    with open("feature_importance.json", "r") as f:
        all_importances = json.load(f)
else:
    all_importances = {}
pipeline = joblib.load("randommodel.pkl")

feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
values = pipeline.named_steps['model'].feature_importances_

df = pd.DataFrame({'feature': feature_names, 'value': values})
df['abs_value'] = df['value'].abs()

# Your original columns, in the SAME order you listed them in ColumnTransformer
onehot_columns = ["car_name", "brand", "model", "fuel_type", "seller_type"]
numeric_columns = ["km_driven", "vehicle_age", "engine", "max_power", "mileage"]

aggregated = {}

# Step 1: for EACH onehot column, find every feature that belongs to it
for col in onehot_columns:
    prefix = f"onehot__{col}_"
    matching_rows = df[df['feature'].str.startswith(prefix)]
    aggregated[col] = matching_rows['abs_value'].sum()

# Step 2: numeric columns are simpler — they weren't split into multiple pieces
for col in numeric_columns:
    matching_rows = df[df['feature'].str.contains(col)]
    aggregated[col] = matching_rows['abs_value'].sum()

result = pd.Series(aggregated).sort_values(ascending=False)
result=result.reset_index()
result.columns=["label","value"]
all_importances["RandomForest"]=json_data=result.to_dict(orient="records")
print(all_importances)
with open("feature_importance.json", "w") as f:
    json.dump(all_importances, f,indent=4)
print(result)