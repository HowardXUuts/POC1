import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest
import xgboost as xgb
import matplotlib.pyplot as plt

# Add a banner image at the top of the page
banner_image_path = "SIME.png"  # Update with the correct path to your PNG image
st.image(banner_image_path, use_container_width =True)

# Load and clean data
dataset_path = "Results11.csv"  # Update path to your dataset
data = pd.read_csv(dataset_path)

# Drop irrelevant columns from the dataset
for column in ['RegoNo', 'EstimatedDeliveryDate', 'StockNo', 'Stock','SaleslogType','OrderNo','Site2','Department','No_Days','ModelID','DealNo','Site','CleanPhone']:
    if column in data.columns:
        data = data.drop(columns=[column])

# Fill missing values in the dataset
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

categorical_columns = data.select_dtypes(include=['object']).columns
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

# Convert categorical features to numeric using LabelEncoder
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature selection for the model
target_column = 'InvoiceAmount'
X = data.drop(columns=[target_column])
y = data[target_column]

k = 7  # Number of top features to select (updated to 7)
selector = SelectKBest(score_func=f_regression, k=k)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Convert X_selected back to a DataFrame with proper feature names
X_selected_df = pd.DataFrame(X_selected, columns=selected_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y, test_size=0.2, random_state=42)

# Scaling features and retaining feature names
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_features)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=selected_features)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Train Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Added Random Forest
rf_model.fit(X_train_scaled, y_train)

# Evaluate models
xgb_predictions = xgb_model.predict(X_test_scaled)
dt_predictions = dt_model.predict(X_test_scaled)
rf_predictions = rf_model.predict(X_test_scaled)  # Random Forest predictions

xgb_r2 = r2_score(y_test, xgb_predictions)
dt_r2 = r2_score(y_test, dt_predictions)
rf_r2 = r2_score(y_test, rf_predictions)  # Random Forest R-squared

# Choose the best model
best_r2 = max(xgb_r2, dt_r2, rf_r2)  # Get the highest R-squared value
if best_r2 == xgb_r2:
    best_model = xgb_model
    best_model_name = "XGBoost"
elif best_r2 == dt_r2:
    best_model = dt_model
    best_model_name = "Decision Tree"
else:
    best_model = rf_model
    best_model_name = "Random Forest"

# Streamlit App UI
st.title('SIME Used Car Price Estimation')

# Create a two-column layout
left_col, right_col = st.columns([2, 1])

with left_col:
    st.write("Please input the categorical feature name and value:")

    inputs = {}

    # User Input for Selected Features
    for feature in selected_features:
        if feature in categorical_columns:
            # Handle categorical features
            categories = label_encoders[feature].classes_  # Get original categories
            selected_label = st.selectbox(f"Select {feature}", categories)
            inputs[feature] = label_encoders[feature].transform([selected_label])[0]

        else:
            # For numeric features, provide a number input
            inputs[feature] = st.number_input(f"Enter {feature}", value=0.0, min_value=0.0, max_value=200000.0)

    # Create a DataFrame for the input data using the inputs dictionary
    input_df = pd.DataFrame([inputs])

    # Ensure the input dataframe has the same columns as the selected features
    missing_features = set(selected_features) - set(input_df.columns)
    for feature in missing_features:
        input_df[feature] = np.nan

    # Reorder the columns of input_df to match the selected_features order
    input_df = input_df[selected_features]

    # Scale input data using the same scaler that was fit on the training data
    input_data_scaled = pd.DataFrame(scaler.transform(input_df), columns=selected_features)


with right_col:
    # Place Suggested Purchase Price Button at the bottom-right
    st.write("")  # Space before button
    if st.button('Suggested Purchase Price'):
        # Check if predicted_selling_price is available
        if 'predicted_selling_price' in st.session_state:
            predicted_selling_price = st.session_state.predicted_selling_price
            margin_percentage = st.slider('% Margin', min_value=1, max_value=30, step=1, value=10)
            st.write(f"Margin selected: {margin_percentage}%")

            # Calculate Suggested Purchase Price
            suggested_purchase_price = predicted_selling_price / (1 + margin_percentage / 100)
            st.write(f"Suggested Purchase Price is: ${suggested_purchase_price:.2f}")
        else:
            st.error("Please calculate the predicted selling price first!")

    # Predicted Selling Price Button at the top-right
    if st.button('Predicted Selling Price'):
        predicted_selling_price = best_model.predict(input_data_scaled)[0]
        st.session_state.predicted_selling_price = predicted_selling_price  # Store value in session state
        st.write(f"The predicted Selling Price is: ${predicted_selling_price:.2f}")

# Display the best model and its accuracy
st.write("### Model Details:")
st.write(f"Best Model: {best_model_name}")
st.write(f"Best Model Test Score (R-squared): {best_r2 * 100:.2f}%")

# Display a chart for the top 7 features with the updated colors
st.sidebar.write("### Top 6 Features")
feature_scores = selector.scores_[selector.get_support()]
feature_scores_percentage = (feature_scores / feature_scores.sum()) * 100

sorted_features = sorted(zip(selected_features, feature_scores_percentage), key=lambda x: x[1], reverse=True)
sorted_features_names = [x[0] for x in sorted_features]
sorted_features_values = [x[1] for x in sorted_features]

# Plotting the bar chart with the Warm and Elegant style and light gray background
plt.figure(figsize=(6, 8))
plt.barh(sorted_features_names, sorted_features_values, color='#FF6F00')  # Orange bars
plt.gca().set_facecolor('#D3D3D3')  # Light gray background for the entire chart
plt.xlabel('Feature Importance (%)', color='#333333')
plt.ylabel('Features', color='#333333')
plt.title('Top 7 Features by Importance', color='#333333')
plt.gca().invert_yaxis()

st.sidebar.pyplot(plt)

# Move the collected inputs section to the sidebar under the Top 7 Features and display as 7 rows * 1 column table
st.sidebar.write("### Collected Inputs")
inputs_table = pd.DataFrame(list(inputs.items()), columns=["Feature", "Value"])
st.sidebar.table(inputs_table)

# Add a sidebar section for saving the predicted selling price
st.sidebar.write("### Save Predicted Selling Price")

if 'predicted_selling_price' in st.session_state:
    # Display the current predicted selling price in the sidebar
    st.sidebar.write(f"Predicted Selling Price: ${st.session_state.predicted_selling_price:.2f}")

if st.sidebar.button("Save Predicted Price"):
    # Decode categorical values back to their original form
    decoded_inputs = {}
    for feature, value in inputs.items():
        if feature in label_encoders:  # Decode only if it's a categorical column
            decoded_inputs[feature] = label_encoders[feature].inverse_transform([value])[0]
        else:
            decoded_inputs[feature] = value

    # Add the predicted price to the inputs
    decoded_inputs['Predicted Selling Price'] = f"${st.session_state.predicted_selling_price:.2f}"

    # Convert to DataFrame (as a single row)
    output_df = pd.DataFrame([decoded_inputs])

    # Save to CSV
    csv_data = output_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download Predicted Price CSV",
        data=csv_data,
        file_name="predicted_price.csv",
        mime="text/csv",
    )
    st.sidebar.success("Predicted selling price saved!")
