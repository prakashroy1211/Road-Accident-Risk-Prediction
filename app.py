import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load Model & Encoders ---
@st.cache_resource
def load_model_assets(filename='road_safety_model.joblib'):
    """
    Loads the saved model, label encoders, and feature names.
    This function is designed to load the dictionary saved by
    the RoadSafetyModel class.
    """
    try:
        saved_assets = joblib.load(filename)
        model = saved_assets['model']
        label_encoders = saved_assets['label_encoders']
        # The 'feature_names' in the new train.py might be incomplete,
        # so we'll be cautious, but we still load it.
        feature_names = saved_assets.get('feature_names') # Use .get for safety
        
        return model, label_encoders, feature_names
    except FileNotFoundError:
        st.error(f"Error: Model file '{filename}' not found. "
                 "Please run your new train.py script to create it.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, label_encoders, feature_names = load_model_assets()

# --- 2. Feature Engineering Function ---
def engineer_features_app(df):
    """
    Replicates the exact feature engineering logic from your
    RoadSafetyModel class's engineer_features method.
    """
    df = df.copy()
    
    # Interaction features
    # This will be skipped if 'road_width' isn't a column, which is correct.
    if 'num_lanes' in df.columns and 'road_width' in df.columns:
        df['lane_width'] = df['road_width'] / (df['num_lanes'] + 1)
    
    # This feature *will* be created.
    if 'curvature' in df.columns and 'speed_limit' in df.columns:
        df['curve_speed'] = df['curvature'] * df['speed_limit']
    
    # Add polynomial features.
    # Your script uses .select_dtypes().columns[:3].
    # We will explicitly list them to be robust.
    numeric_cols_to_square = ['num_lanes', 'curvature', 'speed_limit']
    
    for col in numeric_cols_to_square:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
            
    return df

# --- 3. Streamlit Page Setup ---
st.set_page_config(
    page_title="Road Accident Risk Predictor",
    page_icon="",
    layout="wide"
)
st.title("Road Accident Risk Predictor")

if model and label_encoders:
    # --- 4. Sidebar Inputs ---
    # Use the 9 original features from our dataset
    st.sidebar.header("Select Road Conditions")

    road_type = st.sidebar.selectbox(
        "Road Type", 
        ['highway', 'rural', 'Other']
    )
    
    lighting = st.sidebar.selectbox(
        "Lighting", 
        ['dim', 'daylight', 'Other']
    )
    
    weather = st.sidebar.selectbox(
        "Weather",
        ['foggy', 'clear', 'Other']
    )
    
    time_of_day = st.sidebar.selectbox(
        "Time of Day",
        ['morning', 'evening', 'Other']
    )

    road_signs_present = st.sidebar.radio(
        "Road Signs Present?",
        [True, False],
        index=0,
        format_func=lambda x: str(x) # Ensure it's treated as a string if needed
    )
    
    public_road = st.sidebar.radio(
        "Public Road?",
        [True, False],
        index=0,
        format_func=lambda x: str(x) # Ensure it's treated as a string if needed
    )

    num_lanes = st.sidebar.slider(
        "Number of Lanes", 
        min_value=1, max_value=4, value=2, step=1
    )
    
    curvature = st.sidebar.slider(
        "Road Curvature", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.01
    )
    
    speed_limit = st.sidebar.slider(
        "Speed Limit",
        min_value=25, max_value=70, value=45, step=5
    )
    
    # --- 5. Prediction Logic ---
    if st.sidebar.button("Predict Risk", type="primary"):
        
        # Create a single-row DataFrame from the inputs
        input_data = {
            'num_lanes': [num_lanes],
            'curvature': [curvature],
            'speed_limit': [speed_limit],
            'road_type': [road_type],
            'lighting': [lighting],
            'weather': [weather],
            'road_signs_present': [road_signs_present], # FIX: Pass the boolean value directly
            'public_road': [public_road],       # FIX: Pass the boolean value directly
            'time_of_day': [time_of_day]
        }
        input_df = pd.DataFrame(input_data, index=[0])

        try:
            # --- Step 1: Apply Label Encoding ---
            # Your new script encodes all 'object' types.
            input_df_processed = input_df.copy()
            encoded_cols = []
            
            for col, le in label_encoders.items():
                if col in input_df_processed.columns:
                    # Get the string value from the DataFrame
                    val = input_df_processed[col].astype(str).values[0]
                    
                    # Check if the value is known to the encoder
                    if val in le.classes_:
                        input_df_processed[col] = le.transform([val])[0]
                        encoded_cols.append(col)
                    else:
                        # Handle unseen label (e.g., assign -1 or 0)
                        st.warning(f"Unseen label '{val}' for feature '{col}'. Using 0.")
                        input_df_processed[col] = 0 # Or a default index
            
            # --- FIX: Convert encoded columns to numeric type ---
            # This forces the column dtype from 'object' to 'int'
            for col in encoded_cols:
                input_df_processed[col] = pd.to_numeric(input_df_processed[col])

            # --- Step 2: Apply Feature Engineering ---
            input_df_fe = engineer_features_app(input_df_processed)

            # --- Step 3: Ensure Column Order (if feature_names is reliable) ---
            # The model expects columns in the *exact* order it was trained on.
            # Your train.py saves feature_names *before* engineering,
            # so we must manually add the new feature names.
            if feature_names:
                final_feature_names = feature_names.copy()
                if 'curve_speed' in input_df_fe.columns:
                    final_feature_names.append('curve_speed')
                for col in ['num_lanes', 'curvature', 'speed_limit']:
                    if f'{col}_squared' in input_df_fe.columns:
                        final_feature_names.append(f'{col}_squared')
                
                # Reorder DataFrame
                final_input = input_df_fe.reindex(columns=final_feature_names, fill_value=0)
            else:
                # Fallback if feature_names was not saved
                final_input = input_df_fe

            # --- Step 4: Predict ---
            prediction = model.predict(final_input)
            predicted_risk = np.clip(prediction[0], 0, 1)

            # --- 6. Display the Result ---
            st.header("Predicted Accident Risk")
            
            if hasattr(st, 'gauge'):
                st.gauge(
                    value=predicted_risk,
                    min_value=0.0,
                    max_value=1.0,
                    label="Risk Score",
                    gauge_color_thresholds=[0.33, 0.66, 1.0],
                    gauge_color_palette=['#2ca02c', '#ff7f0e', '#d62728'] # Green, Orange, Red
                )
            else:
                st.metric(label="Risk Score", value=f"{predicted_risk:.1%}")
                st.progress(predicted_risk)
            
            # Add a dynamic message
            if predicted_risk < 0.33:
                st.success("Risk Level: **Low**")
            elif predicted_risk < 0.66:
                st.warning("Risk Level: **Moderate**")
            else:
                st.error("Risk Level: **High**")
                
            with st.expander("Show Input Parameters"):
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure the model file is compatible with this script.")
else:
    st.warning("Model assets are not loaded. Please check the 'road_safety_model.joblib' file.")
