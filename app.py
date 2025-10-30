import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Load the Saved Model Pipeline ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads the saved model pipeline from disk."""
    try:
        pipeline = joblib.load('accident_risk_model.joblib')
        return pipeline
    except FileNotFoundError:
        return None
    except Exception as e:
        # This will still catch other errors, like the 'libomp' one if it persists
        st.error(f"Error loading model: {e}")
        return None

model_pipeline = load_model()

# --- 2. Set up the Streamlit Page ---
st.set_page_config(
    page_title="Road Accident Risk Predictor",
    page_icon="",
    layout="wide"
)
st.title("Road Accident Risk Predictor")

# Check if model was loaded successfully
if model_pipeline is None:
    st.error("Model file 'accident_risk_model.joblib' not found or failed to load. "
             "Please run the `train.py` script first and ensure all dependencies (like 'libomp' on macOS) are installed.")
else:
    # --- 3. Create Sidebar Inputs (Based on your data summary) ---
    st.sidebar.header("Select Road Conditions")

    # --- Categorical Features ---
    road_type = st.sidebar.selectbox(
        "Road Type", 
        ['highway', 'rural', 'Other'] # Assuming 'Other' is a valid category name
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
        index=0 # Default to True
    )
    
    public_road = st.sidebar.radio(
        "Public Road?",
        [True, False],
        index=0 # Default to True
    )

    # --- Numerical Features (with min/max from your summary) ---
    num_lanes = st.sidebar.slider(
        "Number of Lanes", 
        min_value=1, max_value=4, value=2, step=1 # Range 1-4 from summary
    )
    
    curvature = st.sidebar.slider(
        "Road Curvature", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.01 # Range 0-1 from summary
    )
    
    speed_limit = st.sidebar.slider(
        "Speed Limit",
        min_value=25, max_value=70, value=45, step=5 # Range 25-70 from summary
    )

    # --- 4. Prediction Logic ---
    if st.sidebar.button("Predict Risk", type="primary"):
        
        # Create a single-row DataFrame from the inputs
        # The column names MUST exactly match those used in training
        input_data = {
            'num_lanes': [num_lanes],
            'curvature': [curvature],
            'speed_limit': [speed_limit],
            'road_type': [road_type],
            'lighting': [lighting],
            'weather': [weather],
            'road_signs_present': [road_signs_present],
            'public_road': [public_road],
            'time_of_day': [time_of_day]
        }
        
        input_df = pd.DataFrame(input_data)

        # Use the pipeline to predict (it handles all preprocessing)
        try:
            prediction = model_pipeline.predict(input_df)
            
            # Clip the prediction between 0 and 1
            predicted_risk = np.clip(prediction[0], 0, 1)

            # --- 5. Display the Result ---
            st.header("Predicted Accident Risk")

            # --- CODE CHANGED HERE ---
            # Use st.metric and st.progress as an alternative to st.gauge
            
            # Display the score as a percentage
            st.metric(label="Risk Score", value=f"{predicted_risk:.1%}")
            
            # Display a progress bar for a visual representation
            st.progress(predicted_risk)
            
            # --- END OF CHANGE ---
            
            # Add a dynamic message
            if predicted_risk < 0.33:
                st.success("Risk Level: **Low**")
            elif predicted_risk < 0.66:
                st.warning("Risk Level: **Moderate**")
            else:
                st.error("Risk Level: **High**")
                
            st.subheader("Input Parameters")
            st.dataframe(input_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Please ensure your inputs match the model's training data.")

    st.sidebar.info("Adjust the parameters and click 'Predict Risk' to see the updated accident risk score.")