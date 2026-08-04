import streamlit as st
import pickle
import numpy as np

# 1. Load ONLY the exported model using Pickle
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# 2. Set up the web interface title
st.title("🎓 Student Placement Predictor & Evaluator")
st.write("Input the student's metrics below to evaluate their placement probability.")

# 3. Interactive input fields (Modify names and limits to match your training dataset)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
iq = st.slider("IQ Score", 0, 200, 100)
# internships = st.number_input("Number of Internships Completed", min_value=0, max_value=5, value=1)
# dsa_score = st.slider("Data Structures & Algorithms Score (%)", 0, 100, 70)
# backlogs = st.radio("Active Backlogs?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# 4. Handle prediction logic
if st.button("Evaluate Profile"):
    # Organize input exactly how your model expects it (Raw numbers)
    features = np.array([[cgpa, iq]])  # Add other features if your model requires them
    
    # Run prediction and fetch probabilities directly on raw features
    prediction = model.predict(features)
    
    # Check if your model supports probability estimates (e.g., Logistic Regression, Random Forest)
    try:
        prediction_proba = model.predict_proba(features)[0][1] # Get probability of class 1 (Placed)
        proba_text = f" ({prediction_proba * 100:.2f}% confidence)"
    except AttributeError:
        proba_text = "" # Falls back to standard output if model doesn't support predict_proba

    # 5. Display evaluation results
    st.subheader("Evaluation Results")
    if prediction[0] == 1:
        st.success(f"🎉 **High Chance of Placement!**{proba_text}")
    else:
        st.error(f"⚠️ **Low Chance of Placement.**{proba_text} Focus on enhancing core skills.")
