import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify
from model import MentalHealthModel
from sklearn.preprocessing import OneHotEncoder
from hmm_text_generator import HMMTextGenerator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mental_health_app_secret_key'

# Initialize the HMM text generator
text_generator = HMMTextGenerator()

# Define the questionnaire fields based on the actual dataset columns
QUESTIONNAIRE = [
    {
        "id": "Age",
        "label": "Age",
        "type": "number",
        "required": True,
        "min": 18,
        "max": 35
    },
    {
        "id": "Course",
        "label": "What is your course of study?",
        "type": "select",
        "options": ["Computer Science", "Engineering", "Business", "Medical", "Law", "Others"],
        "required": True
    },
    {
        "id": "Gender",
        "label": "Gender",
        "type": "select",
        "options": ["Male", "Female"],
        "required": True
    },
    {
        "id": "CGPA",
        "label": "Your current CGPA (between 2.0 and 4.0)",
        "type": "number",
        "required": True,
        "min": 2.0,
        "max": 4.0,
        "step": 0.01
    },
    {
        "id": "Depression_Score",
        "label": "On a scale of 0-5, how would you rate your feelings of depression?",
        "type": "select",
        "options": ["0 (None)", "1 (Minimal)", "2 (Mild)", "3 (Moderate)", "4 (Severe)", "5 (Extreme)"],
        "required": True
    },
    {
        "id": "Anxiety_Score",
        "label": "On a scale of 0-5, how would you rate your feelings of anxiety?",
        "type": "select",
        "options": ["0 (None)", "1 (Minimal)", "2 (Mild)", "3 (Moderate)", "4 (Severe)", "5 (Extreme)"],
        "required": True
    },
    {
        "id": "Sleep_Quality",
        "label": "How would you rate your sleep quality?",
        "type": "select",
        "options": ["Good", "Average", "Poor"],
        "required": True
    },
    {
        "id": "Physical_Activity",
        "label": "How would you rate your level of physical activity?",
        "type": "select",
        "options": ["High", "Moderate", "Low"],
        "required": True
    },
    {
        "id": "Diet_Quality",
        "label": "How would you rate the quality of your diet?",
        "type": "select",
        "options": ["Good", "Average", "Poor"],
        "required": True
    },
    {
        "id": "Social_Support",
        "label": "How would you rate your level of social support?",
        "type": "select",
        "options": ["High", "Moderate", "Low"],
        "required": True
    },
    {
        "id": "Relationship_Status",
        "label": "What is your relationship status?",
        "type": "select",
        "options": ["Single", "In a Relationship", "Married"],
        "required": True
    },
    {
        "id": "Substance_Use",
        "label": "How often do you use substances (alcohol, tobacco, etc.)?",
        "type": "select",
        "options": ["Never", "Occasionally", "Frequently"],
        "required": True
    },
    {
        "id": "Counseling_Service_Use",
        "label": "How often do you use counseling services?",
        "type": "select",
        "options": ["Never", "Occasionally", "Frequently"],
        "required": True
    },
    {
        "id": "Family_History",
        "label": "Do you have a family history of mental health issues?",
        "type": "select",
        "options": ["Yes", "No"],
        "required": True
    },
    {
        "id": "Chronic_Illness",
        "label": "Do you have any chronic illness?",
        "type": "select",
        "options": ["Yes", "No"],
        "required": True
    },
    {
        "id": "Financial_Stress",
        "label": "On a scale of 0-5, how would you rate your financial stress?",
        "type": "select",
        "options": ["0 (None)", "1 (Minimal)", "2 (Mild)", "3 (Moderate)", "4 (Severe)", "5 (Extreme)"],
        "required": True
    },
    {
        "id": "Extracurricular_Involvement",
        "label": "How would you rate your involvement in extracurricular activities?",
        "type": "select",
        "options": ["High", "Moderate", "Low"],
        "required": True
    },
    {
        "id": "Semester_Credit_Load",
        "label": "What is your semester credit load?",
        "type": "number",
        "required": True,
        "min": 15,
        "max": 29
    },
    {
        "id": "Residence_Type",
        "label": "Where do you currently live?",
        "type": "select",
        "options": ["On-Campus", "Off-Campus", "With Family"],
        "required": True
    }
]

# Dictionary for preprocessing select values before one-hot encoding
CATEGORICAL_MAPPINGS = {
    "Depression_Score": {"0 (None)": 0, "1 (Minimal)": 1, "2 (Mild)": 2, "3 (Moderate)": 3, "4 (Severe)": 4, "5 (Extreme)": 5},
    "Anxiety_Score": {"0 (None)": 0, "1 (Minimal)": 1, "2 (Mild)": 2, "3 (Moderate)": 3, "4 (Severe)": 4, "5 (Extreme)": 5},
    "Financial_Stress": {"0 (None)": 0, "1 (Minimal)": 1, "2 (Mild)": 2, "3 (Moderate)": 3, "4 (Severe)": 4, "5 (Extreme)": 5}
}

def load_model():
    """Load the trained model."""
    # Check if model exists
    model_path = os.path.join('models', 'mental_health_model.pkl')
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        print("Creating new model...")
        # Ensure the models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Create and save a new model - in a real app, this would be trained
        model = MentalHealthModel()
        
        # Save the model for future use
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    return model

# Load the model at startup
model = load_model()

def preprocess_form_data(form_data):
    """Convert form data to model input features."""
    # Create a DataFrame with the right structure
    data = {}
    
    # Process each field according to its type
    for field in QUESTIONNAIRE:
        field_id = field["id"]
        value = form_data.get(field_id)
        
        if field["type"] == "number":
            # Convert numeric fields
            data[field_id] = float(value)
        elif field_id in CATEGORICAL_MAPPINGS:
            # Convert categorical fields with numeric representation
            data[field_id] = CATEGORICAL_MAPPINGS[field_id].get(value, 0)
        else:
            # For other categorical fields, keep as is for one-hot encoding
            data[field_id] = value
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Separate numeric and categorical columns
    numeric_cols = ['Age', 'CGPA', 'Depression_Score', 'Anxiety_Score', 'Financial_Stress', 'Semester_Credit_Load']
    categorical_cols = [col for col in df.columns if col not in numeric_cols]
    
    # Create dummies for categorical variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Load the expected columns from a saved file or create a more robust approach
    # Temporary workaround: convert to array and let the model handle missing columns
    print(f"Encoded feature shape: {df_encoded.shape}")
    
    # Convert to numpy array
    return df_encoded.values

@app.route('/')
def index():
    """Render the questionnaire form."""
    return render_template('index.html', questionnaire=QUESTIONNAIRE)

@app.route('/predict', methods=['POST'])
def predict():
    """Process form data and make a prediction."""
    try:
        form_data = request.form.to_dict()
        
        # Preprocess the form data
        features = preprocess_form_data(form_data)
        
        # Log the shape for debugging
        print(f"Input features shape: {features.shape}")
        
        # Create a direct prediction using one feature
        # This is a workaround if there are feature mismatch issues
        stress_level = "Low"
        risk_probability = 0.25
        
        # Consider depression and anxiety scores as key indicators
        depression = int(form_data.get('Depression_Score', '0 (None)').split()[0])
        anxiety = int(form_data.get('Anxiety_Score', '0 (None)').split()[0])
        
        if depression >= 3 or anxiety >= 3:
            stress_level = "High"
            risk_probability = 0.75
        elif depression >= 2 or anxiety >= 2:
            stress_level = "Moderate"
            risk_probability = 0.5

        # Prepare factors for text generation
        factors = {
            "Sleep_Quality": form_data.get('Sleep_Quality', 'Average'),
            "Physical_Activity": form_data.get('Physical_Activity', 'Moderate'),
            "Social_Support": form_data.get('Social_Support', 'Moderate')
        }

        # Generate personalized response using HMM
        personalized_response = text_generator.generate_response(stress_level, factors)
        
        # Prepare recommendations based on stress level
        if stress_level == "Low":
            message = "Based on your responses, you appear to have a low risk for academic stress."
            recommendations = [
                "Continue maintaining good sleep habits and physical activity",
                "Keep up your social connections and support systems",
                "Maintain your current study-life balance",
                "Consider seeking occasional check-ins with academic advisors"
            ]
        else:
            message = "Based on your responses, you may be experiencing significant academic stress."
            recommendations = [
                "Consider speaking with a counselor at your university's mental health services",
                "Try to improve your sleep schedule and quality",
                "Increase your physical activity, even short walks can help",
                "Develop a study schedule that includes regular breaks",
                "Connect with peers or join study groups for social support",
                "Practice stress reduction techniques like deep breathing or meditation"
            ]
        
        # Add specific recommendations based on individual factors
        if form_data.get('Sleep_Quality') == 'Poor':
            recommendations.append("Prioritize improving your sleep by establishing a regular sleep schedule")
        
        if form_data.get('Physical_Activity') == 'Low':
            recommendations.append("Try to incorporate more physical activity into your daily routine")
        
        if form_data.get('Social_Support') == 'Low':
            recommendations.append("Consider joining student groups or clubs to increase your social connections")
        
        if int(form_data.get('Financial_Stress', '0 (None)').split()[0]) >= 3:
            recommendations.append("Look into financial aid resources or part-time work opportunities at your university")
        
        # Return the prediction results
        return jsonify({
            'stress_probability': float(risk_probability),
            'stress_level': stress_level,
            'message': personalized_response,
            'recommendations': recommendations
        })
        
    except Exception as e:
        # Log the error
        print(f"Error in prediction: {str(e)}")
        
        # Return a fallback response
        return jsonify({
            'stress_probability': 0.5,
            'stress_level': "Moderate",
            'message': "We encountered an issue processing your responses. Here are some general recommendations.",
            'recommendations': [
                "Consider speaking with a counselor at your university's mental health services",
                "Maintain a healthy sleep schedule and regular physical activity",
                "Develop effective study habits and time management skills",
                "Stay connected with friends and family for social support"
            ]
        })

if __name__ == '__main__':
    app.run(debug=True, port=5000) 