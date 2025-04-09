# Student Mental Health Analysis Project

This project involves the analysis of student mental health data through a two-part approach: an experimental component that explores various machine learning techniques, and a web application that provides a practical interface for real-world use.

## Project Overview

The Student Mental Health Analysis Project is designed to:

1. **Analyze factors affecting student mental health** - Identify key predictors of stress levels in academic environments
2. **Compare different machine learning approaches** - Evaluate various preprocessing, balancing, and modeling techniques
3. **Provide a practical application** - Offer a user-friendly web interface for students to assess their mental health risk

## Dataset

The project uses the `students_mental_health_survey.csv` dataset, which contains various features related to student mental health:

- **Demographics**: Age, Gender, Course
- **Academic Variables**: CGPA, Semester Credit Load
- **Mental Health Indicators**: Depression Score, Anxiety Score, Stress Level
- **Lifestyle Factors**: Sleep Quality, Physical Activity, Diet Quality
- **Social Factors**: Social Support, Relationship Status
- **Health Behaviors**: Substance Use, Counseling Service Use
- **Living Situation**: Residence Type
- **Other Factors**: Family History, Chronic Illness, Financial Stress, Extracurricular Involvement

The primary target variable is `Stress_Level` for classification purposes.

## Project Components

### Part 1: Experimental Analysis and Model Development

The experimental component focuses on exploring and comparing different machine learning techniques for mental health prediction:

#### Key Files:
- `main.py` - Main script for running the complete analysis pipeline
- `model.py` - Contains the model implementation and evaluation methods
- `train_and_save_model.py` - Script to train and save the best model for web app use
- `requirements.txt` - Package dependencies

#### Experiment Features:
1. **Data Preprocessing**
   - Missing value imputation
   - Feature encoding
   - Exploratory data analysis

2. **Class Balancing Techniques Comparison**
   - Naive random undersampling
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Hybrid approach (SMOTE + Tomek Links)

3. **Dimensionality Reduction**
   - PCA (Principal Component Analysis)
   - ICA (Independent Component Analysis)
   - Feature importance-based selection

4. **Model Ensemble**
   - Random Forest
   - Gradient Boosting
   - Support Vector Machine (SVM)
   - Neural Network
   - XGBoost
   - Voting Classifier (combining all models)

5. **Performance Evaluation**
   - Cross-validation
   - Holdout testing
   - Confusion matrices
   - Classification reports
   - Bootstrap confidence intervals


### Part 2: Web Application

The web application provides a user-friendly interface for students to assess their mental health risk based on the trained model:

#### Key Files:
- `app.py` - Flask web application
- `templates/` - HTML templates for the web interface
- `static/css/styles.css` - CSS styling for the web app
- `models/` - Directory for saved model files

#### Web App Features:
1. **User-Friendly Questionnaire**
   - Comprehensive assessment based on the dataset variables
   - Dynamic form with appropriate input types for each question

2. **Real-time Analysis**
   - Client-side form validation
   - Server-side processing with the trained model
   - Results displayed without page refresh

3. **Personalized Recommendations**
   - Stress level assessment (Low/High)
   - Tailored recommendations based on individual responses
   - Specific suggestions for identified risk areas

4. **Resource Information**
   - Campus resources for mental health support
   - Crisis hotlines and external support options
   - Educational content about student mental health

## Running the Project

### Prerequisites
- Python 3.8+ installed
- Required Python packages (install via `pip install -r requirements.txt`)
- Available port for web application (default: 5000)

### Running the Experimental Analysis

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the main analysis pipeline**:
   ```bash
   python main.py
   ```
   This will:
   - Load and preprocess the dataset
   - Train models using different balancing techniques
   - Compare performance and select the best approach
   - Perform feature selection and hyperparameter tuning
   - Evaluate the final model
   - Save the trained model

3. **Generate a detailed report** (optional):
   ```bash
   python run_report.py
   ```
   This will create a comprehensive PDF report of the analysis results

### Running the Web Application

1. **Train and save the model** (if not already done):
   ```bash
   python train_and_save_model.py
   ```

2. **Launch the web application**:
   ```bash
   python app.py
   ```
   This will start the Flask server on http://127.0.0.1:5000

3. **Using the application**:
   - Open a web browser and navigate to http://127.0.0.1:5000
   - Fill out the mental health questionnaire
   - Submit the form to receive your stress level assessment and personalized recommendations

### All-in-One Script


## Project Structure

```
student_mental_health/
│
├── students_mental_health_survey.csv
│
├── main.py                     # Main analysis script
├── model.py                    # Model implementation
├── train_and_save_model.py     # Script to train and save model
├── prepare_output_directory.py # Output directory preparation
│
├── app.py                      # Flask web application
│
├── templates/                  # HTML templates
│   ├── base.html               # Base template with common elements
│   └── index.html              # Main questionnaire page
│
├── static/                     # Static web files
│   └── css/
│       └── styles.css          # Custom CSS for web app
│
├── models/                     # Saved model files
│   └── mental_health_model.pkl # Trained model
│
├── output/                     # Analysis outputs and visualizations
│
├── requirements.txt            # Package dependencies
└── PROJECT_DOCUMENTATION.md    # This documentation file
```

## Conclusions and Future Work

This project demonstrates the application of machine learning to student mental health analysis, providing both experimental insights and a practical tool for assessment.

### Key Findings:
- Class balancing significantly improves model performance on imbalanced mental health data
- Ensemble models generally outperform individual algorithms for this task
- Feature selection helps identify the most important predictors of student stress

### Future Directions:
- Incorporate more sophisticated analysis of temporal patterns in student stress
- Develop a mobile application version for increased accessibility
- Integrate with institutional support systems for direct referral capabilities
- Add user accounts and longitudinal tracking of mental health indicators
- Expand the model with additional contextual factors (e.g., academic deadlines, exam periods)

## Disclaimer

This tool is for educational purposes only and is not intended to diagnose or treat any medical condition. If you are experiencing mental health issues, please consult a healthcare professional. 
