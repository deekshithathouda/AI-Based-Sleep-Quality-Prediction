# AI-Based Sleep Quality Prediction

## Overview
The AI-Based Sleep Quality Prediction project aims to predict a person’s sleep quality (Good, Fair, or Poor) based on their health and lifestyle factors using Machine Learning models.  
This system uses algorithms such as XGBoost and Random Forest to analyze key factors like sleep duration, stress level, caffeine intake, physical activity, and more.  
It provides users with accurate predictions and personalized recommendations to improve sleep quality, along with an interactive chatbot interface built using Streamlit.

## Features
- Predicts sleep quality based on daily health and lifestyle data  
- Compares multiple ML models (XGBoost and Random Forest)  
- Interactive chatbot to provide sleep tips and recommendations  
- User-friendly web interface using Streamlit  
- Model trained and saved using Joblib  
- Displays evaluation metrics: accuracy, classification report, and confusion matrix  


## Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python |
| ML Libraries | XGBoost, Scikit-learn, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App Framework | Streamlit |
| Model Saving | Joblib |
| IDE / Platforms | Google Colab, VS Code, CMD |

## Dataset
The dataset (`ai_sleep_quality_dataset.csv`) contains various health and lifestyle parameters that influence sleep quality, including:

- Age  
- Gender  
- Sleep Duration (hrs)  
- Physical Activity (mins/day)  
- Stress Level (1–10)  
- Caffeine Intake (cups/day)  
- Alcohol Intake (units/day)  
- Smoking (Yes/No)  
- Heart Rate (bpm)  
- Screen Time Before Bed (hrs)  
- BMI  
- Wake-up Consistency (Regular/Irregular)  
- Sleep Environment Score (1–10)  
- Daily Water Intake (litres)  
- Sleep Disorder History (Yes/No)  
- **Target Variable:** Sleep Quality (`Poor`, `Fair`, `Good`)

## Model Training
The models were trained using the preprocessed dataset:
- Data preprocessing with StandardScaler  
- Label encoding for categorical variables  
- Model training with XGBoost and Random Forest  
- Evaluation metrics: Accuracy, Confusion Matrix, Classification Report  

**Final Results:**

| Model | Accuracy |
|--------|-----------|
| XGBoost | 98.25% |
| Random Forest | 97.50% |

XGBoost was selected as the final model due to its higher accuracy and robust performance.

## Project Workflow
1. Data Collection and Preprocessing  
2. Model Training (XGBoost & Random Forest)  
3. Model Evaluation and Comparison  
4. Saving Best Model and Scaler  
5. Deployment using Streamlit  
6. Chatbot Integration for user interaction and sleep tips  


## Chatbot Feature
The integrated chatbot provides:
- Quick responses to user queries about sleep habits  
- Suggestions for stress reduction, caffeine control, and better sleep routines  
- Natural conversational experience similar to chat applications  

