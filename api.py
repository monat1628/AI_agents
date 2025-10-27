from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Initialize FastAPI
app = FastAPI(
    title="AutoAnalyst API",
    description="AI-Powered Healthcare Prediction API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "outputs/best_model.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Pydantic models for request/response
class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=120, description="Patient age in years")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    blood_pressure_systolic: int = Field(..., ge=70, le=250, description="Systolic blood pressure")
    blood_pressure_diastolic: int = Field(..., ge=40, le=150, description="Diastolic blood pressure")
    cholesterol: int = Field(..., ge=100, le=400, description="Cholesterol level")
    glucose: int = Field(..., ge=50, le=300, description="Glucose level")
    exercise_hours_per_week: int = Field(..., ge=0, le=20, description="Weekly exercise hours")
    gender: str = Field(..., description="Gender (Male/Female)")
    smoker: str = Field(..., description="Smoking status (Yes/No)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 55,
                "bmi": 28.5,
                "blood_pressure_systolic": 140,
                "blood_pressure_diastolic": 90,
                "cholesterol": 220,
                "glucose": 120,
                "exercise_hours_per_week": 3,
                "gender": "Male",
                "smoker": "Yes"
            }
        }

class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    risk_level: str
    confidence: float
    timestamp: str
    model_version: str

class HealthStatus(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    timestamp: str

# In-memory prediction log (for monitoring)
prediction_log = []

# Root endpoint
@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to AutoAnalyst API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/health", response_model=HealthStatus)
def health_check():
    """Check API and model health"""
    return HealthStatus(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
        timestamp=datetime.now().isoformat()
    )

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    """
    Predict heart disease risk for a single patient
    
    Returns prediction, probability, and risk assessment
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare features (must match training data format)
        features = pd.DataFrame({
            'age': [patient.age],
            'bmi': [patient.bmi],
            'blood_pressure_systolic': [patient.blood_pressure_systolic],
            'blood_pressure_diastolic': [patient.blood_pressure_diastolic],
            'cholesterol': [patient.cholesterol],
            'glucose': [patient.glucose],
            'exercise_hours_per_week': [patient.exercise_hours_per_week],
            'gender_Male': [1 if patient.gender == 'Male' else 0],
            'smoker_Yes': [1 if patient.smoker == 'Yes' else 0]
        })
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Get probability for positive class (heart disease)
        prob_positive = probability[1]
        
        # Determine risk level
        if prob_positive < 0.3:
            risk_level = "Low"
        elif prob_positive < 0.6:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(prob_positive - 0.5) * 2
        
        # Log prediction
        prediction_log.append({
            "timestamp": datetime.now().isoformat(),
            "prediction": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "probability": float(prob_positive),
            "risk_level": risk_level
        })
        
        return PredictionResponse(
            prediction="Heart Disease" if prediction == 1 else "No Heart Disease",
            probability=float(prob_positive),
            risk_level=risk_level,
            confidence=float(confidence),
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    """
    Predict heart disease risk for multiple patients from CSV file
    
    Upload a CSV file with patient data
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Validate required columns
        required_cols = ['age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                        'cholesterol', 'glucose', 'exercise_hours_per_week', 'gender', 'smoker']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_cols}")
        
        # Prepare features
        df['gender_Male'] = (df['gender'] == 'Male').astype(int)
        df['smoker_Yes'] = (df['smoker'] == 'Yes').astype(int)
        
        feature_cols = ['age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                       'cholesterol', 'glucose', 'exercise_hours_per_week', 'gender_Male', 'smoker_Yes']
        
        X = df[feature_cols]
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Add results to dataframe
        df['prediction'] = ['Heart Disease' if p == 1 else 'No Heart Disease' for p in predictions]
        df['probability'] = probabilities
        df['risk_level'] = ['Low' if p < 0.3 else 'Moderate' if p < 0.6 else 'High' for p in probabilities]
        
        # Return results
        return {
            "total_predictions": len(df),
            "results": df[['prediction', 'probability', 'risk_level']].to_dict('records'),
            "summary": {
                "heart_disease_count": int((predictions == 1).sum()),
                "no_heart_disease_count": int((predictions == 0).sum()),
                "average_probability": float(probabilities.mean())
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Statistics endpoint
@app.get("/stats")
def get_stats():
    """Get API usage statistics"""
    if len(prediction_log) == 0:
        return {
            "total_predictions": 0,
            "message": "No predictions made yet"
        }
    
    df = pd.DataFrame(prediction_log)
    
    return {
        "total_predictions": len(prediction_log),
        "predictions_last_hour": len([p for p in prediction_log 
                                      if (datetime.now() - datetime.fromisoformat(p['timestamp'])).seconds < 3600]),
        "average_probability": float(df['probability'].mean()),
        "risk_distribution": df['risk_level'].value_counts().to_dict(),
        "latest_predictions": prediction_log[-5:]  # Last 5 predictions
    }

# Model info endpoint
@app.get("/model-info")
def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "model_path": MODEL_PATH,
        "features": ['age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
                    'cholesterol', 'glucose', 'exercise_hours_per_week', 'gender_Male', 'smoker_Yes'],
        "target": "heart_disease",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
