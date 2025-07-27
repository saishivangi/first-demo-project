from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
import joblib
import os
from datetime import datetime, timedelta
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'upi_fraud_detection_secret_key_2024'

class TransactionMode(Enum):
    UPI_PIN = "UPI_PIN"
    BIOMETRIC = "BIOMETRIC"
    QR_CODE = "QR_CODE"
    BANK_TRANSFER = "BANK_TRANSFER"
    MOBILE_BANKING = "MOBILE_BANKING"

@dataclass
class TransactionData:
    """Advanced dataclass for transaction information"""
    sender_upi: str
    receiver_upi: str
    amount: float
    transaction_mode: str
    transaction_time: str
    day_of_week: str
    transaction_frequency: int = 0
    account_age_days: int = 0
    is_weekend: bool = False
    is_night_time: bool = False
    amount_category: str = "normal"
    
    def __post_init__(self):
        self.validate_upi_format()
        self.calculate_derived_features()
    
    def validate_upi_format(self) -> None:
        """Validate UPI ID format"""
        upi_pattern = r'^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+$'
        if not re.match(upi_pattern, self.sender_upi):
            raise ValueError(f"Invalid sender UPI format: {self.sender_upi}")
        if not re.match(upi_pattern, self.receiver_upi):
            raise ValueError(f"Invalid receiver UPI format: {self.receiver_upi}")
    
    def calculate_derived_features(self) -> None:
        """Calculate derived features for enhanced fraud detection"""
        # Parse transaction time
        try:
            time_obj = datetime.strptime(self.transaction_time, "%H:%M")
            hour = time_obj.hour
            self.is_night_time = hour < 6 or hour > 22
        except ValueError:
            self.is_night_time = False
        
        # Determine weekend
        weekend_days = ['Saturday', 'Sunday']
        self.is_weekend = self.day_of_week in weekend_days
        
        # Categorize amount
        if self.amount < 100:
            self.amount_category = "micro"
        elif self.amount < 1000:
            self.amount_category = "small"
        elif self.amount < 10000:
            self.amount_category = "medium"
        elif self.amount < 100000:
            self.amount_category = "large"
        else:
            self.amount_category = "very_large"

class FraudDetectionModel:
    """Advanced ML model class with ensemble methods"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
    
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate comprehensive synthetic UPI transaction data"""
        np.random.seed(42)
        
        # UPI providers
        upi_providers = ['@paytm', '@googlepay', '@phonepe', '@ybl', '@okaxis', '@oksbi', '@okhdfcbank']
        
        # Generate base data
        data = []
        for i in range(n_samples):
            # Basic transaction info
            sender_name = f"user{np.random.randint(1000, 9999)}"
            receiver_name = f"merchant{np.random.randint(100, 999)}" if np.random.random() > 0.7 else f"user{np.random.randint(1000, 9999)}"
            
            sender_upi = f"{sender_name}{np.random.choice(upi_providers)}"
            receiver_upi = f"{receiver_name}{np.random.choice(upi_providers)}"
            
            # Transaction details
            amount = np.random.lognormal(mean=5, sigma=2)  # Log-normal distribution for realistic amounts
            amount = max(1, min(amount, 100000))  # Cap between 1 and 100,000
            
            transaction_mode = np.random.choice(list(TransactionMode)).value
            
            # Time features
            hour = np.random.randint(0, 24)
            minute = np.random.randint(0, 60)
            transaction_time = f"{hour:02d}:{minute:02d}"
            
            day_of_week = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            
            # Advanced features
            transaction_frequency = np.random.poisson(5)  # Average 5 transactions per day
            account_age_days = np.random.randint(1, 3650)  # Up to 10 years
            
            # Fraud patterns
            is_fraud = 0
            
            # High-risk patterns for fraud
            if (amount > 50000 and hour > 22) or \
               (amount > 75000 and transaction_frequency > 20) or \
               (account_age_days < 30 and amount > 25000) or \
               (sender_upi == receiver_upi) or \
               (transaction_frequency > 50) or \
               (amount > 90000 and day_of_week in ['Saturday', 'Sunday']):
                is_fraud = 1 if np.random.random() > 0.3 else 0
            
            # Add some random fraud cases
            if np.random.random() < 0.05:  # 5% base fraud rate
                is_fraud = 1
            
            data.append({
                'sender_upi': sender_upi,
                'receiver_upi': receiver_upi,
                'amount': amount,
                'transaction_mode': transaction_mode,
                'transaction_time': transaction_time,
                'day_of_week': day_of_week,
                'transaction_frequency': transaction_frequency,
                'account_age_days': account_age_days,
                'is_fraud': is_fraud
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced preprocessing with feature engineering"""
        df = df.copy()
        
        # Create derived features
        df['hour'] = df['transaction_time'].apply(lambda x: int(x.split(':')[0]))
        df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
        df['is_night_time'] = ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)
        
        # Amount categories
        df['amount_log'] = np.log1p(df['amount'])
        df['amount_category'] = pd.cut(df['amount'], 
                                     bins=[0, 100, 1000, 10000, 50000, np.inf], 
                                     labels=['micro', 'small', 'medium', 'large', 'very_large'])
        
        # UPI provider features
        df['sender_provider'] = df['sender_upi'].str.extract(r'@(.+)$')
        df['receiver_provider'] = df['receiver_upi'].str.extract(r'@(.+)$')
        df['same_provider'] = (df['sender_provider'] == df['receiver_provider']).astype(int)
        df['same_upi'] = (df['sender_upi'] == df['receiver_upi']).astype(int)
        
        # Risk score based on transaction frequency and account age
        df['risk_score'] = (df['transaction_frequency'] / df['account_age_days'].clip(lower=1)) * 1000
        
        return df
    
    def train_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train ensemble models with comprehensive evaluation"""
        logger.info("Starting model training...")
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Prepare features
        categorical_columns = ['transaction_mode', 'day_of_week', 'amount_category', 'sender_provider', 'receiver_provider']
        numerical_columns = ['amount', 'amount_log', 'transaction_frequency', 'account_age_days', 'hour', 'risk_score']
        binary_columns = ['is_weekend', 'is_night_time', 'same_provider', 'same_upi']
        
        # Encode categorical variables
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Select features
        feature_columns = [col + '_encoded' for col in categorical_columns] + numerical_columns + binary_columns
        self.feature_columns = feature_columns
        
        X = df_processed[feature_columns]
        y = df_processed['is_fraud']
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train models
        results = {}
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            logger.info(f"{name} accuracy: {accuracy:.4f}")
        
        self.is_trained = True
        logger.info("Model training completed!")
        return results
    
    def predict_fraud(self, transaction_data: Dict) -> Tuple[int, float, Dict]:
        """Predict fraud with confidence scores and explanations"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        # Create DataFrame
        df = pd.DataFrame([transaction_data])
        df_processed = self.preprocess_data(df)
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df_processed.columns:
                try:
                    df_processed[col + '_encoded'] = encoder.transform(df_processed[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    df_processed[col + '_encoded'] = 0
        
        # Prepare features
        X = df_processed[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0]
            predictions[name] = pred
            probabilities[name] = prob[1]  # Probability of fraud
        
        # Ensemble prediction (majority vote with weighted average)
        fraud_votes = sum(predictions.values())
        avg_probability = np.mean(list(probabilities.values()))
        
        # Final prediction
        final_prediction = 1 if fraud_votes >= len(self.models) / 2 else 0
        
        # Generate explanation
        explanation = self._generate_explanation(transaction_data, df_processed.iloc[0], avg_probability)
        
        return final_prediction, avg_probability, explanation
    
    def _generate_explanation(self, original_data: Dict, processed_data: pd.Series, probability: float) -> Dict:
        """Generate explanation for the prediction"""
        risk_factors = []
        
        # Check various risk factors
        if original_data['amount'] > 50000:
            risk_factors.append("High transaction amount")
        
        if processed_data['is_night_time']:
            risk_factors.append("Late night transaction")
        
        if processed_data['same_upi']:
            risk_factors.append("Same sender and receiver UPI")
        
        if original_data['transaction_frequency'] > 20:
            risk_factors.append("High transaction frequency")
        
        if original_data['account_age_days'] < 30:
            risk_factors.append("New account")
        
        if processed_data['risk_score'] > 100:
            risk_factors.append("High risk score")
        
        return {
            'probability': probability,
            'risk_factors': risk_factors,
            'confidence': 'High' if abs(probability - 0.5) > 0.3 else 'Medium'
        }

# Initialize the model
fraud_model = FraudDetectionModel()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Fraud prediction page"""
    if request.method == 'POST':
        try:
            # Extract form data
            transaction_data = {
                'sender_upi': request.form['sender_upi'],
                'receiver_upi': request.form['receiver_upi'],
                'amount': float(request.form['amount']),
                'transaction_mode': request.form['transaction_mode'],
                'transaction_time': request.form['transaction_time'],
                'day_of_week': request.form['day_of_week'],
                'transaction_frequency': int(request.form['transaction_frequency']),
                'account_age_days': int(request.form['account_age_days'])
            }
            
            # Validate data using dataclass
            trans_obj = TransactionData(**transaction_data)
            
            # Make prediction
            prediction, probability, explanation = fraud_model.predict_fraud(transaction_data)
            
            result = {
                'prediction': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
                'probability': f"{probability:.2%}",
                'explanation': explanation,
                'transaction_data': transaction_data
            }
            
            return render_template('predict.html', result=result)
            
        except Exception as e:
            flash(f"Error: {str(e)}", 'error')
            return render_template('predict.html')
    
    return render_template('predict.html')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Model training page"""
    if request.method == 'POST':
        try:
            # Generate training data
            sample_size = int(request.form.get('sample_size', 10000))
            
            logger.info(f"Generating {sample_size} samples for training...")
            training_data = fraud_model.generate_synthetic_data(sample_size)
            
            # Train model
            results = fraud_model.train_model(training_data)
            
            # Save model
            joblib.dump(fraud_model, 'fraud_detection_model.pkl')
            
            flash(f"Model trained successfully! Results: {results}", 'success')
            return render_template('train.html', results=results, trained=True)
            
        except Exception as e:
            flash(f"Training error: {str(e)}", 'error')
            return render_template('train.html')
    
    return render_template('train.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        prediction, probability, explanation = fraud_model.predict_fraud(data)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'explanation': explanation
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    return render_template('analytics.html')

if __name__ == '__main__':
    # Load existing model if available
    if os.path.exists('fraud_detection_model.pkl'):
        try:
            fraud_model = joblib.load('fraud_detection_model.pkl')
            logger.info("Existing model loaded successfully!")
        except Exception as e:
            logger.warning(f"Could not load existing model: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)