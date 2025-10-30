import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
import matplotlib.pyplot as plt
import seaborn as sns
import os
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()

class PassengerFeature(BaseModel):
    Pclass: int = Field(description="Passenger class: 1, 2, or 3")
    Sex: str = Field(description="male or female")
    Age: int = Field(description="Age of the passenger")
    SibSp: int = Field(description="Number of siblings/spouses aboard")
    Parch: int = Field(description="Number of parents/children aboard")
    Fare: float = Field(description="Amount of money paid for ticket")
    Occupation: str = Field(description="Occupation of the passenger")
    # FavoriteColor: str = Field(description="Favourite color of the passenger")
    # Hobby: str = Field(description="Hobby of the passenger")


class SKLearnLogisticRegressionClassifier:
    
    def __init__(self):
        self.csv_path = os.path.join(os.path.dirname(__file__), 'stage-3-data.csv')
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None

        self.train_model(model_type='logistic_regression')
    
    def extract_features(self, input: str) -> PassengerFeature:

        system_prompt = f"""You are an expert at extracting structured features from unstructured text descriptions of passengers.

        

        """

        llm = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model="gpt-4o",
            api_version="2024-12-01-preview",
            temperature=0.0
        ).with_structured_output(PassengerFeature)



        response = llm.invoke([
            HumanMessage(content=f"""Extract following features from this input: {input}

            features to extract:
                        
            -Sex: male or female
            -Age: age of the passenger, e.g. 29
            -SibSp: number of siblings/spouses aboard, e.g. 0, 1, 2, etc.
            -Parch: number of parents/children aboard, e.g. 0, 1, 2, etc.
            -Fare: amount of money paid for ticket, e.g. 7.875
            -Occupation: occupation of the passenger, e.g. Butcher, Blacksmith, teacher, Assistant Mechanic, photographer, etc
            -FavoriteColor: favourite color of the passenger, e.g. Blue, Red, Green, Yellow, etc.
            -Hobby: hobby of the passenger, e.g: Reading, Embroidery, Piano, etc.
                         
            if not sure of the feature or missing feature, make up a value based on the feature description.
            """)
        ])

        return {
            "Pclass": response.Pclass,
            "Sex": response.Sex,
            "Age": response.Age,
            "SibSp": response.SibSp,
            "Parch": response.Parch,
            "Fare": response.Fare,
            "Occupation": response.Occupation
            # "FavoriteColor": response.FavoriteColor,
            # "Hobby": response.Hobby
        }   
        
    def load_and_prepare_data(self):
        """
        Load CSV and prepare data for training
        """
        # Load data
        df = pd.read_csv(self.csv_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Target variable is 'Selected' (0 or 1)
        print(f"\nTarget distribution (Selected):")
        print(df['Selected'].value_counts())
        print(f"Class balance: {df['Selected'].value_counts(normalize=True)}")
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess features and target
        """
        # Drop columns not useful for prediction
        columns_to_drop = ['PassengerId', 'Name', 'Selected', 'FavoriteColor', 'Hobby']
        
        X = df.drop(columns_to_drop, axis=1, errors='ignore')
        y = df['Selected']
        
        # Handle missing values in Age (fill with median)
        if X['Age'].isnull().any():
            age_median = X['Age'].median()
            X['Age'].fillna(age_median, inplace=True)
            print(f"\nFilled {X['Age'].isnull().sum()} missing Age values with median: {age_median}")
        
        # Encode categorical variables
        categorical_cols = ['Sex', 'Occupation'] #, 'FavoriteColor', 'Hobby']
        
        for col in categorical_cols:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded {col}: {len(le.classes_)} unique values")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"\nFeatures after preprocessing: {self.feature_names}")
        print(f"Feature types:\n{X.dtypes}")
        
        return X, y
    
    def train_model(self, model_type: str = 'random_forest', use_grid_search: bool = False):
        """
        Train the classification model
        """
        # Load and prepare data
        df = self.load_and_prepare_data()
        X, y = self.preprocess_data(df)
        
        # Split data (80/20 train/test split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
        print(f"Train class distribution:\n{y_train.value_counts()}")
        print(f"Test class distribution:\n{y_test.value_counts()}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose model
        if model_type == 'random_forest':
            if use_grid_search:
                print("\nPerforming GridSearchCV for hyperparameter tuning...")
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
                base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
                self.model = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1)
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        elif model_type == 'svc':
            self.model = SVC(
                kernel='rbf',  # or 'linear', 'poly', 'sigmoid'
                C=1.0,
                probability=True,  # Required for predict_proba
                class_weight='balanced',
                random_state=42
            )

        # Train model
        print(f"\nTraining {model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        if use_grid_search:
            print(f"Best parameters: {self.model.best_params_}")
            print(f"Best CV score: {self.model.best_score_:.4f}")
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Print metrics
        self.print_metrics(y_test, y_pred, y_pred_proba)
        
        # Plot results
        #self.plot_results(y_test, y_pred, y_pred_proba, model_type)
        
        # Feature importance (for Random Forest)
        # if model_type == 'random_forest':
        #     self.plot_feature_importance(model_type)
        
        return X_test_scaled, y_test, y_pred
    
    def print_metrics(self, y_true, y_pred, y_pred_proba):
        """
        Print classification metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        
        print(f"\nAccuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
        print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
        print(f"ROC AUC:   {roc_auc_score(y_true, y_pred_proba):.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Not Selected', 'Selected']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        print(f"True Negatives:  {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives:  {cm[1,1]}")
    
    def plot_results(self, y_true, y_pred, y_pred_proba, model_type):
        """
        Plot confusion matrix and ROC curve
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                    xticklabels=['Not Selected', 'Selected'],
                    yticklabels=['Not Selected', 'Selected'])
        axes[0].set_title(f'Confusion Matrix - {model_type}')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title(f'ROC Curve - {model_type}')
        axes[1].legend(loc="lower right")
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        filename = f'classification_results_{model_type}.png'
        plt.savefig(filename, dpi=150)
        print(f"\n✓ Results saved to {filename}")
        plt.close()
    
    def plot_feature_importance(self, model_type):
        """
        Plot feature importance for Random Forest
        """
        model = self.model.best_estimator_ if hasattr(self.model, 'best_estimator_') else self.model
        
        if isinstance(model, RandomForestClassifier):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            plt.title("Feature Importances - Random Forest")
            colors = plt.cm.viridis(importances[indices] / importances[indices].max())
            plt.bar(range(len(importances)), importances[indices], color=colors)
            plt.xticks(range(len(importances)), 
                      [self.feature_names[i] for i in indices], 
                      rotation=45, ha='right')
            plt.ylabel('Importance Score')
            plt.xlabel('Features')
            plt.tight_layout()
            
            filename = f'feature_importance_{model_type}.png'
            plt.savefig(filename, dpi=150)
            print(f"✓ Feature importance saved to {filename}")
            plt.close()
            
            # Print top features
            print("\nTop 5 Most Important Features:")
            for i in range(min(5, len(importances))):
                idx = indices[i]
                print(f"{i+1}. {self.feature_names[idx]}: {importances[idx]:.4f}")
    

    def predict_single(self, passenger_data: dict) -> dict:
        """
        Predict selection for a single passenger
        
        Args:
            passenger_data: dict with keys: Pclass, Sex, Age, SibSp, Parch, Fare, Occupation, FavoriteColor, Hobby
        """
        # Create dataframe from input
        df = pd.DataFrame([passenger_data])
        
        # Handle missing Age
        if 'Age' in df.columns and pd.isna(df['Age'].iloc[0]):
            df['Age'].fillna(self.scaler.mean_[self.feature_names.index('Age')], inplace=True)
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in df.columns:
                # df[col] = le.transform(df[col].astype(str))
                def safe_transform(value,col, label_encoder):
                    try:
                        return label_encoder.transform([str(value)])[0]
                    except ValueError:

                        if col.lower() == 'occupation':
                            # If unseen occupation, map to 'Other' if exists
                            if 'Other' in label_encoder.classes_:
                                return label_encoder.transform(['Other'])[0]
                        # If unseen label, return the most common class (first class)
                        # Or you can use -1 or len(classes) to indicate unknown
                        return 0  # Maps to first class in training data
                
                df[col] = df[col].apply(lambda x: safe_transform(x,col,le))
        
        # Ensure correct column order
        df = df[self.feature_names]
        
        # Scale
        X_scaled = self.scaler.transform(df)
        
        # Predict
        model = self.model.best_estimator_ if hasattr(self.model, 'best_estimator_') else self.model
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        return {
            "selected": bool(prediction),
            "probability_selected": float(probability[1]),
            "probability_not_selected": float(probability[0]),
            "confidence": float(max(probability))
        }
    
    def invoke(self, input: str) -> str:
        features = self.extract_features(input)
        prediction = self.predict_single(features)
        return prediction['selected'] , prediction, features
        #return f"Selected: {prediction['selected']}, Probability Selected: {prediction['probability_selected']:.2%}, Confidence: {prediction['confidence']:.2%}"


def main():
    """
    Main function to train and evaluate model
    """
    # Get the CSV path
    csv_path = os.path.join(os.path.dirname(__file__), 'stage-3-data.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
    
    # Initialize model
    classifier = SKLearnLogisticRegressionClassifier(csv_path)
    
    # Train Random Forest
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    classifier.train_model(model_type='random_forest', use_grid_search=False)
    
    # Train Logistic Regression
    print("\n\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*60)
    classifier_lr = SKLearnLogisticRegressionClassifier(csv_path)
    classifier_lr.train_model(model_type='logistic_regression')
    
    # Test prediction with Random Forest model
    print("\n" + "="*60)
    print("TESTING SINGLE PREDICTION (Random Forest)")
    print("="*60)
    
    test_passenger = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 35,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0,
        "Occupation": "Engineer",
        "FavoriteColor": "Blue",
        "Hobby": "Reading"
    }
    
    result = classifier.predict_single(test_passenger)
    print(f"\nTest passenger: {test_passenger}")
    print(f"\nPrediction: {'✓ Selected' if result['selected'] else '✗ Not Selected'}")
    print(f"Probability of Selection: {result['probability_selected']:.2%}")
    print(f"Confidence: {result['confidence']:.2%}")





if __name__ == "__main__":
    # main()

    test_passenger = {
        "Sex": "female",
        "Age": 35,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 80.0,
        "Occupation": "Engineer",
        "FavoriteColor": "Blue",
        "Hobby": "Reading"
    }
    
    classifier_lr = SKLearnLogisticRegressionClassifier()
    classifier_lr.train_model(model_type='svc')

    # features = classifier_lr.extract_features('Brimming with hope to settle on Azure Utopia, Zimmerman, Mr. Leo is a 29-year-old male third-class Butcher who paid a fare of 7.875, enjoys Embroidery, prefers Brown, and arrives alone with 0 siblings/spouse and 0 parents/children.')
    # result = classifier_lr.predict_single(test_passenger)

    input_1_reject = "Excited to start a new life on Azure Utopia, Oreskovic, Miss. Marija is a 20-year-old female third-class Baker who paid a fare of 8.6625, enjoys BirdWatching, loves Green, and arrives alone with 0 siblings/spouse and 0 parents/children."
   
    input_3_accepted = "Radiant with hope to reach Azure Utopia, Johannesen-Bratthammer, Mr. Bernt is a male third-class Miner of unspecified age who paid a fare of 8.1125, enjoys Gardening, favors Brown, and journeys solo with 0 siblings/spouse and 0 parents/children."
    result, predictions, features = classifier_lr.invoke(input_3_accepted)
    print(result)
    print(features)