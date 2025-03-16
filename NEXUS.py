import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import datetime
import os
import logging
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

class NEXUSCreditEngine:
    """
    NEXUS (Neural Evaluation for Xpanded User Scoring) credit assessment engine
    for offline microfinance institutions to evaluate creditworthiness across sectors.
    """
    
    def __init__(self, base_dir="./models", db_config=None):
        """Initialize the NEXUS engine with database connection"""
        self.base_dir = base_dir
        self.models = {}
        self.sector_configs = {}
        self.default_config = None
        self.logger = self._setup_logging()
        self.db_connection = None

        # Load environment variables if .env exists
        load_dotenv()
        
        # Initialize database connection if config provided
        if db_config or os.getenv('DB_HOST'):
            self.connect_to_database(db_config or {
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT'),
                'database': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD')
            })
        
        # Ensure model directory exists
        os.makedirs(base_dir, exist_ok=True)
        
        # Load configurations and models
        self._load_configurations()
        self._load_models()
        
        self.logger.info("NEXUS Credit Engine initialized successfully")
    
    def _setup_logging(self):
        """Set up logging for the engine."""
        logger = logging.getLogger("NEXUS")
        logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler("nexus.log")
        console_handler = logging.StreamHandler()
        
        # Create formatters and add to handlers
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_configurations(self):
        """Load sector-specific configurations."""
        # In a real implementation, these would be loaded from configuration files
        # For this example, we'll define them directly
        
        # Default configuration for all sectors
        self.default_config = {
            "min_score": 300,
            "max_score": 850,
            "approval_threshold": 620,
            "review_threshold": 580,
            "numerical_features": [
                "age", "income", "expense_ratio", "business_age", 
                "cash_flow_stability", "avg_monthly_profit"
            ],
            "categorical_features": [
                "education", "business_type", "location_type"
            ],
            "feature_weights": {
                "payment_history": 0.35,
                "business_performance": 0.25,
                "cash_flow": 0.20,
                "collateral": 0.10,
                "business_stability": 0.10
            }
        }
        
        # Sector-specific configurations
        self.sector_configs = {
            "agriculture": {
                "feature_weights": {
                    "payment_history": 0.30,
                    "business_performance": 0.20,
                    "cash_flow": 0.15,
                    "collateral": 0.15,
                    "business_stability": 0.10,
                    "seasonal_adjustment": 0.10
                },
                "additional_features": ["crop_type", "farm_size", "seasonal_income_pattern"],
                "seasonal_factors": True
            },
            "retail": {
                "feature_weights": {
                    "payment_history": 0.35,
                    "business_performance": 0.25,
                    "cash_flow": 0.20,
                    "collateral": 0.10,
                    "business_stability": 0.10
                },
                "additional_features": ["inventory_turnover", "customer_base_size", "location_quality"],
                "seasonal_factors": True
            },
            "service": {
                "feature_weights": {
                    "payment_history": 0.40,
                    "business_performance": 0.20,
                    "cash_flow": 0.20,
                    "collateral": 0.05,
                    "business_stability": 0.15
                },
                "additional_features": ["service_type", "client_retention", "skill_level"],
                "seasonal_factors": False
            },
            "manufacturing": {
                "feature_weights": {
                    "payment_history": 0.30,
                    "business_performance": 0.20,
                    "cash_flow": 0.15,
                    "collateral": 0.20,
                    "business_stability": 0.15
                },
                "additional_features": ["equipment_value", "production_capacity", "raw_material_access"],
                "seasonal_factors": False
            }
        }
    
    def _load_models(self):
        """Load pre-trained models for different sectors."""
        # In a production system, you would load saved models from files
        # For this example, we'll create new models when needed
        
        self.models = {}
        # Models will be created on-demand when assess_credit is called
    
    def _create_preprocessing_pipeline(self, config):
        """Create a preprocessing pipeline for data transformation."""
        numerical_features = config.get("numerical_features", self.default_config["numerical_features"])
        categorical_features = config.get("categorical_features", self.default_config["categorical_features"])
        
        # Preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def _create_fallback_model(self, sector):
        """Create a fallback model when no pre-trained model exists."""
        # This would be replaced with a more sophisticated approach in production
        config = self.sector_configs.get(sector, self.default_config)
        
        # For classification (approve/deny)
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        # For regression (score/amount recommendation)
        reg = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        
        preprocessor = self._create_preprocessing_pipeline(config)
        
        return {
            'classification': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', clf)]),
            'regression': Pipeline(steps=[('preprocessor', preprocessor), ('regressor', reg)]),
            'config': config
        }
    
    def _extract_features(self, client_data, financial_data, sector):
        """Extract relevant features from client and financial data."""
        # Get configuration for the sector
        config = self.sector_configs.get(sector, self.default_config)
        
        # Basic features that should be present for all sectors
        features = {
            # Client demographics
            'age': float(client_data.get('age', 0)),
            'education': client_data.get('education', 'unknown'),
            'dependents': float(client_data.get('dependents', 0)),
            
            # Business information
            'business_age': float(client_data.get('business_age', 0)),
            'business_type': client_data.get('business_type', 'unknown'),
            'location_type': client_data.get('location_type', 'unknown'),
            
            # Financial indicators
            'income': float(financial_data.get('average_monthly_income', 0)),
            'expense_ratio': float(financial_data.get('expense_to_income_ratio', 0)),
            'avg_monthly_profit': float(financial_data.get('average_monthly_profit', 0)),
            'cash_flow_stability': float(financial_data.get('cash_flow_stability_index', 0)),
            
            # Credit history
            'previous_loans': float(financial_data.get('previous_loans_count', 0)),
            'repayment_rate': float(financial_data.get('previous_repayment_rate', 1.0)),
            'days_overdue': float(financial_data.get('average_days_overdue', 0))
        }
        
        # Add sector-specific features if available
        for feature in config.get('additional_features', []):
            features[feature] = financial_data.get(feature, 0)
        
        return features
    
    def _calculate_component_scores(self, features, sector, financial_data):
        """Calculate component scores based on feature weights."""
        config = self.sector_configs.get(sector, self.default_config)
        weights = config.get('feature_weights', self.default_config['feature_weights'])
        
        component_scores = {}
        
        # Payment history component (35%)
        payment_history_score = 0
        if features['previous_loans'] > 0:
            # Scale repayment rate (0-1) to 0-100
            repayment_score = min(features['repayment_rate'] * 100, 100)
            # Penalize for days overdue
            overdue_penalty = min(features['days_overdue'] * 2, 50)
            payment_history_score = max(repayment_score - overdue_penalty, 0)
        else:
            # No credit history, use a default score
            payment_history_score = 50
        
        component_scores['payment_history'] = payment_history_score
        
        # Business performance component (25%)
        profit_margin = features['avg_monthly_profit'] / max(features['income'], 1) * 100
        business_performance_score = min(profit_margin * 5, 100)  # Scale profit margin
        component_scores['business_performance'] = business_performance_score
        
        # Cash flow component (20%)
        cash_flow_score = min(features['cash_flow_stability'] * 100, 100)
        component_scores['cash_flow'] = cash_flow_score
        
        # Business stability component (10%)
        stability_score = min(features['business_age'] * 10, 100)  # 10 years = max score
        component_scores['business_stability'] = stability_score
        
        # Collateral component - default to medium score if not directly assessed
        component_scores['collateral'] = financial_data.get('collateral_value_ratio', 50) * 100
        
        # Seasonal adjustment if applicable
        if config.get('seasonal_factors', False) and 'seasonal_income_pattern' in features:
            seasonal_score = 100 - (features.get('seasonal_income_pattern', 0.5) * 100)
            component_scores['seasonal_adjustment'] = seasonal_score
        
        return component_scores
    
    def _calculate_final_score(self, component_scores, sector):
        """Calculate the final credit score from component scores."""
        config = self.sector_configs.get(sector, self.default_config)
        weights = config.get('feature_weights', self.default_config['feature_weights'])
        
        final_score = 0
        for component, score in component_scores.items():
            if component in weights:
                final_score += score * weights[component]
        
        # Scale to standard credit score range (300-850)
        min_score = config.get('min_score', self.default_config['min_score'])
        max_score = config.get('max_score', self.default_config['max_score'])
        score_range = max_score - min_score
        
        normalized_score = min_score + (final_score / 100 * score_range)
        return round(normalized_score)
    
    def _determine_loan_eligibility(self, credit_score, sector, loan_amount, income):
        """Determine loan eligibility and terms based on credit score."""
        config = self.sector_configs.get(sector, self.default_config)
        approval_threshold = config.get('approval_threshold', self.default_config['approval_threshold'])
        review_threshold = config.get('review_threshold', self.default_config['review_threshold'])
        
        # Basic eligibility
        if credit_score >= approval_threshold:
            eligibility = "APPROVED"
        elif credit_score >= review_threshold:
            eligibility = "REVIEW"
        else:
            eligibility = "DENIED"
        
        # Loan amount recommendation
        # Simple approach: approve up to 6 months of income based on credit score
        max_score = config.get('max_score', self.default_config['max_score'])
        if credit_score >= approval_threshold:
            max_multiplier = 6 * (credit_score / max_score)
            recommended_amount = min(loan_amount, income * max_multiplier)
        elif credit_score >= review_threshold:
            max_multiplier = 3 * (credit_score / max_score)
            recommended_amount = min(loan_amount, income * max_multiplier)
        else:
            recommended_amount = 0
        
        # Interest rate determination (simple approach)
        if credit_score >= 750:
            interest_rate = 10.0  # Prime rate
        elif credit_score >= 700:
            interest_rate = 12.0
        elif credit_score >= 650:
            interest_rate = 15.0
        elif credit_score >= review_threshold:
            interest_rate = 18.0
        else:
            interest_rate = 22.0
        
        # Term length based on amount and credit score
        if recommended_amount > income * 3:
            term_months = 24
        else:
            term_months = 12
        
        return {
            "eligibility": eligibility,
            "recommended_amount": recommended_amount,
            "interest_rate": interest_rate,
            "term_months": term_months
        }
    
    def _generate_explanation(self, component_scores, credit_score, eligibility, sector):
        """Generate human-readable explanation for the credit decision."""
        config = self.sector_configs.get(sector, self.default_config)
        weights = config.get('feature_weights', self.default_config['feature_weights'])
        
        # Sort components by their contribution to the final score
        sorted_components = sorted(
            [(component, score, weights.get(component, 0)) 
             for component, score in component_scores.items()],
            key=lambda x: x[1] * x[2],
            reverse=True
        )
        
        explanation = {
            "summary": f"Credit score: {credit_score}, Decision: {eligibility}",
            "top_factors": [],
            "improvement_areas": []
        }
        
        # Add top positive factors
        for component, score, weight in sorted_components[:3]:
            if score >= 70:
                explanation["top_factors"].append({
                    "factor": component.replace("_", " ").title(),
                    "impact": "Positive",
                    "description": f"Score of {score:.1f}/100"
                })
        
        # Add areas for improvement
        for component, score, weight in sorted_components:
            if score < 60 and weight > 0.1:
                explanation["improvement_areas"].append({
                    "factor": component.replace("_", " ").title(),
                    "impact": "Negative",
                    "description": f"Score of {score:.1f}/100",
                    "recommendation": self._get_improvement_recommendation(component, score)
                })
        
        return explanation
    def _get_improvement_recommendation(self, component, score):
        """Generate specific improvement recommendations based on component and score."""
        recommendations = {
            "payment_history": "Improve repayment consistency and clear any outstanding debts.",
            "business_performance": "Work on increasing profit margins by reducing costs or increasing prices.",
            "cash_flow": "Stabilize cash flow by better managing receivables and payables.",
            "business_stability": "Continue business operations to build longer history.",
            "collateral": "Consider providing additional assets as collateral.",
            "seasonal_adjustment": "Develop more consistent income streams throughout the year."
        }
        
        return recommendations.get(component, "Review this area for potential improvements.")
    
    def assess_credit(self, client_data, financial_data, sector="retail", loan_request=None):
        """
        Perform credit assessment based on client and financial data.
        
        Args:
            client_data (dict): Client demographic and business information
            financial_data (dict): Financial history and metrics
            sector (str): Business sector/industry
            loan_request (dict): Details about the requested loan
            
        Returns:
            dict: Assessment results including score, eligibility, and explanations
        """
        self.logger.info(f"Processing credit assessment for sector: {sector}")
        
        # Validate inputs
        if not client_data or not financial_data:
            self.logger.error("Missing required data for credit assessment")
            return {"error": "Incomplete data provided"}
        
        try:
            # Generate assessment ID first
            assessment_id = f"NEXUS-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

            # Extract features from raw data
            features = self._extract_features(client_data, financial_data, sector)
            
            # Calculate component scores
            component_scores = self._calculate_component_scores(features, sector, financial_data)
            
            # Calculate final credit score
            credit_score = self._calculate_final_score(component_scores, sector)
            
            # Determine loan eligibility and terms
            loan_amount = loan_request.get('amount', 0) if loan_request else 0
            monthly_income = financial_data.get('average_monthly_income', 0)
            
            eligibility_result = self._determine_loan_eligibility(
                credit_score, sector, loan_amount, monthly_income
            )
            
            # Generate explanation
            explanation = self._generate_explanation(
                component_scores, credit_score, eligibility_result['eligibility'], sector
            )
            
            # Create the full assessment result
            assessment = {
                "client_id": client_data.get('client_id', 'unknown'),
                "assessment_id": assessment_id,
                "assessment_date": datetime.datetime.now().isoformat(),
                "sector": sector,
                "credit_score": credit_score,
                "component_scores": component_scores,
                "eligibility": eligibility_result['eligibility'],
                "loan_terms": {
                    "recommended_amount": eligibility_result['recommended_amount'],
                    "interest_rate": eligibility_result['interest_rate'],
                    "term_months": eligibility_result['term_months'],
                    "monthly_payment": self._calculate_monthly_payment(
                        eligibility_result['recommended_amount'],
                        eligibility_result['interest_rate'],
                        eligibility_result['term_months']
                    )
                },
                "explanation": explanation,
                "risk_factors": self._identify_risk_factors(features, component_scores, sector)
            }
            
            self.logger.info(f"Assessment completed: {assessment['assessment_id']}, Score: {credit_score}")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error in credit assessment: {str(e)}")
            return {"error": f"Assessment failed: {str(e)}",
            "assessment_id": assessment_id }
    
    def _calculate_monthly_payment(self, principal, annual_rate, term_months):
        """Calculate monthly loan payment."""
        if principal <= 0 or term_months <= 0:
            return 0
            
        monthly_rate = annual_rate / 100 / 12
        if monthly_rate == 0:
            return principal / term_months
            
        return principal * monthly_rate * (1 + monthly_rate) ** term_months / ((1 + monthly_rate) ** term_months - 1)
    
    def _identify_risk_factors(self, features, component_scores, sector):
        """Identify specific risk factors for this application."""
        risk_factors = []
        
        # Check for common risk indicators
        if component_scores.get('payment_history', 0) < 50:
            risk_factors.append({
                "factor": "Payment History",
                "severity": "High",
                "description": "Poor repayment history on previous loans."
            })
            
        if features.get('cash_flow_stability', 0) < 0.6:
            risk_factors.append({
                "factor": "Cash Flow",
                "severity": "Medium",
                "description": "Unstable or unpredictable cash flow patterns."
            })
            
        if features.get('business_age', 0) < 1:
            risk_factors.append({
                "factor": "Business Maturity",
                "severity": "Medium",
                "description": "Business has less than 1 year of operating history."
            })
            
        if features.get('expense_ratio', 0) > 0.8:
            risk_factors.append({
                "factor": "Expense Ratio",
                "severity": "High",
                "description": "High expense to income ratio indicates low profitability."
            })
            
        # Add sector-specific risk factors
        if sector == "agriculture" and features.get('seasonal_income_pattern', 0) > 0.7:
            risk_factors.append({
                "factor": "Seasonal Income",
                "severity": "Medium",
                "description": "Highly seasonal income pattern increases repayment risk."
            })
            
        if sector == "retail" and features.get('inventory_turnover', 0) < 4:
            risk_factors.append({
                "factor": "Inventory Turnover",
                "severity": "Medium",
                "description": "Low inventory turnover indicates potential cashflow issues."
            })
            
        return risk_factors
    
    def update_model(self, sector, training_data):
        """Update the model for a specific sector with new training data."""
        self.logger.info(f"Updating model for sector: {sector}")
        
        try:
            # In a real implementation, this would retrain the model with new data
            # For this example, we'll just log the request
            self.logger.info(f"Model update requested with {len(training_data)} samples")
            return {"status": "success", "message": "Model update simulation completed"}
            
        except Exception as e:
            self.logger.error(f"Error updating model: {str(e)}")
            return {"error": f"Model update failed: {str(e)}"}
    
    def export_model(self, sector, filepath):
        """Export the trained model for a specific sector."""
        if sector not in self.models:
            self._create_fallback_model(sector)
            
        try:
            joblib.dump(self.models[sector], filepath)
            return {"status": "success", "path": filepath}
        except Exception as e:
            return {"error": f"Export failed: {str(e)}"}
    
    def import_model(self, sector, filepath):
        """Import a pre-trained model for a specific sector."""
        try:
            self.models[sector] = joblib.load(filepath)
            return {"status": "success", "sector": sector}
        except Exception as e:
            return {"error": f"Import failed: {str(e)}"}

    #Database Methods

    def connect_to_database(self, db_config):
        """Connect to PostgreSQL database."""
        try:
            self.db_connection = psycopg2.connect(
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 5432),
                database=db_config.get('database'),
                user=db_config.get('user'),
                password=db_config.get('password')
            )
            self.logger.info(f"Connected to database: {db_config.get('database')}")
            return True
        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            return False
    
    def close_database_connection(self):
        """Close the database connection."""
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
            self.logger.info("Database connection closed")
    
    def get_client_data_from_db(self, client_id):
        """Retrieve client data from the database."""
        if not self.db_connection:
            self.logger.error("No database connection established")
            return None
        
        try:
            cursor = self.db_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("""
                SELECT client_id, name, age, education, dependents, 
                       business_name, business_type, business_age, location_type
                FROM clients
                WHERE client_id = %s
            """, (client_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return dict(result)
            else:
                self.logger.warning(f"No client found with ID: {client_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving client data: {str(e)}")
            return None
    
    def get_financial_data_from_db(self, client_id):
        """Retrieve financial data from the database."""
        if not self.db_connection:
            self.logger.error("No database connection established")
            return None
        
        try:
            cursor = self.db_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("""
                SELECT average_monthly_income, average_monthly_expenses, 
                       expense_to_income_ratio, average_monthly_profit,
                       cash_flow_stability_index, previous_loans_count,
                       previous_repayment_rate, average_days_overdue,
                       inventory_turnover, customer_base_size, location_quality,
                       collateral_value_ratio
                FROM financial_data
                WHERE client_id = %s
                ORDER BY report_date DESC
                LIMIT 1
            """, (client_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return {
                'average_monthly_income': float(result['average_monthly_income']),
                'average_monthly_expenses': float(result['average_monthly_expenses']),
                'expense_to_income_ratio': float(result['expense_to_income_ratio']),
                'average_monthly_profit': float(result['average_monthly_profit']),
                'cash_flow_stability_index': float(result['cash_flow_stability_index']),
                'previous_loans_count': float(result['previous_loans_count']),
                'previous_repayment_rate': float(result['previous_repayment_rate']),
                'average_days_overdue': float(result['average_days_overdue']),
                'inventory_turnover': float(result['inventory_turnover']),
                'customer_base_size': float(result['customer_base_size']),
                'location_quality': result['location_quality'],
                'collateral_value_ratio': float(result['collateral_value_ratio']),
            }
            else:
                self.logger.warning(f"No financial data found for client ID: {client_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving financial data: {str(e)}")
            return None
    
    def get_loan_request_from_db(self, request_id):
        """Retrieve loan request from the database."""
        if not self.db_connection:
            self.logger.error("No database connection established")
            return None
        
        try:
            cursor = self.db_connection.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("""
                SELECT request_id, client_id, amount, purpose, term_requested
                FROM loan_requests
                WHERE request_id = %s
            """, (request_id,))
            
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return dict(result)
            else:
                self.logger.warning(f"No loan request found with ID: {request_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving loan request: {str(e)}")
            return None
    
    def save_assessment_result_to_db(self, assessment_result):
        """Save assessment result to the database."""
        if not self.db_connection:
            self.logger.error("No database connection established")
            return False
        
        if 'assessment_id' not in assessment_result:
            self.logger.error("Missing assessment_id in result")
            return False
        
        try:
            cursor = self.db_connection.cursor()
            
            # Save main assessment record
            cursor.execute("""
                INSERT INTO credit_assessments (
                    assessment_id, client_id, assessment_date, sector, 
                    credit_score, eligibility, recommended_amount, 
                    interest_rate, term_months, monthly_payment
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                assessment_result['assessment_id'],
                assessment_result['client_id'],
                assessment_result['assessment_date'],
                assessment_result['sector'],
                assessment_result['credit_score'],
                assessment_result['eligibility'],
                assessment_result['loan_terms']['recommended_amount'],
                assessment_result['loan_terms']['interest_rate'],
                assessment_result['loan_terms']['term_months'],
                assessment_result['loan_terms']['monthly_payment']
            ))
            
            assessment_db_id = cursor.fetchone()[0]
            
            # Save component scores
            for component, score in assessment_result['component_scores'].items():
                cursor.execute("""
                    INSERT INTO assessment_components (
                        assessment_id, component_name, score
                    ) VALUES (%s, %s, %s)
                """, (
                    assessment_db_id,
                    component,
                    score
                ))
            
            # Save risk factors
            for risk in assessment_result['risk_factors']:
                cursor.execute("""
                    INSERT INTO assessment_risk_factors (
                        assessment_id, factor, severity, description
                    ) VALUES (%s, %s, %s, %s)
                """, (
                    assessment_db_id,
                    risk['factor'],
                    risk['severity'],
                    risk['description']
                ))
            
            self.db_connection.commit()
            cursor.close()
            
            self.logger.info(f"Assessment {assessment_result['assessment_id']} saved to database")
            return True
            
        except Exception as e:
            self.db_connection.rollback()
            self.logger.error(f"Error saving assessment result: {str(e)}")
            return False
    
    def assess_credit_from_db(self, client_id, loan_request_id=None, sector=None):
        """Perform credit assessment using data from the database."""
        # Get client data
        client_data = self.get_client_data_from_db(client_id)
        if not client_data:
            return {"error": f"Client with ID {client_id} not found"}
        
        # Get financial data
        financial_data = self.get_financial_data_from_db(client_id)
        if not financial_data:
            return {"error": f"Financial data for client {client_id} not found"}
        
        # Get loan request if provided
        loan_request = None
        if loan_request_id:
            loan_request = self.get_loan_request_from_db(loan_request_id)
        
        # Use client's business sector if not specified
        if not sector:
            sector = client_data.get('business_type', 'retail')
        
        # Perform assessment
        assessment_result = self.assess_credit(client_data, financial_data, sector, loan_request)
        
        # Save result to database
        if 'error' not in assessment_result:
            self.save_assessment_result_to_db(assessment_result)
        
        return assessment_result



    # Perform credit assessment
if __name__ == "__main__":
    # Initialize engine with database config
    engine = NEXUSCreditEngine(db_config={
        'host': 'localhost',
        'database': 'nexus',
        'port': 5050,
        'user': 'postgres',
        'password': 'admin'
    })
    
    # Perform assessment for client 1
    result = engine.assess_credit_from_db(client_id=1)
    
    # Print the results
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Credit Score: {result['credit_score']}")
        print(f"Decision: {result['eligibility']}")
        print(f"Recommended Amount: ${result['loan_terms']['recommended_amount']:.2f}")
        print(f"Interest Rate: {result['loan_terms']['interest_rate']}%")
        print(f"Term: {result['loan_terms']['term_months']} months")
        print(f"Monthly Payment: ${result['loan_terms']['monthly_payment']:.2f}")
        print("\nTop Factors:")
        for factor in result['explanation']['top_factors']:
            print(f"- {factor['factor']}: {factor['description']}")
            print("\nImprovement Areas:")
        for area in result['explanation']['improvement_areas']:
            print(f"- {area['factor']}: {area['description']}")
            print(f"  Recommendation: {area['recommendation']}")

    # Save assessment to database
    engine.save_assessment_result_to_db(result)