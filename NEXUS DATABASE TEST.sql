CREATE TABLE clients (
    client_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    age INT,
    education VARCHAR(100),
    dependents INT,
    business_name VARCHAR(255),
    business_type VARCHAR(100),
    business_age INT,
    location_type VARCHAR(100)
);

CREATE TABLE financial_data (
    id SERIAL PRIMARY KEY,
    client_id INT REFERENCES clients(client_id) ON DELETE CASCADE,
    report_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    average_monthly_income DECIMAL(12,2),
    average_monthly_expenses DECIMAL(12,2),
    expense_to_income_ratio DECIMAL(5,2),
    average_monthly_profit DECIMAL(12,2),
    cash_flow_stability_index DECIMAL(5,2),
    previous_loans_count INT,
    previous_repayment_rate DECIMAL(5,2),
    average_days_overdue INT,
    inventory_turnover DECIMAL(5,2),
    customer_base_size INT,
    location_quality VARCHAR(100),
    collateral_value_ratio DECIMAL(5,2)
);

CREATE TABLE loan_requests (
    request_id SERIAL PRIMARY KEY,
    client_id INT REFERENCES clients(client_id) ON DELETE CASCADE,
    amount DECIMAL(12,2),
    purpose TEXT,
    term_requested INT,
    request_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE credit_assessments (
    id SERIAL PRIMARY KEY,
    assessment_id VARCHAR(50) UNIQUE,
    client_id INT REFERENCES clients(client_id) ON DELETE CASCADE,
    assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sector VARCHAR(100),
    credit_score INT,
    eligibility VARCHAR(20),
    recommended_amount DECIMAL(12,2),
    interest_rate DECIMAL(5,2),
    term_months INT,
    monthly_payment DECIMAL(12,2)
);

CREATE TABLE assessment_components (
    id SERIAL PRIMARY KEY,
    assessment_id INT REFERENCES credit_assessments(id) ON DELETE CASCADE,
    component_name VARCHAR(100),
    score DECIMAL(5,2)
);

CREATE TABLE assessment_risk_factors (
    id SERIAL PRIMARY KEY,
    assessment_id INT REFERENCES credit_assessments(id) ON DELETE CASCADE,
    factor VARCHAR(255),
    severity VARCHAR(50),
    description TEXT
);
