INSERT INTO clients (name, age, education, dependents, business_name, business_type, business_age, location_type) 
VALUES
('John Doe', 35, 'Bachelor''s Degree', 2, 'QuickMart', 'retail', 5, 'urban'),
('Maria Garcia', 42, 'High School Diploma', 3, 'GreenFields', 'agriculture', 8, 'rural'),
('Ahmed Khan', 28, 'Master''s Degree', 0, 'TechSolutions', 'service', 3, 'suburban');


INSERT INTO financial_data (client_id, average_monthly_income, average_monthly_expenses, expense_to_income_ratio, average_monthly_profit, cash_flow_stability_index, previous_loans_count, previous_repayment_rate, average_days_overdue, inventory_turnover, customer_base_size, location_quality, collateral_value_ratio)
VALUES
(1, 4500.00, 3200.00, 0.71, 1300.00, 0.85, 2, 0.95, 5, 6.2, 1500, 'good', 0.75),
(2, 2800.00, 2200.00, 0.79, 600.00, 0.65, 1, 0.88, 12, 3.8, 300, 'fair', 0.60),
(3, 6800.00, 4200.00, 0.62, 2600.00, 0.92, 0, 1.00, 0, 9.1, 450, 'excellent', 0.85);


INSERT INTO loan_requests (client_id, amount, purpose, term_requested)
VALUES
(1, 15000.00, 'Inventory purchase', 12),
(2, 25000.00, 'Farm equipment upgrade', 18),
(3, 10000.00, 'Marketing campaign', 6);