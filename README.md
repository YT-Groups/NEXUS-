# README.md content

# NEXUS Credit Assessment Engine

NEXUS (Neural Evaluation for Xpanded User Scoring) is a credit assessment engine designed for offline microfinance institutions to evaluate creditworthiness across various sectors.

## Project Structure

- `database/`: Contains database connection logic.
- `models/`: Contains database models corresponding to the SQL tables.
- `NEXUS.py`: Main logic for the credit assessment engine.
- `config/`: Contains configuration files, including database settings.

- `requirements.txt`: Lists the dependencies required for the project.

## Setup Instructions

1. Clone the repository:

   ```
   git clone <repository-url>
   cd NEXUS-
   ```

2. Install the required dependencies (Create a virtual environment with Python3.11):

   ```
   pip install -r requirements.txt
   ```

3. Configure the database settings in `config/.env`.

4. Initializing the database:
   With PostgreSQL installed create a database and user:

   ```
   CREATE DATABASE nexus;
   CREATE USER postgres WITH PASSWORD 'admin';
   GRANT ALL PRIVILEGES ON DATABASE nexus TO postgres;
   ```

   Initialize the database using the NEXUS_DATABASE_TEST schema:
   Be sure to specify the port your Postgres Server is running on in place of <your_port>

   ```
   psql -U postgres -d nexus -p <your_port> -f database/NEXUS_DATABASE_TEST.sql
   ```

   Populate the database with test data (insert_test_data):

   ```
   psql -U postgres -d nexus -p <your_port> -f database/insert_test_data.sql
   ```

5. Run the application:
   ```
   python NEXUS.py
   ```

## Usage

To assess credit, use the `assess_credit_from_db` method in the `NEXUS.py` file, providing the necessary client id.

## License

This project is licensed under the MIT License.
