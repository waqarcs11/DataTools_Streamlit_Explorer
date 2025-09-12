-- SnowPrep Native App Setup Script
-- This script runs when the app is installed

-- Create application database and schema
CREATE APPLICATION ROLE app_user;
CREATE SCHEMA IF NOT EXISTS core;

-- Create stage for Streamlit files
CREATE STAGE IF NOT EXISTS app_stage.streamlit
  DIRECTORY = ( ENABLE = true )
  COMMENT = 'Stage for SnowPrep Streamlit application files';

-- Grant necessary privileges to application roles
GRANT USAGE ON SCHEMA core TO APPLICATION ROLE app_user;
GRANT USAGE ON STAGE app_stage.streamlit TO APPLICATION ROLE app_user;

-- Create configuration table for app settings
CREATE TABLE IF NOT EXISTS core.app_config (
  config_key VARCHAR(100) PRIMARY KEY,
  config_value VARIANT,
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE core.app_config TO APPLICATION ROLE app_user;

-- Create user preferences table
CREATE TABLE IF NOT EXISTS core.user_preferences (
  user_name VARCHAR(255),
  preference_key VARCHAR(100),
  preference_value VARIANT,
  created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
  updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
  PRIMARY KEY (user_name, preference_key)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE core.user_preferences TO APPLICATION ROLE app_user;

-- Create query history table for tracking user queries
CREATE TABLE IF NOT EXISTS core.query_history (
  query_id NUMBER AUTOINCREMENT PRIMARY KEY,
  user_name VARCHAR(255),
  database_name VARCHAR(255),
  schema_name VARCHAR(255),
  table_name VARCHAR(255),
  sql_query CLOB,
  execution_time TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
  rows_returned NUMBER,
  execution_status VARCHAR(20) DEFAULT 'SUCCESS'
);

GRANT SELECT, INSERT, UPDATE ON TABLE core.query_history TO APPLICATION ROLE app_user;

-- Create view for recent queries
CREATE OR REPLACE VIEW core.recent_queries AS
SELECT 
  query_id,
  user_name,
  database_name || '.' || schema_name || '.' || table_name as full_table_name,
  LEFT(sql_query, 100) as sql_preview,
  execution_time,
  rows_returned,
  execution_status
FROM core.query_history
WHERE execution_time >= DATEADD(day, -30, CURRENT_TIMESTAMP())
ORDER BY execution_time DESC
LIMIT 100;

GRANT SELECT ON VIEW core.recent_queries TO APPLICATION ROLE app_user;

-- Create stored procedures for app functionality
CREATE OR REPLACE PROCEDURE core.log_query(
  p_user_name VARCHAR(255),
  p_database_name VARCHAR(255), 
  p_schema_name VARCHAR(255),
  p_table_name VARCHAR(255),
  p_sql_query CLOB,
  p_rows_returned NUMBER DEFAULT NULL
)
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
  INSERT INTO core.query_history (
    user_name, database_name, schema_name, table_name, 
    sql_query, rows_returned
  ) VALUES (
    :p_user_name, :p_database_name, :p_schema_name, :p_table_name,
    :p_sql_query, :p_rows_returned
  );
  
  RETURN 'Query logged successfully';
END;
$$;

GRANT USAGE ON PROCEDURE core.log_query(VARCHAR, VARCHAR, VARCHAR, VARCHAR, CLOB, NUMBER) TO APPLICATION ROLE app_user;

-- Create procedure to get user's database access
CREATE OR REPLACE PROCEDURE core.get_accessible_databases()
RETURNS TABLE (database_name VARCHAR)
LANGUAGE SQL
AS
$$
BEGIN
  LET result_set RESULTSET := (
    SELECT database_name 
    FROM information_schema.databases 
    WHERE database_name NOT IN ('SNOWFLAKE', 'INFORMATION_SCHEMA')
    ORDER BY database_name
  );
  RETURN TABLE(result_set);
END;
$$;

GRANT USAGE ON PROCEDURE core.get_accessible_databases() TO APPLICATION ROLE app_user;

-- Set up initial configuration
INSERT INTO core.app_config (config_key, config_value) VALUES
  ('app_version', '1.0.0'),
  ('max_query_limit', 10000),
  ('default_warehouse', NULL),
  ('enable_query_logging', true)
ON DUPLICATE KEY UPDATE 
  updated_at = CURRENT_TIMESTAMP();

-- Create Streamlit app
CREATE STREAMLIT snowprep_streamlit
  FROM 'app_stage.streamlit'
  MAIN_FILE = 'snowprep_app.py'
  TITLE = 'SnowPrep - Visual SQL Builder'
  COMMENT = 'Interactive visual SQL builder for Snowflake data exploration';

-- Grant access to Streamlit app
GRANT USAGE ON STREAMLIT snowprep_streamlit TO APPLICATION ROLE app_user;