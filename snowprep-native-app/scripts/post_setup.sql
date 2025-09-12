-- SnowPrep Native App Post-Setup Script
-- This script runs after the main setup is complete

-- Create additional helper functions and views
USE SCHEMA core;

-- Function to check if user has access to a specific database
CREATE OR REPLACE FUNCTION core.has_database_access(db_name VARCHAR)
RETURNS BOOLEAN
LANGUAGE SQL
AS
$$
  SELECT COUNT(*) > 0 
  FROM information_schema.databases 
  WHERE database_name = UPPER(db_name)
$$;

-- Function to get column statistics for a table
CREATE OR REPLACE FUNCTION core.get_column_stats(
  db_name VARCHAR, 
  schema_name VARCHAR, 
  table_name VARCHAR
)
RETURNS TABLE (
  column_name VARCHAR,
  data_type VARCHAR,
  is_nullable VARCHAR,
  column_default VARCHAR,
  ordinal_position NUMBER
)
LANGUAGE SQL
AS
$$
  SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default,
    ordinal_position
  FROM information_schema.columns
  WHERE table_schema = UPPER(schema_name)
    AND table_name = UPPER(table_name)
    AND table_catalog = UPPER(db_name)
  ORDER BY ordinal_position
$$;

-- Create a view for monitoring app usage
CREATE OR REPLACE VIEW core.app_usage_stats AS
SELECT 
  DATE_TRUNC('day', execution_time) as usage_date,
  user_name,
  COUNT(*) as query_count,
  COUNT(DISTINCT database_name || '.' || schema_name || '.' || table_name) as unique_tables_accessed,
  AVG(rows_returned) as avg_rows_returned
FROM core.query_history
WHERE execution_time >= DATEADD(day, -90, CURRENT_TIMESTAMP())
GROUP BY DATE_TRUNC('day', execution_time), user_name
ORDER BY usage_date DESC, query_count DESC;

-- Grant permissions for the new objects
GRANT USAGE ON FUNCTION core.has_database_access(VARCHAR) TO APPLICATION ROLE app_user;
GRANT USAGE ON FUNCTION core.get_column_stats(VARCHAR, VARCHAR, VARCHAR) TO APPLICATION ROLE app_user;
GRANT SELECT ON VIEW core.app_usage_stats TO APPLICATION ROLE app_user;

-- Initialize application state
UPDATE core.app_config 
SET config_value = CURRENT_TIMESTAMP(), updated_at = CURRENT_TIMESTAMP()
WHERE config_key = 'last_setup_time';