-- SnowPrep Deployment Script
-- Use this script to deploy the Native App to your Snowflake account

-- ================================
-- STEP 1: Create Application Package
-- ================================

-- Switch to ACCOUNTADMIN role (required for Native App operations)
USE ROLE ACCOUNTADMIN;

-- Create the application package
CREATE APPLICATION PACKAGE IF NOT EXISTS snowprep_package
  COMMENT = 'SnowPrep Visual SQL Builder - Application Package';

-- Use the package
USE APPLICATION PACKAGE snowprep_package;

-- Create a stage for the application files
CREATE STAGE IF NOT EXISTS app_stage
  FILE_FORMAT = (TYPE = CSV)
  COMMENT = 'Stage for SnowPrep application files';

-- ================================
-- STEP 2: Upload Application Files
-- ================================

-- Upload all application files to the stage
-- Note: You'll need to run these PUT commands from SnowSQL or equivalent
-- PUT file://manifest.yml @app_stage overwrite=true;
-- PUT file://README.md @app_stage overwrite=true;  
-- PUT file://scripts/setup.sql @app_stage/scripts/ overwrite=true;
-- PUT file://scripts/post_setup.sql @app_stage/scripts/ overwrite=true;
-- PUT file://streamlit/snowprep_app.py @app_stage/streamlit/ overwrite=true;
-- PUT file://streamlit/environment.yml @app_stage/streamlit/ overwrite=true;

-- Verify files were uploaded
LIST @app_stage;

-- ================================
-- STEP 3: Create Application Version
-- ================================

-- Create the application version
CREATE VERSION v1_0 USING @app_stage
  COMMENT = 'SnowPrep v1.0 - Initial release';

-- Set as default version
ALTER APPLICATION PACKAGE snowprep_package 
  SET DEFAULT RELEASE DIRECTIVE VERSION = v1_0 PATCH = 0;

-- ================================
-- STEP 4: Create Application Instance
-- ================================

-- Create the application instance
CREATE APPLICATION IF NOT EXISTS snowprep_app
  FROM APPLICATION PACKAGE snowprep_package
  USING VERSION v1_0
  COMMENT = 'SnowPrep Visual SQL Builder Instance';

-- ================================
-- STEP 5: Grant Privileges
-- ================================

-- Grant necessary privileges to roles that should access the app
-- Adjust these grants based on your organization's needs

-- Example: Grant to a specific role
-- GRANT USAGE ON APPLICATION snowprep_app TO ROLE data_analyst_role;
-- GRANT USAGE ON APPLICATION snowprep_app TO ROLE business_user_role;

-- Example: Grant to all users (use with caution)
-- GRANT USAGE ON APPLICATION snowprep_app TO ROLE PUBLIC;

-- ================================
-- STEP 6: Configure Application References
-- ================================

-- Set up warehouse reference (users will be prompted to select)
-- This allows the app to use warehouses for query execution
ALTER APPLICATION snowprep_app 
  SET REFERENCE warehouse_reference = YOUR_WAREHOUSE_NAME;

-- ================================
-- STEP 7: Verify Installation
-- ================================

-- Check application status
SHOW APPLICATIONS LIKE 'snowprep_app';

-- View application details
DESCRIBE APPLICATION snowprep_app;

-- Test the application
-- You can now access SnowPrep through the Snowflake UI under Apps section

-- ================================
-- OPTIONAL: Monitoring and Maintenance
-- ================================

-- Create monitoring view for app usage
CREATE OR REPLACE VIEW snowprep_monitoring AS
SELECT 
  app_name,
  user_name,
  execution_time,
  query_id,
  warehouse_name,
  database_name,
  schema_name
FROM snowflake.account_usage.query_history
WHERE query_text LIKE '%snowprep_app%'
  AND start_time >= DATEADD(day, -30, CURRENT_TIMESTAMP())
ORDER BY start_time DESC;

-- ================================
-- Deployment Complete!
-- ================================

-- Your SnowPrep Native App is now deployed and ready to use.
-- 
-- Next steps:
-- 1. Access the app through Snowflake UI > Apps section
-- 2. Configure user access and permissions as needed
-- 3. Test the application with sample data
-- 4. Provide training to end users
--
-- For troubleshooting, check:
-- - Application logs: SELECT * FROM snowprep_app.core.query_history;
-- - Configuration: SELECT * FROM snowprep_app.core.app_config;
-- - User activity: SELECT * FROM snowprep_app.core.app_usage_stats;