-- SnowPrep Uninstall Script
-- Use this script to completely remove the SnowPrep Native App

-- ================================
-- WARNING: DESTRUCTIVE OPERATIONS
-- ================================
-- This script will permanently delete:
-- - The application instance
-- - All stored data (query history, user preferences)
-- - Application package and versions
-- 
-- Make sure to backup any important data before proceeding!

-- Switch to ACCOUNTADMIN role
USE ROLE ACCOUNTADMIN;

-- ================================
-- STEP 1: Export Data (Optional)
-- ================================
-- Uncomment and modify these queries to backup your data before uninstalling

-- Backup query history
-- CREATE TABLE backup_snowprep_query_history AS 
-- SELECT * FROM snowprep_app.core.query_history;

-- Backup user preferences  
-- CREATE TABLE backup_snowprep_user_preferences AS
-- SELECT * FROM snowprep_app.core.user_preferences;

-- Backup app configuration
-- CREATE TABLE backup_snowprep_app_config AS
-- SELECT * FROM snowprep_app.core.app_config;

-- ================================
-- STEP 2: Remove Application Instance
-- ================================

-- Check current applications
SHOW APPLICATIONS LIKE 'snowprep_app';

-- Drop the application instance
-- This removes the application and all its data
DROP APPLICATION IF EXISTS snowprep_app;

-- ================================
-- STEP 3: Remove Application Package
-- ================================

-- Check current packages
SHOW APPLICATION PACKAGES LIKE 'snowprep_package';

-- Drop all versions first
USE APPLICATION PACKAGE snowprep_package;
SHOW VERSIONS IN APPLICATION PACKAGE;

-- Drop specific versions (adjust version names as needed)
-- DROP VERSION IF EXISTS v1_0;
-- DROP VERSION IF EXISTS v1_1;

-- Drop the application package
DROP APPLICATION PACKAGE IF EXISTS snowprep_package;

-- ================================
-- STEP 4: Clean Up Grants and References
-- ================================

-- Remove any grants that were made to roles
-- Note: These will show errors if grants don't exist, which is normal
-- REVOKE USAGE ON APPLICATION snowprep_app FROM ROLE data_analyst_role;
-- REVOKE USAGE ON APPLICATION snowprep_app FROM ROLE business_user_role;
-- REVOKE USAGE ON APPLICATION snowprep_app FROM ROLE PUBLIC;

-- ================================
-- STEP 5: Clean Up Monitoring Objects
-- ================================

-- Remove monitoring views created during deployment
DROP VIEW IF EXISTS snowprep_monitoring;

-- Clean up any warehouses that were created specifically for the app
-- DROP WAREHOUSE IF EXISTS snowprep_warehouse;

-- ================================
-- STEP 6: Verification
-- ================================

-- Verify applications are removed
SHOW APPLICATIONS;
SHOW APPLICATION PACKAGES;

-- Check for any remaining references
-- SHOW GRANTS TO APPLICATION snowprep_app; -- Should show "no data"

-- ================================
-- Uninstall Complete!
-- ================================

-- The SnowPrep Native App has been completely removed from your account.
-- 
-- What was removed:
-- ✓ Application instance (snowprep_app)
-- ✓ Application package (snowprep_package)
-- ✓ All application data (query history, preferences, config)
-- ✓ Streamlit interface and associated resources
-- ✓ Database schemas and tables created by the app
--
-- What was NOT removed:
-- • Your original data (databases, schemas, tables)
-- • Warehouses (unless specifically created for the app)
-- • User roles and permissions
-- • Any backup tables you may have created
--
-- To reinstall SnowPrep, run the deployment script again.