# SnowPrep Native App - Deployment Guide

This guide walks you through deploying SnowPrep as a Snowflake Native Application.

## Prerequisites

### Required Roles and Privileges
- **ACCOUNTADMIN** role (for initial setup and deployment)
- **CREATE APPLICATION PACKAGE** privilege
- **CREATE APPLICATION** privilege
- Access to a warehouse for query execution

### Required Tools
- **SnowSQL** (recommended) or Snowflake Web UI
- Access to upload files to Snowflake stages

## Deployment Steps

### 1. Prepare Application Files

Ensure you have all the required files in the `snowprep-native-app/` directory:

```
snowprep-native-app/
├── manifest.yml                 # Application manifest
├── README.md                   # Application documentation
├── scripts/
│   ├── setup.sql              # Initial setup script
│   ├── post_setup.sql         # Post-setup configuration
│   ├── deploy.sql             # Deployment automation
│   └── uninstall.sql          # Uninstall script
└── streamlit/
    ├── snowprep_app.py        # Main Streamlit application
    └── environment.yml        # Python dependencies
```

### 2. Connect to Snowflake

Using SnowSQL:
```bash
snowsql -a your_account.snowflakecomputing.com -u your_username
```

Or use the Snowflake Web UI with a worksheet.

### 3. Run Deployment Script

#### Option A: Automated Deployment
```sql
-- Execute the complete deployment script
@scripts/deploy.sql
```

#### Option B: Manual Step-by-Step

1. **Create Application Package**:
```sql
USE ROLE ACCOUNTADMIN;
CREATE APPLICATION PACKAGE IF NOT EXISTS snowprep_package;
USE APPLICATION PACKAGE snowprep_package;
```

2. **Create Stage and Upload Files**:
```sql
CREATE STAGE IF NOT EXISTS app_stage;
```

Using SnowSQL to upload files:
```bash
PUT file://manifest.yml @app_stage overwrite=true;
PUT file://README.md @app_stage overwrite=true;
PUT file://scripts/setup.sql @app_stage/scripts/ overwrite=true;
PUT file://scripts/post_setup.sql @app_stage/scripts/ overwrite=true;
PUT file://streamlit/snowprep_app.py @app_stage/streamlit/ overwrite=true;
PUT file://streamlit/environment.yml @app_stage/streamlit/ overwrite=true;
```

3. **Create Version**:
```sql
CREATE VERSION v1_0 USING @app_stage;
ALTER APPLICATION PACKAGE snowprep_package 
  SET DEFAULT RELEASE DIRECTIVE VERSION = v1_0 PATCH = 0;
```

4. **Create Application Instance**:
```sql
CREATE APPLICATION snowprep_app
  FROM APPLICATION PACKAGE snowprep_package
  USING VERSION v1_0;
```

### 4. Configure Application Access

Grant usage to appropriate roles:

```sql
-- For specific roles
GRANT USAGE ON APPLICATION snowprep_app TO ROLE data_analyst_role;
GRANT USAGE ON APPLICATION snowprep_app TO ROLE business_user_role;

-- For broader access (use with caution)
-- GRANT USAGE ON APPLICATION snowprep_app TO ROLE PUBLIC;
```

Set up warehouse reference:
```sql
ALTER APPLICATION snowprep_app 
  SET REFERENCE warehouse_reference = your_warehouse_name;
```

### 5. Verify Installation

```sql
-- Check application status
SHOW APPLICATIONS LIKE 'snowprep_app';
DESCRIBE APPLICATION snowprep_app;

-- Test application access
SELECT * FROM snowprep_app.core.app_config;
```

## Post-Deployment Configuration

### Grant Data Access

Users need access to databases they want to explore:

```sql
-- Example: Grant access to specific databases
GRANT USAGE ON DATABASE sample_db TO APPLICATION snowprep_app;
GRANT USAGE ON ALL SCHEMAS IN DATABASE sample_db TO APPLICATION snowprep_app;
GRANT SELECT ON ALL TABLES IN DATABASE sample_db TO APPLICATION snowprep_app;
GRANT SELECT ON ALL VIEWS IN DATABASE sample_db TO APPLICATION snowprep_app;

-- For future objects
GRANT SELECT ON FUTURE TABLES IN DATABASE sample_db TO APPLICATION snowprep_app;
GRANT SELECT ON FUTURE VIEWS IN DATABASE sample_db TO APPLICATION snowprep_app;
```

### Configure Application Settings

```sql
-- Adjust query limits
UPDATE snowprep_app.core.app_config 
SET config_value = 5000 
WHERE config_key = 'max_query_limit';

-- Enable/disable query logging
UPDATE snowprep_app.core.app_config 
SET config_value = true 
WHERE config_key = 'enable_query_logging';
```

## Accessing the Application

1. **Snowflake Web UI**: Navigate to **Apps** section and click on **SnowPrep**
2. **Direct URL**: Use the application URL provided in the deployment output
3. **Streamlit Interface**: The app will open in a new tab with the visual query builder

## Monitoring and Maintenance

### Monitor Usage
```sql
-- View application usage statistics
SELECT * FROM snowprep_app.core.app_usage_stats;

-- Check recent queries
SELECT * FROM snowprep_app.core.recent_queries;
```

### Update Application
To update the application with new features:

1. Upload new files to the stage
2. Create a new version
3. Upgrade the application instance

```sql
-- Create new version
CREATE VERSION v1_1 USING @app_stage;

-- Upgrade application
ALTER APPLICATION snowprep_app UPGRADE USING VERSION v1_1;
```

## Troubleshooting

### Common Issues

**Issue**: "Application package not found"
- **Solution**: Ensure you're using ACCOUNTADMIN role and the package was created successfully

**Issue**: "No databases visible" in the app
- **Solution**: Grant the application access to your databases and schemas

**Issue**: "Query failed" errors
- **Solution**: Check warehouse permissions and ensure the application has access to query the data

**Issue**: Streamlit app won't load
- **Solution**: Verify all files were uploaded correctly and the Python environment is configured properly

### Getting Help

1. Check application logs:
```sql
SELECT * FROM snowprep_app.information_schema.event_table 
WHERE timestamp >= DATEADD(hour, -1, CURRENT_TIMESTAMP());
```

2. Verify configuration:
```sql
SELECT * FROM snowprep_app.core.app_config;
```

3. Test basic functionality:
```sql
SELECT snowprep_app.core.has_database_access('YOUR_DB_NAME');
```

## Uninstalling

To completely remove the application:

```sql
@scripts/uninstall.sql
```

**Warning**: This will permanently delete all application data including query history and user preferences.

## Security Considerations

- The application runs within your Snowflake environment with your security controls
- All data access is governed by existing Snowflake privileges
- Query history is stored within the application's database
- No data leaves your Snowflake account

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Snowflake's Native Apps documentation
3. Contact your Snowflake administrator
4. Visit [datatoolspro.com](https://datatoolspro.com/) for additional support

---

**Congratulations! Your SnowPrep Native App is now deployed and ready for use.**