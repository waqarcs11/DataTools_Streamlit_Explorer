# SnowPrep - Visual SQL Builder for Snowflake

**Transform data exploration with an intuitive, visual interface for building complex SQL queries in Snowflake.**

## Overview

SnowPrep is a powerful Snowflake Native App that provides a user-friendly, visual interface for building SQL queries. Whether you're a data analyst, business user, or SQL expert, SnowPrep makes data exploration faster, easier, and more intuitive.

## Key Features

### 🎯 Visual Query Builder
- **Point-and-click interface** - No need to write complex SQL from scratch
- **Real-time SQL preview** - See your query as you build it
- **Drag-and-drop functionality** - Intuitive column selection and configuration

### 📊 Advanced Analytics
- **Dimensions & Measures** - Build aggregated queries with ease
- **Smart filtering** - Multiple filter types with AND/OR logic
- **Custom aggregations** - Support for COUNT, SUM, AVG, MIN, MAX
- **Having clauses** - Filter on aggregated results

### 🔧 Enterprise Features
- **Query history tracking** - Keep track of all your queries
- **User preferences** - Personalized settings and favorites
- **Performance monitoring** - Built-in query performance tracking
- **Secure execution** - All queries run within your Snowflake environment

### 📈 Data Visualization
- **Automatic chart suggestions** - Quick visualizations for grouped data
- **Export capabilities** - Easy data export and sharing
- **Interactive results** - Explore your query results interactively

## Installation

### Prerequisites
- Snowflake account with appropriate privileges
- Access to install Native Apps (ACCOUNTADMIN or APP_ADMIN role)

### Installation Steps

1. **Create Application Package** (if deploying from source):
```sql
CREATE APPLICATION PACKAGE snowprep_package;
USE APPLICATION PACKAGE snowprep_package;

-- Upload application files to stage
CREATE STAGE app_stage;
PUT file://snowprep-native-app/* @app_stage;

-- Create version
CREATE VERSION v1_0 USING @app_stage;
```

2. **Install Application**:
```sql
-- Create application from package
CREATE APPLICATION snowprep_app 
FROM APPLICATION PACKAGE snowprep_package 
USING VERSION v1_0;

-- Grant necessary privileges
GRANT INSTALL ON APPLICATION PACKAGE snowprep_package TO ROLE your_role;
GRANT CREATE APPLICATION ON ACCOUNT TO ROLE your_role;
```

3. **Configure Application**:
```sql
-- Grant application access to your data
GRANT USAGE ON WAREHOUSE your_warehouse TO APPLICATION snowprep_app;
GRANT USAGE ON DATABASE your_database TO APPLICATION snowprep_app;
GRANT USAGE ON ALL SCHEMAS IN DATABASE your_database TO APPLICATION snowprep_app;
GRANT SELECT ON ALL TABLES IN DATABASE your_database TO APPLICATION snowprep_app;
```

## Usage Guide

### Getting Started

1. **Launch SnowPrep**: Navigate to your Snowflake UI and open the SnowPrep application
2. **Select Data Source**: Choose database, schema, and table/view from the sidebar
3. **Build Your Query**: Use the visual interface to configure your query
4. **Execute**: Click "Run Query" to see results

### Building Queries

#### Basic Queries
- Select columns using the multiselect dropdown
- Apply filters using intuitive operators
- Sort results with the ORDER BY section
- Limit results as needed

#### Advanced Analytics
- **Add Dimensions**: Group your data by specific columns
- **Create Measures**: Add aggregations like SUM, COUNT, AVG
- **Filter Aggregates**: Use HAVING clauses to filter grouped results
- **Complex Filters**: Combine multiple conditions with AND/OR logic

### Query Types Supported

#### Row-Level Filters (WHERE)
- Text filters: contains, starts with, ends with
- Numeric comparisons: equals, greater than, between
- Date ranges and comparisons
- NULL checks and list filters (IN/NOT IN)

#### Aggregation Filters (HAVING)
- Filter on calculated values
- Compare aggregated results
- Support for all standard comparison operators

### Performance Tips

- Use appropriate warehouse sizes for your queries
- Add filters to reduce data scanning
- Limit result sets for large tables
- Utilize the query history to track performance

## Architecture

### Native App Benefits
- **Security**: All queries execute within your Snowflake environment
- **Performance**: Direct access to Snowflake's compute resources
- **Governance**: Leverages your existing Snowflake security model
- **Integration**: Seamlessly works with your data and users

### Technical Stack
- **Frontend**: Streamlit for interactive UI
- **Backend**: Snowflake Snowpark for data processing
- **Storage**: Native Snowflake tables for configuration and history
- **Security**: Snowflake's built-in security and access controls

## Configuration

### App Settings
Access configuration through the app's admin interface or SQL:

```sql
-- View current configuration
SELECT * FROM snowprep_app.core.app_config;

-- Update settings
UPDATE snowprep_app.core.app_config 
SET config_value = 5000 
WHERE config_key = 'max_query_limit';
```

### User Preferences
Each user can customize their experience:
- Default query limits
- Preferred databases/schemas
- UI preferences
- Query history settings

## Support & Documentation

### Troubleshooting

**Common Issues:**
- **No databases visible**: Ensure proper grants are in place
- **Query fails**: Check warehouse permissions and SQL syntax
- **Slow performance**: Consider warehouse size and query optimization

### Getting Help
- Check query history for error details
- Review Snowflake's query profile for performance insights
- Consult your Snowflake administrator for privilege issues

## Version History

### v1.0 (Current)
- Initial release
- Core visual query builder functionality
- Basic analytics and filtering capabilities
- Query history and user preferences
- Streamlit-based user interface

## License & Terms

This application is provided under standard Snowflake Native App terms. Please review your Snowflake agreement for specific usage terms and conditions.

## About DataTools Pro

SnowPrep is developed by DataTools Pro, specialists in data analytics and visualization solutions. Visit [datatoolspro.com](https://datatoolspro.com/) for more information about our products and services.

---

**Ready to transform your data exploration experience? Install SnowPrep today and start building queries visually in Snowflake!**