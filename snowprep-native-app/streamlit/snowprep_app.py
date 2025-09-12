# SnowPrep - Snowflake Native App Version
# Visual SQL Builder for Snowflake: Select, Filter, Sort, Group By, Having

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="SnowPrep by DataTools Pro", layout="wide")

# Title with small clickable byline
st.markdown(
    '<div style="display:flex;align-items:baseline;gap:8px;">'
    '<h1 style="margin:0">SnowPrep</h1>'
    '<span style="font-size:0.9rem"><a href="https://datatoolspro.com/" target="_blank">by DataTools Pro</a></span>'
    '</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Snowflake session for Native App
# -----------------------------
@st.cache_resource
def get_session():
    """Get active session for Native App environment"""
    return get_active_session()

session = get_session()

# -----------------------------
# Native App Helper Functions
# -----------------------------
def log_query_to_app_db(db: str, schema: str, table: str, sql: str, rows_returned: int = None):
    """Log query execution to the app's tracking table"""
    try:
        user_name = session.get_current_user()
        session.call(
            "core.log_query",
            user_name, db, schema, table, sql, rows_returned
        )
    except Exception as e:
        # Silent fail - don't break the app if logging fails
        st.warning(f"Query logging failed: {e}")

def get_app_config(key: str, default_value=None):
    """Get configuration value from app config table"""
    try:
        result = session.sql(f"""
            SELECT config_value 
            FROM core.app_config 
            WHERE config_key = '{key}'
        """).collect()
        if result:
            return result[0][0] if result[0][0] is not None else default_value
        return default_value
    except Exception:
        return default_value

# -----------------------------
# Existing Helper Functions (adapted for Native App)
# -----------------------------
def q_ident(name: str) -> str:
    return f'"{name}"'

def escape_literal(val: str) -> str:
    return val.replace("'", "''")

def _normalize_cols(df):
    df.columns = [c.strip().strip('"').lower().replace(" ", "_") for c in df.columns]
    return df

def _safe_rerun():
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        elif hasattr(st, "rerun"):
            st.rerun()
    except Exception:
        return

@st.cache_data
def list_databases() -> list[str]:
    """Return list of accessible databases"""
    try:
        # Use the app's stored procedure to get accessible databases
        result = session.call("core.get_accessible_databases")
        df = result.to_pandas()
        return df["DATABASE_NAME"].tolist()
    except Exception:
        # Fallback to direct query
        df = session.sql("SHOW DATABASES").to_pandas()
        df = _normalize_cols(df)
        return df["name"].tolist()

@st.cache_data
def list_schemas(db: str) -> list[str]:
    df = session.sql(f'SHOW SCHEMAS IN DATABASE "{db}"').to_pandas()
    df = _normalize_cols(df)
    return (df["name"] if "name" in df.columns else df["schema_name"]).tolist()

@st.cache_data
def list_tables(db: str, schema: str) -> list[str]:
    df = session.sql(f'SHOW TABLES IN SCHEMA "{db}"."{schema}"').to_pandas()
    df = _normalize_cols(df)
    return (df["name"] if "name" in df.columns else df["table_name"]).tolist()

@st.cache_data
def list_views(db: str, schema: str) -> list[str]:
    df = session.sql(f'SHOW VIEWS IN SCHEMA "{db}"."{schema}"').to_pandas()
    df = _normalize_cols(df)
    return (df["name"] if "name" in df.columns else df["view_name"]).tolist()

@st.cache_data
def get_columns(db: str, schema: str, table: str) -> pd.DataFrame:
    sql = f"""
    SELECT COLUMN_NAME, DATA_TYPE, ORDINAL_POSITION
    FROM {q_ident(db)}.INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = '{escape_literal(schema.upper())}'
      AND TABLE_NAME   = '{escape_literal(table.upper())}'
    ORDER BY ORDINAL_POSITION
    """
    return session.sql(sql).to_pandas()

def is_numeric_type(dtype: str) -> bool:
    t = dtype.upper()
    return any(kw in t for kw in ["NUMBER", "DECIMAL", "INT", "FLOAT", "DOUBLE", "REAL"])

def is_date_type(dtype: str) -> bool:
    t = dtype.upper()
    return any(kw in t for kw in ["DATE", "TIME", "TIMESTAMP"])

def is_boolean_type(dtype: str) -> bool:
    t = dtype.upper()
    return any(kw in t for kw in ["BOOL", "BOOLEAN", "BIT"])

def classify_dtype(dtype: str) -> str:
    if not dtype:
        return "other"
    t = dtype.upper()
    if is_numeric_type(t):
        return "numeric"
    if is_date_type(t):
        return "date"
    if is_boolean_type(t):
        return "boolean"
    if any(kw in t for kw in ["CHAR", "VARCHAR", "STRING", "TEXT"]):
        return "text"
    return "other"

ICONS = {
    "text": "🔤",
    "numeric": "🔢",
    "date": "📅",
    "boolean": "🔘",
    "other": "❓",
}

def ops_for_dtype(dtype: str) -> list:
    cat = classify_dtype(dtype)
    if cat == "numeric" or cat == "date":
        return ["=", "!=", ">", ">=", "<", "<=", "BETWEEN", "NOT BETWEEN", "IN", "NOT IN", "IS NULL", "IS NOT NULL"]
    if cat == "text":
        return [
            "=", "!=", "CONTAINS", "NOT_CONTAINS", "STARTS_WITH", "NOT_STARTS_WITH",
            "ENDS_WITH", "NOT_ENDS_WITH", "IS_EMPTY", "IS_NOT_EMPTY", 
            "IS NULL", "IS NOT NULL", "IN", "NOT IN",
        ]
    if cat == "boolean":
        return ["IS_TRUE", "IS_FALSE", "IS NULL", "IS NOT NULL"]
    return ["=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "NOT IN", "BETWEEN", "NOT BETWEEN", "IS NULL", "IS NOT NULL"]

def ops_for_agg_target(target: str, agg_rows: list, dtype_map: dict) -> list:
    for a in agg_rows:
        alias = a.get("alias") or f"{a['func'].lower()}_{a['col'].lower()}"
        if alias == target:
            func = a["func"].upper()
            if func == "COUNT":
                return ops_for_dtype("NUMBER")
            if func in ["SUM", "AVG"]:
                return ops_for_dtype("NUMBER")
            if func in ["MIN", "MAX"]:
                return ops_for_dtype(dtype_map.get(a["col"], ""))
    return ["=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "BETWEEN", "IS NULL", "IS NOT NULL"]

# Show app info in sidebar
with st.sidebar:
    st.info("🎯 **SnowPrep Native App**\n\nRunning inside Snowflake with enhanced security and performance.")
    
    # Show query history if enabled
    if get_app_config("enable_query_logging", True):
        if st.button("📊 View Query History"):
            try:
                history_df = session.sql("""
                    SELECT * FROM core.recent_queries 
                    WHERE user_name = CURRENT_USER()
                    LIMIT 10
                """).to_pandas()
                if not history_df.empty:
                    st.dataframe(history_df)
                else:
                    st.info("No recent queries found")
            except Exception as e:
                st.error(f"Could not load query history: {e}")

# -----------------------------
# Main UI (using existing logic from original app)
# -----------------------------

# Add all the existing UI code from the original datatools_streamlit.py here
# This includes the sidebar source selection, query configuration, etc.
# For brevity, I'll include key sections and indicate where to copy the rest

# Sidebar: source selection
with st.sidebar:
    st.header("1) Select dataset")
    dbs = list_databases()
    if not dbs:
        st.error("No databases visible for this role.")
        st.stop()

    db = st.selectbox("Database", dbs, index=min(1, len(dbs)-1) if len(dbs) > 1 else 0)
    schemas = list_schemas(db)
    schema = st.selectbox("Schema", schemas)
    
    object_type = st.radio("Object type", options=["Table", "View", "Both"], index=0, horizontal=True)
    tables, views = [], []
    if object_type in ["Table", "Both"]:
        tables = list_tables(db, schema)
    if object_type in ["View", "Both"]:
        views = list_views(db, schema)

    objects = []
    if tables:
        objects.extend([("table", t) for t in tables])
    if views:
        objects.extend([("view", v) for v in views])

    if not objects:
        st.warning(f"No {object_type.lower()}s in this schema.")
        st.stop()

    display_names = [f"{typ.upper()}: {name}" for typ, name in objects]
    sel_idx = st.selectbox("Object", display_names)
    sel_type, sel_name = objects[display_names.index(sel_idx)]
    table = sel_name

    cols_df = get_columns(db, schema, table)
    all_cols = cols_df["COLUMN_NAME"].tolist()
    dtype_map = dict(zip(cols_df["COLUMN_NAME"], cols_df["DATA_TYPE"]))

# [INSERT ALL THE EXISTING UI CODE FROM ORIGINAL APP HERE]
# This includes:
# - Query configuration section
# - Filters, Dimensions, Measures, Aggregations
# - SQL generation and execution
# The code is too long to include in full, but would be identical to the original

# For the query execution section, add the logging:
if st.button("▶️ Run query", type="primary"):
    with st.spinner("Running..."):
        try:
            # Get max query limit from app config
            max_limit = get_app_config("max_query_limit", 10000)
            if limit > max_limit:
                st.warning(f"Query limit reduced to maximum allowed: {max_limit}")
                limit = max_limit
                
            result_df = session.sql(sql_text).to_pandas()
            rows_returned = len(result_df)
            
            st.success(f"Returned {rows_returned:,} rows")
            st.dataframe(result_df, use_container_width=True)
            
            # Log the query execution
            if get_app_config("enable_query_logging", True):
                log_query_to_app_db(db, schema, table, sql_text, rows_returned)
            
            # Quick chart suggestion
            if group_cols and len(agg_rows) == 1:
                try:
                    x = group_cols[0]
                    y = agg_rows[0]["alias"]
                    if x in result_df.columns and y in result_df.columns:
                        st.bar_chart(result_df.set_index(x)[y])
                except Exception:
                    pass
                    
        except Exception as e:
            st.error(f"Query failed: {e}")
            # Log failed queries too
            if get_app_config("enable_query_logging", True):
                log_query_to_app_db(db, schema, table, sql_text, 0)