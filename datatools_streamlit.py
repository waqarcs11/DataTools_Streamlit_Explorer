# datatools_streamlit.py
# Visual SQL Builder for Snowflake: Select, Filter, Sort, Group By, Having

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional

st.set_page_config(page_title="DataTools: Visual SQL Builder", layout="wide")
st.title("Visual SQL Builder for Snowflake.")

# -----------------------------
# Snowflake session bootstrap
# -----------------------------
def get_session():
    try:
        # Running inside Streamlit in Snowflake
        from snowflake.snowpark.context import get_active_session
        return get_active_session()
    except Exception:
        # Fallback: build a session from Streamlit secrets (for Streamlit Community Cloud or local)
        try:
            from snowflake.snowpark import Session
            cfg = st.secrets["snowflake"]
            return Session.builder.configs(cfg).create()
        except Exception as e:
            st.error(
                "No Snowflake session available. If you are not running in Snowflake, "
                "please add your Snowflake credentials to `st.secrets['snowflake']` "
                "(account, user, password, role, warehouse, database, schema)."
            )
            st.stop()

session = get_session()

# -----------------------------
# Helpers
# -----------------------------
def q_ident(name: str) -> str:
    # Quote identifiers safely
    return f'"{name}"'

def escape_literal(val: str) -> str:
    # Minimal escaping for SQL string literals
    return val.replace("'", "''")

def _normalize_cols(df):
    df.columns = [c.strip().strip('"').lower().replace(" ", "_") for c in df.columns]
    return df

def list_databases() -> list[str]:
    df = session.sql("SHOW DATABASES").to_pandas()
    df = _normalize_cols(df)
    return df["name"].tolist()  # now it exists

def list_schemas(db: str) -> list[str]:
    df = session.sql(f'SHOW SCHEMAS IN DATABASE "{db}"').to_pandas()
    df = _normalize_cols(df)
    # Snowflake sometimes returns schema_name; handle both
    return (df["name"] if "name" in df.columns else df["schema_name"]).tolist()

def list_tables(db: str, schema: str) -> list[str]:
    df = session.sql(f'SHOW TABLES IN SCHEMA "{db}"."{schema}"').to_pandas()
    df = _normalize_cols(df)
    return (df["name"] if "name" in df.columns else df["table_name"]).tolist()

def list_views(db: str, schema: str) -> list[str]:
    df = session.sql(f'SHOW VIEWS IN SCHEMA "{db}"."{schema}"').to_pandas()
    df = _normalize_cols(df)
    return (df["name"] if "name" in df.columns else df["view_name"]).tolist()

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

# -----------------------------
# Sidebar: source selection
# -----------------------------
with st.sidebar:
    st.header("1) Select dataset")
    dbs = list_databases()
    if not dbs:
        st.error("No databases visible for this role.")
        st.stop()

    db = st.selectbox("Database", dbs, index=min(1, len(dbs)-1))
    schemas = list_schemas(db)
    schema = st.selectbox("Schema", schemas)
    # Allow choosing between tables and views
    object_type = st.radio("Object type", options=["Table", "View", "Both"], index=0, horizontal=True)
    tables = []
    views = []
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

    # Present combined list but show type label
    display_names = [f"{typ.upper()}: {name}" for typ, name in objects]
    sel_idx = st.selectbox("Object", display_names)
    sel_type, sel_name = objects[display_names.index(sel_idx)]
    table = sel_name

    cols_df = get_columns(db, schema, table)
    all_cols = cols_df["COLUMN_NAME"].tolist()
    dtype_map = dict(zip(cols_df["COLUMN_NAME"], cols_df["DATA_TYPE"]))

# -----------------------------
# UI state containers
# -----------------------------
st.header("2) Configure query")

with st.expander("Select columns", expanded=True):
    select_cols = st.multiselect("Pick columns to SELECT", options=all_cols, default=all_cols)

with st.expander("Aggregations (optional)"):
    st.caption("Add aggregate measures. If you add any, you can also Group By and use Having.")
    agg_rows = st.session_state.get("agg_rows", [])
    if "agg_rows" not in st.session_state:
        st.session_state.agg_rows = []

    add_agg = st.button("➕ Add aggregate")
    if add_agg:
        st.session_state.agg_rows.append({"func": "COUNT", "col": all_cols[0] if all_cols else "", "alias": ""})

    # Render agg rows
    to_remove = []
    for i, row in enumerate(st.session_state.agg_rows):
        c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
        with c1:
            func = st.selectbox(
                f"Function #{i+1}",
                ["COUNT", "SUM", "AVG", "MIN", "MAX"],
                key=f"agg_func_{i}",
                index=["COUNT", "SUM", "AVG", "MIN", "MAX"].index(row["func"]),
            )
        with c2:
            # For SUM/AVG, restrict to numeric columns
            options = all_cols if func in ["COUNT", "MIN", "MAX"] else [c for c in all_cols if is_numeric_type(dtype_map[c])]
            if not options:
                options = all_cols
            col = st.selectbox(f"Column #{i+1}", options, key=f"agg_col_{i}", index=max(0, options.index(row["col"])) if row["col"] in options else 0)
        with c3:
            alias = st.text_input(f"Alias #{i+1}", value=row.get("alias") or f"{func.lower()}_{col.lower()}", key=f"agg_alias_{i}")
        with c4:
            if st.button("🗑️", key=f"agg_del_{i}"):
                to_remove.append(i)
        # persist
        st.session_state.agg_rows[i] = {"func": func, "col": col, "alias": alias}
    # remove marked
    for idx in sorted(to_remove, reverse=True):
        del st.session_state.agg_rows[idx]
    agg_rows = st.session_state.agg_rows

# Implement a custom expander for Filters so we can detect collapsed/open state
# Use a session-state toggle, because Streamlit's built-in expander doesn't expose its state
with st.expander("Filters (WHERE)", expanded=False):
    st.caption("Build row-level filters. For IN, comma-separate values.")
    if "filters" not in st.session_state:
        st.session_state.filters = []
    if st.button("➕ Add filter"):
        st.session_state.filters.append({"col": all_cols[0] if all_cols else "", "op": "=", "val": ""})

    del_idx = []
    for i, f in enumerate(st.session_state.filters):
        c1, c2, c3, c4 = st.columns([2, 1.2, 3, 0.6])
        with c1:
            col = st.selectbox(f"Column #{i+1}", all_cols, key=f"f_col_{i}", index=max(0, all_cols.index(f["col"])) if f["col"] in all_cols else 0)
        with c2:
            dtype = dtype_map.get(col, "")
            ops = ["=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "BETWEEN", "IS NULL", "IS NOT NULL"]
            op = st.selectbox(f"Op #{i+1}", ops, key=f"f_op_{i}", index=ops.index(f["op"]) if f["op"] in ops else 0)
        with c3:
            show_val = op not in ["IS NULL", "IS NOT NULL"]
            val = st.text_input(f"Value #{i+1} ({dtype})", value=f.get("val", ""), key=f"f_val_{i}") if show_val else ""
        with c4:
            if st.button("🗑️", key=f"f_del_{i}"):
                del_idx.append(i)
        st.session_state.filters[i] = {"col": col, "op": op, "val": val}
    for idx in sorted(del_idx, reverse=True):
        del st.session_state.filters[idx]
    filters = st.session_state.filters
    # Place filter mode control below the filter rows for better UX
    if "filter_mode" not in st.session_state:
        st.session_state.filter_mode = "AND"
    st.session_state.filter_mode = st.radio("Combine filters with", options=["AND", "OR"], index=0 if st.session_state.filter_mode == "AND" else 1, horizontal=True)
    filter_mode = st.session_state.filter_mode

with st.expander("Group By", expanded=False):
    group_cols = st.multiselect("Columns to GROUP BY", all_cols, default=[])

with st.expander("Having (on aggregates)", expanded=False):
    st.caption("Apply conditions on aggregate results (e.g., SUM(amount) > 1000).")
    if "having" not in st.session_state:
        st.session_state.having = []
    if st.button("➕ Add having"):
        # Default to first agg if exists
        default_alias = agg_rows[0]["alias"] if agg_rows else ""
        st.session_state.having.append({"target": default_alias, "op": ">", "val": ""})

    del_h = []
    agg_aliases = [a["alias"] for a in agg_rows]
    for i, h in enumerate(st.session_state.having):
        c1, c2, c3, c4 = st.columns([2, 1.2, 3, 0.6])
        with c1:
            target = st.selectbox(f"Aggregate/alias #{i+1}", agg_aliases, key=f"h_target_{i}") if agg_aliases else st.text_input(f"Aggregate expr #{i+1}", key=f"h_target_{i}", value=h.get("target", ""))
        with c2:
            ops = ["=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "BETWEEN"]
            op = st.selectbox(f"Op #{i+1}", ops, key=f"h_op_{i}", index=ops.index(h["op"]) if h["op"] in ops else 0)
        with c3:
            val = st.text_input(f"Value #{i+1}", value=h.get("val", ""), key=f"h_val_{i}")
        with c4:
            if st.button("🗑️", key=f"h_del_{i}"):
                del_h.append(i)
        st.session_state.having[i] = {"target": target if isinstance(target, str) else target, "op": op, "val": val}
    for idx in sorted(del_h, reverse=True):
        del st.session_state.having[idx]
    having = st.session_state.having

with st.expander("Sort", expanded=False):
    sort_cols = st.multiselect("ORDER BY columns", all_cols, default=[])
    sort_dirs: Dict[str, str] = {}
    if sort_cols:
        c = st.columns(len(sort_cols))
        for i, col in enumerate(sort_cols):
            with c[i]:
                sort_dirs[col] = st.radio(f"{col}", options=["ASC", "DESC"], horizontal=True, key=f"sort_{col}")

limit = st.number_input("Limit rows", min_value=1, max_value=100000, value=1000, step=100)

# -----------------------------
# SQL generation
# -----------------------------
def build_sql() -> str:
    select_parts: List[str] = []

    # Plain columns
    if select_cols:
        select_parts.extend([f"{q_ident(col)}" for col in select_cols])

    # Aggregates
    for a in agg_rows:
        func = a["func"].upper()
        col = a["col"]
        alias = a["alias"] or f"{func.lower()}_{col.lower()}"
        select_parts.append(f"{func}({q_ident(col)}) AS {q_ident(alias)}")

    # Fallback to * if nothing chosen
    if not select_parts:
        select_clause = "*"
    else:
        select_clause = ", ".join(select_parts)

    from_clause = f"{q_ident(db)}.{q_ident(schema)}.{q_ident(table)}"

    # WHERE
    where_clauses: List[str] = []
    for f in filters:
        col = q_ident(f["col"])
        op = f["op"].upper()
        val = f["val"]
        if op in ["IS NULL", "IS NOT NULL"]:
            where_clauses.append(f"{col} {op}")
        elif op in ["IN"]:
            items = [v.strip() for v in val.split(",") if v.strip() != ""]
            if not items:
                continue
            # Quote as strings; Snowflake will cast as needed
            items_sql = ", ".join([f"'{escape_literal(x)}'" for x in items])
            where_clauses.append(f"{col} IN ({items_sql})")
        elif op == "BETWEEN":
            parts = [p.strip() for p in val.split(",")]
            if len(parts) == 2:
                where_clauses.append(f"{col} BETWEEN '{escape_literal(parts[0])}' AND '{escape_literal(parts[1])}'")
        else:
            where_clauses.append(f"{col} {op} '{escape_literal(val)}'")

    # Join WHERE clauses using selected filter_mode
    where_sql = ""
    if where_clauses:
        joined = f" {filter_mode} ".join(where_clauses)
        where_sql = f"WHERE {joined}"

    # GROUP BY
    group_sql = ""
    if group_cols:
        group_sql = "GROUP BY " + ", ".join(q_ident(c) for c in group_cols)

    # HAVING
    having_sql = ""
    if having:
        h_parts = []
        for h in having:
            target = h["target"]
            op = h["op"].upper()
            val = h["val"]
            # Allow reference by alias or raw expression
            tgt_sql = q_ident(target) if target in [a["alias"] for a in agg_rows] else target
            if op == "IN":
                items = [v.strip() for v in val.split(",") if v.strip() != ""]
                if items:
                    items_sql = ", ".join([f"'{escape_literal(x)}'" for x in items])
                    h_parts.append(f"{tgt_sql} IN ({items_sql})")
            elif op == "BETWEEN":
                parts = [p.strip() for p in val.split(",")]
                if len(parts) == 2:
                    h_parts.append(f"{tgt_sql} BETWEEN '{escape_literal(parts[0])}' AND '{escape_literal(parts[1])}'")
            else:
                h_parts.append(f"{tgt_sql} {op} '{escape_literal(val)}'")
        if h_parts:
            having_sql = "HAVING " + " AND ".join(h_parts)

    # ORDER BY
    order_sql = ""
    if sort_cols:
        order_sql = "ORDER BY " + ", ".join(f"{q_ident(c)} {sort_dirs.get(c,'ASC')}" for c in sort_cols)

    # LIMIT
    limit_sql = f"LIMIT {int(limit)}"

    sql = f"""
SELECT {select_clause}
FROM {from_clause}
{where_sql}
{group_sql}
{having_sql}
{order_sql}
{limit_sql}
""".strip()
    # Collapse extra spaces/newlines for neatness
    sql = "\n".join([line.rstrip() for line in sql.splitlines() if line.strip() != ""])
    return sql

sql_text = build_sql()

# -----------------------------
# Preview & Run
# -----------------------------
st.subheader("3) Query preview")
st.code(sql_text, language="sql")

# Show a compact preview of the WHERE clause when Filters expander is closed
if filters:
    # Build the preview using the same logic as build_sql but only for WHERE
    preview_clauses = []
    for f in filters:
        col = q_ident(f["col"])
        op = f["op"].upper()
        val = f["val"]
        if op in ["IS NULL", "IS NOT NULL"]:
            preview_clauses.append(f"{col} {op}")
        elif op in ["IN"]:
            items = [v.strip() for v in val.split(",") if v.strip() != ""]
            if not items:
                continue
            items_sql = ", ".join([f"'{escape_literal(x)}'" for x in items])
            preview_clauses.append(f"{col} IN ({items_sql})")
        elif op == "BETWEEN":
            parts = [p.strip() for p in val.split(",")]
            if len(parts) == 2:
                preview_clauses.append(f"{col} BETWEEN '{escape_literal(parts[0])}' AND '{escape_literal(parts[1])}'")
        else:
            preview_clauses.append(f"{col} {op} '{escape_literal(val)}'")

    # if preview_clauses:
    #     where_preview = f"WHERE and  {' ' + filter_mode + ' '.join(['']+preview_clauses).strip()}"
    #     st.info(where_preview)

c1, c2 = st.columns([1, 1])
with c1:
    run = st.button("▶️ Run query", type="primary")
with c2:
    st.caption("Tip: You can copy this SQL and use it in Worksheets too.")

if run:
    with st.spinner("Running..."):
        try:
            result_df = session.sql(sql_text).to_pandas()
            st.success(f"Returned {len(result_df):,} rows")
            st.dataframe(result_df, use_container_width=True)
            # Quick chart suggestion when grouped with single aggregate
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
