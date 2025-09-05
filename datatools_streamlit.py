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
@st.cache_resource
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

def _safe_rerun():
    """Attempt to trigger a Streamlit rerun in a backwards/forwards-compatible way.

    Some Streamlit versions expose `st.experimental_rerun`, others `st.rerun`. Some
    environments may not expose either. Calling this helper will try available APIs
    and otherwise silently continue (preferred to raising AttributeError at runtime).
    """
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        elif hasattr(st, "rerun"):
            st.rerun()
    except Exception:
        # Best-effort: if rerun APIs are unavailable or fail, do nothing.
        return

@st.cache_data
def list_databases() -> list[str]:
    """Return list of databases. Cached to avoid repeated network calls on reruns."""
    df = session.sql("SHOW DATABASES").to_pandas()
    df = _normalize_cols(df)
    return df["name"].tolist()  # now it exists

@st.cache_data
def list_schemas(db: str) -> list[str]:
    """Return list of schemas for a database. Cached by db."""
    df = session.sql(f'SHOW SCHEMAS IN DATABASE "{db}"').to_pandas()
    df = _normalize_cols(df)
    # Snowflake sometimes returns schema_name; handle both
    return (df["name"] if "name" in df.columns else df["schema_name"]).tolist()

@st.cache_data
def list_tables(db: str, schema: str) -> list[str]:
    """Return list of tables for a schema. Cached by db+schema."""
    df = session.sql(f'SHOW TABLES IN SCHEMA "{db}"."{schema}"').to_pandas()
    df = _normalize_cols(df)
    return (df["name"] if "name" in df.columns else df["table_name"]).tolist()

@st.cache_data
def list_views(db: str, schema: str) -> list[str]:
    """Return list of views for a schema. Cached by db+schema."""
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
    """Classify a Snowflake DATA_TYPE into one of: text, numeric, date, boolean, other."""
    if not dtype:
        return "other"
    t = dtype.upper()
    if is_numeric_type(t):
        return "numeric"
    if is_date_type(t):
        return "date"
    if is_boolean_type(t):
        return "boolean"
    # treat common text types as text
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
    """Return appropriate operator list for a given column data type."""
    cat = classify_dtype(dtype)
    if cat == "numeric" or cat == "date":
        return ["=", "!=", ">", ">=", "<", "<=", "BETWEEN", "NOT BETWEEN", "IN", "NOT IN", "IS NULL", "IS NOT NULL"]
    if cat == "text":
        return [
            "=",
            "!=",
            "CONTAINS",
            "NOT_CONTAINS",
            "STARTS_WITH",
            "NOT_STARTS_WITH",
            "ENDS_WITH",
            "NOT_ENDS_WITH",
            "IS_EMPTY",
            "IS_NOT_EMPTY",
            "IS NULL",
            "IS NOT NULL",
            "IN",
            "NOT IN",
        ]
    if cat == "boolean":
        return ["IS_TRUE", "IS_FALSE", "IS NULL", "IS NOT NULL"]
    return ["=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "NOT IN", "BETWEEN", "NOT BETWEEN", "IS NULL", "IS NOT NULL"]

def ops_for_agg_target(target: str, agg_rows: list, dtype_map: dict) -> list:
    """Determine operator list for a HAVING target (alias or expression).
    If target is one of the aggregate aliases, infer type from the underlying column and function.
    """
    # Find matching agg row by alias
    for a in agg_rows:
        alias = a.get("alias") or f"{a['func'].lower()}_{a['col'].lower()}"
        if alias == target:
            func = a["func"].upper()
            col = a["col"]
            # COUNT always numeric
            if func == "COUNT":
                return ops_for_dtype("NUMBER")
            # SUM/AVG numeric
            if func in ["SUM", "AVG"]:
                return ops_for_dtype("NUMBER")
            # MIN/MAX keep column type
            if func in ["MIN", "MAX"]:
                return ops_for_dtype(dtype_map.get(col, ""))
    # Fallback: if we don't know, return a permissive set
    return ["=", "!=", ">", ">=", "<", "<=", "LIKE", "ILIKE", "IN", "BETWEEN", "IS NULL", "IS NOT NULL"]

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
    select_cols = st.multiselect(
        "Pick columns to SELECT",
        options=all_cols,
        default=all_cols,
        format_func=lambda x: f"{ICONS.get(classify_dtype(dtype_map.get(x, '')) , '')} {x}",
    )

# Place Filters expander directly after Select columns per UX request
with st.expander("Filters (WHERE)", expanded=False):
    st.caption("Build row-level filters. For IN, comma-separate values.")
    # Convert Filters to placeholder-driven rows like Aggregations/Dims
    if "filters" not in st.session_state:
        # each filter: {id, col, op, val}
        st.session_state["filters"] = [{"id": st.session_state.get("_filter_next_id", 0), "col": "", "op": "=", "val": ""}]
        st.session_state["_filter_next_id"] = st.session_state.get("_filter_next_id", 0) + 1

    # Render placeholder-driven filters
    to_remove_filters = []
    new_filters = list(st.session_state["filters"])
    for fi, fil in enumerate(list(st.session_state["filters"])):
        fid = fil.get("id", fi)
        if not fil.get("col"):
            # placeholder: only column select (match measure column widths)
            # Render the placeholder select in the leftmost column with a label so it
            # aligns exactly like full filter rows.
            c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
            with c1:
                key = f"f_col_{fid}"
                opts = [""] + all_cols
                if key in st.session_state:
                    col_sel = st.selectbox(f"Column #{fi+1}", opts, key=key, format_func=lambda x: (f"{ICONS.get(classify_dtype(dtype_map.get(x, '')),'')} {x}" if x else ""))
                else:
                    col_sel = st.selectbox(f"Column #{fi+1}", opts, index=0, key=key, format_func=lambda x: (f"{ICONS.get(classify_dtype(dtype_map.get(x, '')),'')} {x}" if x else ""))
            # if selected, convert to full filter and append placeholder
            if col_sel and col_sel != "":
                # find and update
                for rr in st.session_state["filters"]:
                    if rr.get("id") == fid:
                        rr["col"] = col_sel
                        rr["op"] = "="
                        rr["val"] = ""
                nid = st.session_state.get("_filter_next_id", 0)
                st.session_state["_filter_next_id"] = nid + 1
                st.session_state["filters"].append({"id": nid, "col": "", "op": "=", "val": ""})
                st.session_state["filters"] = list(st.session_state["filters"])
                _safe_rerun()
        else:
            # full filter row (match measure column widths)
            c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
            with c1:
                key = f"f_col_{fid}"
                if key in st.session_state:
                    col_sel = st.selectbox(f"Column #{fi+1}", all_cols, key=key, format_func=lambda x: (f"{ICONS.get(classify_dtype(dtype_map.get(x, '')),'')} {x}" if x else ""))
                else:
                    col_sel = st.selectbox(f"Column #{fi+1}", all_cols, index=max(0, all_cols.index(fil.get("col"))) if fil.get("col") in all_cols else 0, key=key, format_func=lambda x: (f"{ICONS.get(classify_dtype(dtype_map.get(x, '')),'')} {x}" if x else ""))
            with c2:
                dtype = dtype_map.get(col_sel, "")
                ops = ops_for_dtype(dtype)
                op_key = f"f_op_{fid}"
                if op_key in st.session_state:
                    op_sel = st.selectbox(f"Op #{fi+1}", ops, key=op_key)
                else:
                    op_sel = st.selectbox(f"Op #{fi+1}", ops, index=ops.index(fil.get("op")) if fil.get("op") in ops else 0, key=op_key)
            with c3:
                val_key = f"f_val_{fid}"
                show_val = op_sel not in ["IS NULL", "IS NOT NULL"]
                if val_key in st.session_state:
                    val_sel = st.text_input(f"Value #{fi+1} ({dtype})", key=val_key)
                else:
                    val_sel = st.text_input(f"Value #{fi+1} ({dtype})", value=fil.get("val", ""), key=val_key) if show_val else ""
            with c4:
                if st.button("❌", key=f"f_del_{fid}"):
                    to_remove_filters.append(fid)
            # persist
            for idx, rr in enumerate(new_filters):
                if rr.get("id") == fid:
                    new_filters[idx] = {"id": fid, "col": col_sel, "op": op_sel, "val": val_sel}
    # apply removals
    if to_remove_filters:
        st.session_state["filters"] = [r for r in new_filters if r.get("id") not in to_remove_filters]
    else:
        st.session_state["filters"] = new_filters
    # Expose only real filters (exclude placeholders) to downstream logic
    filters = [r for r in st.session_state["filters"] if r.get("col")]
    # Place filter mode control below the filter rows for better UX
    if "filter_mode" not in st.session_state:
        st.session_state.filter_mode = "AND"
    st.session_state.filter_mode = st.radio("Combine filters with", options=["AND", "OR"], index=0 if st.session_state.filter_mode == "AND" else 1, horizontal=True)
    filter_mode = st.session_state.filter_mode

with st.expander("Aggregations (optional)"):
    # -----------------------------
    # Dimensions (new): behave like measures - show a placeholder select, convert to full
    # row on selection and append another placeholder. Dimensions will be used in the
    # SELECT/GROUP BY when present (and will replace explicit "Select columns").
    # -----------------------------
    st.subheader("Dimensions")
    if "dims" not in st.session_state:
        # each dim is {'id': int, 'col': ''}
        st.session_state["dims"] = [{"id": st.session_state.get("_agg_next_id", 0), "col": ""}]
        st.session_state["_agg_next_id"] = st.session_state.get("_agg_next_id", 0) + 1

    # render dims (placeholder-driven)
    dim_to_remove = []
    dims = list(st.session_state["dims"])
    # Ensure selectbox session_state keys are synchronized with the dims list BEFORE rendering
    for dd in dims:
        rid_sync = dd.get("id")
        key_sync = f"dim_col_{rid_sync}"
        # initialize the widget state to match the dims entry (empty string for placeholder)
        if key_sync not in st.session_state or st.session_state.get(key_sync) != (dd.get("col") or ""):
            st.session_state[key_sync] = dd.get("col") or ""
    for di, d in enumerate(dims):
        rid = d.get("id", di)
        if not d.get("col"):
            # placeholder: show only the column select (keep same column width as agg rows)
            c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
            with c1:
                key = f"dim_col_{rid}"
                opts = [""] + all_cols
                if key in st.session_state:
                    sel = st.selectbox("Column:", opts, key=key, format_func=lambda x: (f"{ICONS.get(classify_dtype(dtype_map.get(x, '')) , '')} {x}" if x else ""))
                else:
                    sel = st.selectbox("Column:", opts, index=0, key=key, format_func=lambda x: (f"{ICONS.get(classify_dtype(dtype_map.get(x, '')) , '')} {x}" if x else ""))
            # if user selected, record the selection to convert after rendering loop
            if sel and sel != "":
                # Defer conversion to avoid widget key/state races; store intent in session_state
                st.session_state[f"_dim_set_{rid}"] = sel
        else:
            # full dim row: column + delete
            c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
            with c1:
                key = f"dim_col_{rid}"
                if key in st.session_state:
                    col = st.selectbox("Column:", all_cols, key=key, format_func=lambda x: f"{ICONS.get(classify_dtype(dtype_map.get(x, '')) , '')} {x}")
                else:
                    # preselect current value
                    default_idx = max(0, all_cols.index(d.get("col"))) if d.get("col") in all_cols else 0
                    col = st.selectbox("Column:", all_cols, index=default_idx, key=key, format_func=lambda x: f"{ICONS.get(classify_dtype(dtype_map.get(x, '')) , '')} {x}")
            with c4:
                if st.button("❌", key=f"dim_del_{rid}"):
                    dim_to_remove.append(rid)

    # apply dim removals
    if dim_to_remove:
        st.session_state["dims"] = [r for r in st.session_state["dims"] if r.get("id") not in dim_to_remove]

    # Process any deferred dim selections recorded during rendering
    processed = False
    for k in list(st.session_state.keys()):
        if k.startswith("_dim_set_"):
            try:
                rid = int(k.split("_dim_set_")[1])
            except Exception:
                continue
            sel_val = st.session_state.pop(k)
            # apply the selection to the dims list
            updated = False
            for rr in st.session_state["dims"]:
                if rr.get("id") == rid:
                    rr["col"] = sel_val
                    updated = True
                    break
            if updated:
                nid = st.session_state.get("_agg_next_id", 0)
                st.session_state["_agg_next_id"] = nid + 1
                st.session_state["dims"].append({"id": nid, "col": ""})
                processed = True
    if processed:
        # Persist and rerun once after applying all deferred conversions
        st.session_state["dims"] = list(st.session_state["dims"])
        # Sync selectbox session_state keys to the dims list so widgets show correct values
        for rr in st.session_state["dims"]:
            key = f"dim_col_{rr.get('id')}"
            # ensure a key exists for each dim (placeholder gets empty string)
            st.session_state[key] = rr.get("col") or ""
        _safe_rerun()

    # Debug helper: show dim-related session_state for troubleshooting (toggle)
    try:
        if st.checkbox("Show dims debug", key="_show_dims_debug"):
            dbg = {k: v for k, v in st.session_state.items() if k.startswith("dim_col_") or k.startswith("_dim_set_") or k == "dims"}
            st.json(dbg)
    except Exception:
        pass

    # initialize agg_rows if not present - start with one placeholder that has empty col
    if "agg_rows" not in st.session_state:
        st.session_state.agg_rows = [{"func": "COUNT", "col": "", "alias": ""}]
    # ensure we have a per-row id generator to reliably identify rows across reruns
    if "_agg_next_id" not in st.session_state:
        st.session_state["_agg_next_id"] = 0
    # backfill missing ids for any pre-existing rows (for older saved state)
    updated = False
    rows_with_ids = []
    for r in st.session_state.agg_rows:
        if "id" not in r:
            r_id = st.session_state["_agg_next_id"]
            st.session_state["_agg_next_id"] += 1
            r["id"] = r_id
            updated = True
        rows_with_ids.append(r)
    if updated:
        st.session_state.agg_rows = rows_with_ids

    # Render agg rows. Rows with empty 'col' are placeholders showing only a column select.
    # First, convert any placeholder selects that already have a selection stored in session_state
    new_rows = list(st.session_state.agg_rows)
    for i, row in enumerate(list(st.session_state.agg_rows)):
        if row.get("col", "") == "":
            rid = row.get("id", i)
            sel_key = f"agg_col_{rid}"
            if sel_key in st.session_state:
                sel_val = st.session_state.get(sel_key, "")
                if sel_val and sel_val != "":
                    func = "COUNT"
                    # assign a stable id for this converted row
                    r_id = st.session_state["_agg_next_id"]
                    st.session_state["_agg_next_id"] += 1
                    new_rows[i] = {"id": r_id, "func": func, "col": sel_val, "alias": f"{func.lower()}_{sel_val.lower()}"}
                    # append a new placeholder with its own id
                    ph_id = st.session_state["_agg_next_id"]
                    st.session_state["_agg_next_id"] += 1
                    new_rows.append({"id": ph_id, "func": "COUNT", "col": "", "alias": ""})

    st.session_state.agg_rows = new_rows

    # Now render rows (after conversion). Delete operations are applied immediately to avoid index-shift issues.
    new_rows = list(st.session_state.agg_rows)
    # Measures header
    st.subheader("Measures")
    for i, row in enumerate(st.session_state.agg_rows):
        if row.get("col", "") == "":
            c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
            with c1:
                opts = [""] + all_cols
                rid = row.get("id", i)
                col_key = f"agg_col_{rid}"
                # If the widget already has state, avoid forcing index which would override user choice
                if col_key in st.session_state:
                    sel = st.selectbox(
                        f"Column #{i+1}",
                        opts,
                        key=col_key,
                        format_func=lambda x: (f"{ICONS.get(classify_dtype(dtype_map.get(x, '')) , '')} {x}" if x else ""),
                    )
                else:
                    sel = st.selectbox(
                        f"Column #{i+1}",
                        opts,
                        index=0,
                        key=col_key,
                        format_func=lambda x: (f"{ICONS.get(classify_dtype(dtype_map.get(x, '')) , '')} {x}" if x else ""),
                    )
        else:
            # Layout: Column | Function | Alias | Delete
            c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
            with c1:
                options = all_cols if row.get("func", "COUNT") in ["COUNT", "MIN", "MAX"] else [c for c in all_cols if is_numeric_type(dtype_map[c])]
                if not options:
                    options = all_cols
                rid = row.get("id", i)
                col_key = f"agg_col_{rid}"
                if col_key in st.session_state:
                    col = st.selectbox(
                        f"Column #{i+1}",
                        options,
                        key=col_key,
                        format_func=lambda x: f"{ICONS.get(classify_dtype(dtype_map.get(x, '')) , '')} {x}",
                    )
                else:
                    col = st.selectbox(
                        f"Column #{i+1}",
                        options,
                        index=max(0, options.index(row["col"])) if row["col"] in options else 0,
                        key=col_key,
                        format_func=lambda x: f"{ICONS.get(classify_dtype(dtype_map.get(x, '')) , '')} {x}",
                    )
            with c2:
                rid = row.get("id", i)
                func_key = f"agg_func_{rid}"
                func_options = ["COUNT", "SUM", "AVG", "MIN", "MAX"]
                if func_key in st.session_state:
                    func = st.selectbox(
                        f"Function #{i+1}",
                        func_options,
                        key=func_key,
                    )
                else:
                    func = st.selectbox(
                        f"Function #{i+1}",
                        func_options,
                        index=func_options.index(row.get("func", "COUNT")),
                        key=func_key,
                    )
            with c3:
                rid = row.get("id", i)
                alias_key = f"agg_alias_{rid}"
                # Auto-update alias logic must run BEFORE the widget is created to avoid Streamlit errors
                func_key_local = f"agg_func_{rid}"
                col_key_local = f"agg_col_{rid}"
                cur_func_local = st.session_state.get(func_key_local, row.get("func"))
                cur_col_local = st.session_state.get(col_key_local, row.get("col"))
                prev_default_local = f"{(row.get('func') or '').lower()}_{(row.get('col') or '').lower()}" if row.get('col') else ""
                new_default_local = f"{cur_func_local.lower()}_{cur_col_local.lower()}" if cur_col_local else ""
                if alias_key not in st.session_state:
                    st.session_state[alias_key] = row.get("alias") or new_default_local
                else:
                    cur_val_local = st.session_state.get(alias_key, "")
                    if cur_val_local == prev_default_local or cur_val_local == "":
                        st.session_state[alias_key] = new_default_local
                alias_widget_val = st.text_input(f"Alias #{i+1}", key=alias_key)
            with c4:
                # use stable id-based keys for delete buttons so clicks map to rows reliably
                btn_key = f"agg_del_{row.get('id', i)}"
                if st.button("❌", key=btn_key):
                    # defer deletion: record the id to delete and handle after the widget loop
                    st.session_state["_agg_delete_id"] = row.get("id", i)
            # After rendering widgets, read current widget values from session_state to avoid races
            rid = row.get("id", i)
            func_key = f"agg_func_{rid}"
            col_key = f"agg_col_{rid}"
            alias_key = f"agg_alias_{rid}"
            cur_func = st.session_state.get(func_key, row.get("func"))
            cur_col = st.session_state.get(col_key, row.get("col"))
            cur_alias = st.session_state.get(alias_key, row.get("alias") or "")
            # Do not assign to st.session_state[alias_key] here (widget already instantiated); we already set before widget
            new_rows[i] = {"id": row.get("id"), "func": cur_func, "col": cur_col, "alias": cur_alias}

    # If a delete was requested via session_state, apply it now and trigger a rerun.
    if "_agg_delete_id" in st.session_state:
        del_id = st.session_state.pop("_agg_delete_id")
        rows = list(new_rows)
        # find by id
        del_idx = next((idx for idx, r in enumerate(rows) if r.get("id") == del_id), None)
        if del_idx is not None:
            del rows[del_idx]
        new_rows = rows
        st.session_state.agg_rows = new_rows
        _safe_rerun()

    # ensure a placeholder exists at the end (with a stable id)
    if not new_rows or new_rows[-1].get("col", "") != "":
        ph_id = st.session_state.get("_agg_next_id", 0)
        st.session_state["_agg_next_id"] = ph_id + 1
        new_rows.append({"id": ph_id, "func": "COUNT", "col": "", "alias": ""})

    st.session_state.agg_rows = new_rows
    # Use a cleaned view of agg_rows (exclude placeholders with empty 'col') for downstream logic
    agg_rows = [r for r in st.session_state.agg_rows if r.get("col")]
    # Build dimensions list (exclude placeholders). When present, dimensions will be used
    # in the SELECT and GROUP BY and will replace the main `select_cols` selection.
    dims = [d["col"] for d in st.session_state.get("dims", []) if d.get("col")]

# -----------------------------
# Filter Aggregates (previously Having)
# -----------------------------
with st.expander("Filter Aggregates", expanded=False):
    st.caption("Apply conditions on aggregate results (e.g., SUM(amount) > 1000).")
    # placeholder-driven having rows (no Add button)
    if "having" not in st.session_state:
        st.session_state.having = [{"id": st.session_state.get("_agg_next_id", 0), "target": "", "op": ">", "val": ""}]
        st.session_state["_agg_next_id"] = st.session_state.get("_agg_next_id", 0) + 1

    # Build alias list and map alias -> underlying column for icon inference
    agg_aliases = []
    alias_to_col = {}
    for a in agg_rows:
        alias = a.get("alias") or f"{a['func'].lower()}_{a['col'].lower()}"
        agg_aliases.append(alias)
        alias_to_col[alias] = a["col"]

    new_having = list(st.session_state.having)
    to_remove_h = []
    for hi, h in enumerate(list(st.session_state.having)):
        hid = h.get("id", hi)
        # placeholder: empty target select
        if not h.get("target"):
            c1, c2, c3, c4 = st.columns([2, 1.2, 3, 0.6])
            with c1:
                key = f"h_target_{hid}"
                opts = [""] + agg_aliases if agg_aliases else [""]
                if key in st.session_state:
                    tgt = st.selectbox(f"Aggregate/alias #{hi+1}", opts, key=key, format_func=lambda al: (f"{ICONS.get(classify_dtype(dtype_map.get(alias_to_col.get(al, ''), '')), '')} {al}" if al else ""))
                else:
                    tgt = st.selectbox(f"Aggregate/alias #{hi+1}", opts, index=0, key=key, format_func=lambda al: (f"{ICONS.get(classify_dtype(dtype_map.get(alias_to_col.get(al, ''), '')), '')} {al}" if al else ""))
            if tgt and tgt != "":
                # convert placeholder into full having row and append new placeholder
                for rr in st.session_state.having:
                    if rr.get("id") == hid:
                        rr["target"] = tgt
                        rr["op"] = ">"
                        rr["val"] = ""
                nid = st.session_state.get("_agg_next_id", 0)
                st.session_state["_agg_next_id"] = nid + 1
                st.session_state.having.append({"id": nid, "target": "", "op": ">", "val": ""})
                st.session_state.having = list(st.session_state.having)
                _safe_rerun()
        else:
            # full having row: target (alias), op, val, del
            c1, c2, c3, c4 = st.columns([2, 1.2, 3, 0.6])
            with c1:
                key = f"h_target_{hid}"
                if agg_aliases:
                    if key in st.session_state:
                        target = st.selectbox(f"Aggregate/alias #{hi+1}", agg_aliases, key=key, format_func=lambda al: f"{ICONS.get(classify_dtype(dtype_map.get(alias_to_col.get(al, ''), '')), '')} {al}")
                    else:
                        default_idx = agg_aliases.index(h.get("target")) if h.get("target") in agg_aliases else 0
                        target = st.selectbox(f"Aggregate/alias #{hi+1}", agg_aliases, index=default_idx, key=key, format_func=lambda al: f"{ICONS.get(classify_dtype(dtype_map.get(alias_to_col.get(al, ''), '')), '')} {al}")
                else:
                    target = st.text_input(f"Aggregate expr #{hi+1}", key=key, value=h.get("target", ""))
            with c2:
                ops = ops_for_agg_target(target if isinstance(target, str) else h.get("target", ""), agg_rows, dtype_map)
                op_key = f"h_op_{hid}"
                if op_key in st.session_state:
                    op = st.selectbox(f"Op #{hi+1}", ops, key=op_key)
                else:
                    op_index = ops.index(h.get("op")) if h.get("op") in ops else 0
                    op = st.selectbox(f"Op #{hi+1}", ops, index=op_index, key=op_key)
            with c3:
                val_key = f"h_val_{hid}"
                if val_key in st.session_state:
                    val = st.text_input(f"Value #{hi+1}", key=val_key)
                else:
                    val = st.text_input(f"Value #{hi+1}", value=h.get("val", ""), key=val_key)
            with c4:
                if st.button("❌", key=f"h_del_{hid}"):
                    to_remove_h.append(hid)
            # persist
            for idx, rr in enumerate(new_having):
                if rr.get("id") == hid:
                    new_having[idx] = {"id": hid, "target": target if isinstance(target, str) else target, "op": op, "val": val}

    # apply removals
    if to_remove_h:
        st.session_state.having = [r for r in new_having if r.get("id") not in to_remove_h]
    else:
        st.session_state.having = new_having

    # expose cleaned having list to downstream code (exclude placeholders)
    having = [r for r in st.session_state.having if r.get("target")]

# (Filters expander was moved earlier to immediately follow Select columns)

with st.expander("Group By", expanded=False):
    group_cols = st.multiselect("Columns to GROUP BY", all_cols, default=[])

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
    # If dimensions are present, include them first and ignore the Select columns multiselect
    if dims:
        select_parts.extend([f"{q_ident(col)}" for col in dims])
    else:
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
        # NULL checks
        if op in ["IS NULL", "IS NOT NULL"]:
            where_clauses.append(f"{col} {op}")
        # Boolean shortcuts
        elif op == "IS_TRUE":
            where_clauses.append(f"{col} = TRUE")
        elif op == "IS_FALSE":
            where_clauses.append(f"{col} = FALSE")
        # IN / NOT IN
        elif op in ["IN", "NOT IN"]:
            items = [v.strip() for v in val.split(",") if v.strip() != ""]
            if not items:
                continue
            items_sql = ", ".join([f"'{escape_literal(x)}'" for x in items])
            where_clauses.append(f"{col} {op} ({items_sql})")
        # BETWEEN / NOT BETWEEN
        elif op in ["BETWEEN", "NOT BETWEEN"]:
            parts = [p.strip() for p in val.split(",")]
            if len(parts) == 2:
                where_clauses.append(f"{col} {op} '{escape_literal(parts[0])}' AND '{escape_literal(parts[1])}'")
        # Text helpers: contains / starts/ends / empty checks
        elif op in ["CONTAINS", "NOT_CONTAINS"]:
            if val.strip() == "":
                continue
            sql_op = "ILIKE" if op == "CONTAINS" else "NOT ILIKE"
            where_clauses.append(f"{col} {sql_op} '%{escape_literal(val)}%'")
        elif op in ["STARTS_WITH", "NOT_STARTS_WITH"]:
            if val.strip() == "":
                continue
            sql_op = "ILIKE" if op == "STARTS_WITH" else "NOT ILIKE"
            where_clauses.append(f"{col} {sql_op} '{escape_literal(val)}%'")
        elif op in ["ENDS_WITH", "NOT_ENDS_WITH"]:
            if val.strip() == "":
                continue
            sql_op = "ILIKE" if op == "ENDS_WITH" else "NOT ILIKE"
            where_clauses.append(f"{col} {sql_op} '%{escape_literal(val)}'")
        elif op == "IS_EMPTY":
            where_clauses.append(f"{col} = ''")
        elif op == "IS_NOT_EMPTY":
            where_clauses.append(f"{col} <> ''")
        else:
            # Fallback to binary operator (=, !=, >, <, etc.)
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
