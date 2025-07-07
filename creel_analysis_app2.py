import pandas as pd
import streamlit as st
import os
import plotly.express as px
import datetime
import calendar

# --- Set Pandas Styler max_elements option to handle large DataFrames ---
pd.set_option("styler.render.max_elements", 10_000_000)

# --- Configuration and File Paths ---
# IMPORTANT: All data files are expected to be in the SAME directory as the script.
# This means your 'CREEL3YRS-MOD.csv' and 'Creel_arabic_quality.csv'
# should be directly alongside your 'your_script_name.py' in your GitHub repository.

FILE1_PATH = "CREEL3YRS-MOD.csv"
FILE2_PATH = "Creel_arabic_quality.csv"

# Initial loading for immediate use (e.g., for st.sidebar before @st.cache_data)
# These will be reloaded by the cached function later.
try:
    df_creel_initial = pd.read_csv(FILE1_PATH) # Renamed to avoid conflict with function's df_creel
    df_arabic_quality_initial = pd.read_csv(FILE2_PATH) # Renamed
except FileNotFoundError:
    st.error(f"Initial file load failed. Make sure '{FILE1_PATH}' and '{FILE2_PATH}' exist in the same directory as the script.")
    df_creel_initial = pd.DataFrame() # Provide empty DFs to prevent errors
    df_arabic_quality_initial = pd.DataFrame()


# Define the exact column names for CREEL3YRS-MOD.csv
CREEL_COLUMNS = ['Sales Organizations Name', 'Billing Year', 'Region', 'Customer', 'Creel', 'Value EGP', 'Value USD', 'Quantity SQM']

# --- Monthly Sales Percentages for Forecasting (FIXED as per your rule) ---
INTERNATIONAL_MONTHLY_PERCENTAGES = {
    1: 0.09, 2: 0.07, 3: 0.06, 4: 0.07, 5: 0.07, 6: 0.06,
    7: 0.10, 8: 0.08, 9: 0.08, 10: 0.09, 11: 0.11, 12: 0.12
}

EGYPT_MONTHLY_PERCENTAGES = {
    1: 0.08, 2: 0.07, 3: 0.09, 4: 0.08, 5: 0.10, 6: 0.08,
    7: 0.09, 8: 0.08, 9: 0.07, 10: 0.09, 11: 0.08, 12: 0.08
}

# --- Helper Function: Detect Arabic Characters ---
def contains_arabic(text):
    """Checks if a string contains any Arabic characters."""
    if not isinstance(text, str):
        return False
    for char in text:
        if ('\u0600' <= char <= '\u06FF' or
            '\u0750' <= char <= '\u077F' or
            '\uFB50' <= char <= '\uFDFF' or
            '\uFE70' <= char <= '\uFEFF'):
            return True
    return False

# --- Data Loading and Preprocessing (Cached for Performance) ---
@st.cache_data
def load_and_preprocess_data():
    """
    Loads and preprocesses the sales and quality data.
    Applies data type conversions, strips whitespace, classifies sectors,
    and maps quality information.
    Returns a DataFrame if successful, or an empty DataFrame with expected columns on failure.
    """
    try:
        with st.spinner("Loading and processing sales data..."):
            df_creel = pd.read_csv(FILE1_PATH, sep=',',
                                   encoding='utf-8-sig',
                                   skiprows=1,
                                   names=CREEL_COLUMNS,
                                   header=None,
                                   on_bad_lines='skip',
                                   low_memory=False)

        df_creel['Billing Year'] = pd.to_numeric(df_creel['Billing Year'], errors='coerce').astype('Int64')
        df_creel['Value EGP'] = pd.to_numeric(df_creel['Value EGP'], errors='coerce').astype('Float64')
        df_creel['Value USD'] = pd.to_numeric(df_creel['Value USD'], errors='coerce').astype('Float64')
        df_creel['Quantity SQM'] = pd.to_numeric(df_creel['Quantity SQM'], errors='coerce').astype('Float64')

        # Filter out rows with non-positive quantities/values
        df_creel = df_creel[
            (df_creel['Quantity SQM'] > 0) &
            (df_creel['Value EGP'] > 0) &
            (df_creel['Value USD'].fillna(0) >= 0) # Allow zero USD if EGP is positive
        ].copy()

        for col in ['Sales Organizations Name', 'Region', 'Customer', 'Creel']:
            df_creel[col] = df_creel[col].astype(str).str.strip()

        df_creel['Is_Egypt_Sector'] = df_creel.apply(
            lambda row: contains_arabic(row['Region']) or contains_arabic(row['Customer']),
            axis=1
        )

        df_quality_map = pd.read_csv(FILE2_PATH, encoding='utf-8-sig')
        df_quality_map.rename(columns={'Creel': 'Creel'}, inplace=True)
        df_quality_map.columns = df_quality_map.columns.str.strip()
        df_quality_map['Quality'] = df_quality_map['Quality'].astype(str).str.strip()

        quality_map_series = df_quality_map.drop_duplicates(subset=['Creel']).set_index('Creel')['Quality']

        merged_df = df_creel.copy()
        merged_df['Quality'] = merged_df['Creel'].map(quality_map_series)
        
        merged_df['Sector'] = merged_df['Is_Egypt_Sector'].apply(lambda x: 'Egypt' if x else 'International')

        return merged_df

    except FileNotFoundError:
        st.error(f"Error: One or both files not found. Check paths.")
        st.info(f"Path 1: `{FILE1_PATH}`")
        st.info(f"Path 2: `{FILE2_PATH}`")
        return pd.DataFrame(columns=CREEL_COLUMNS + ['Quality', 'Is_Egypt_Sector', 'Sector'])
    except Exception as e:
        st.error(f"An error occurred during data loading: {e}")
        return pd.DataFrame(columns=CREEL_COLUMNS + ['Quality', 'Is_Egypt_Sector', 'Sector'])

# --- Helper Function: Format Numeric Values ---
def format_value(value, value_type):
    """
    Helper function to format numeric values with currency/units,
    applying specific rounding for large KPI values.
    """
    if pd.isna(value):
        return "N/A"
    
    # Ensure value is float for calculations
    value = float(value)

    if value_type == 'Value EGP':
        if abs(value) >= 1_000_000_000: # Billions
            return f"EGP {value / 1_000_000_000:,.1f} Billion"
        elif abs(value) >= 1_000_000: # Millions
            return f"EGP {value / 1_000_000:,.1f} Million"
        elif abs(value) >= 1_000: # Thousands
            return f"EGP {value / 1_000:,.1f} K"
        else:
            return f"EGP {value:,.0f}"
    elif value_type == 'Value USD':
        if abs(value) >= 1_000_000_000: # Billions
            return f"USD {value / 1_000_000_000:,.1f} Billion"
        elif abs(value) >= 1_000_000: # Millions
            return f"USD {value / 1_000_000:,.1f} Million"
        elif abs(value) >= 1_000: # Thousands
            return f"USD {value / 1_000:,.1f} K"
        else:
            return f"USD {value:,.0f}"
    elif value_type == 'Quantity SQM':
        if abs(value) >= 1_000_000_000: # Billions
            return f"{value / 1_000_000_000:,.1f} Billion SQM"
        elif abs(value) >= 1_000_000: # Millions
            return f"{value / 1_000_000:,.1f} Million SQM"
        elif abs(value) >= 1_000: # Thousands
            return f"{value / 1_000:,.1f} K SQM"
        else:
            return f"{value:,.0f} SQM"
    elif value_type == 'Percentage':
        return f"{value:,.2f}%"
    else:
        return f"{value:,.0f}"

# --- Helper Function: Display Top/Least N Items ---
def display_top_n(dataframe, group_col, value_col, title, num_items=10, ascending=False, show_chart=False):
    """
    Groups the dataframe by a specified column, sums a value column,
    and displays the top/least N items as both a Plotly bar chart (optional)
    and a Streamlit DataFrame with 1-based ranking.
    """
    st.subheader(title)
    if dataframe.empty:
        st.info("No data available.")
        return

    df_clean_group = dataframe.dropna(subset=[group_col])
    if df_clean_group[group_col].dtype == object and 'nan' in df_clean_group[group_col].unique():
        df_clean_group = df_clean_group[df_clean_group[group_col] != 'nan']

    if df_clean_group.empty:
        st.info(f"No valid data in '{group_col}'.")
        return

    summary = df_clean_group.dropna(subset=[value_col]).groupby(group_col)[value_col].sum().sort_values(ascending=ascending)
    
    if not summary.empty:
        chart_df = summary.head(num_items).reset_index()

        if show_chart:
            fig = px.bar(chart_df,
                         x=group_col,
                         y=value_col,
                         title=f"{title} Chart",
                         labels={group_col: group_col.replace('_', ' ').title(),
                                 value_col: f"Total {value_col.replace("_", " ")}"},
                         color=value_col,
                         color_continuous_scale=px.colors.sequential.Viridis if ascending else px.colors.sequential.Plasma_r
                         )
            fig.update_layout(xaxis={'categoryorder':'total ascending' if ascending else 'total descending'})
            st.plotly_chart(fig, use_container_width=True)

        display_df = chart_df.copy()
        display_df['Rank'] = range(1, len(display_df) + 1)
        display_df = display_df[['Rank', group_col, value_col]]

        # For detailed tables, keep original formatting, not the million/billion rounding
        # This allows detailed view to show exact numbers while KPIs are summarized
        # The specific request was for the "dashboard output" KPIs.
        if value_col == 'Value EGP':
            display_df[value_col] = display_df[value_col].apply(lambda x: f"EGP {x:,.0f}")
        elif value_col == 'Value USD':
            display_df[value_col] = display_df[value_col].apply(lambda x: f"USD {x:,.0f}")
        elif value_col == 'Quantity SQM':
            display_df[value_col] = display_df[value_col].apply(lambda x: f"{x:,.0f} SQM")
        
        new_value_col_name = f'Total {value_col.replace("_", " ")}'
        display_df = display_df.rename(columns={group_col: group_col.replace('_', ' ').title(), value_col: new_value_col_name})
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info(f"No sufficient data to show top/least {num_items} for '{group_col}'.")

# --- Helper Function for Back Button ---
def back_to_main_dashboard_button():
    """Renders a button to navigate back to the main dashboard."""
    if st.button("⬅️ Back to Main Dashboard", key="back_to_dashboard_btn"):
        st.session_state.current_page = "Dashboard Overview"
        st.rerun()

# --- Page Functions ---

def render_dashboard_overview(df):
    """Renders the main dashboard overview page with KPIs and navigation cards."""
    st.header("Demand Planning Overview")

    latest_billing_year = df['Billing Year'].dropna().astype(int).max()
    df_latest_year = df[df['Billing Year'] == latest_billing_year].copy()

    st.subheader(f"KPIs YTD ({latest_billing_year})")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value (EGP)", format_value(df_latest_year['Value EGP'].sum(), 'Value EGP'))
        
        # Determine how to get Total Value (USD)
        total_value_usd_from_data = df_latest_year['Value USD'].sum()
        
        # Check if Value USD is largely zero or NaN for this year
        if total_value_usd_from_data < 0.001: # Check for very small sum, indicating no data
            exchange_rate = st.number_input(
                "Enter EGP to USD Exchange Rate (e.g., 1 USD = X EGP)",
                min_value=1.0, value=30.0, step=0.1, format="%.2f", key="dashboard_exchange_rate_kpi"
            )
            total_value_egp_for_usd_calc = df_latest_year['Value EGP'].sum()
            calculated_usd_value = total_value_egp_for_usd_calc / exchange_rate if exchange_rate > 0 else 0
            col2.metric("Total Value (USD)", format_value(calculated_usd_value, 'Value USD'))
        else:
            col2.metric("Total Value (USD)", format_value(total_value_usd_from_data, 'Value USD'))

        col3.metric("Total Quantity (SQM)", format_value(df_latest_year['Quantity SQM'].sum(), 'Quantity SQM'))

    st.markdown("---") 
    st.markdown("## **Detailed Analysis** ", unsafe_allow_html=True) # More appealing header

    # Navigation Cards
    col_nav1, col_nav2, col_nav3 = st.columns(3)

    with col_nav1:
        with st.container(border=True):
            st.markdown("### 📈 Sales Performance") # Added emoji
            if st.button("Explore Sales Data", use_container_width=True, key="nav_sales_performance"):
                st.session_state.current_page = "Sales Performance"
                st.rerun()

    with col_nav2:
        with st.container(border=True):
            st.markdown("### 📉🗑️ Creel Rationalization") # Added emojis
            if st.button("Optimize Creel Portfolio", use_container_width=True, key="nav_creel_rationalization"):
                st.session_state.current_page = "Creel Rationalization"
                st.rerun()

    with col_nav3:
        with st.container(border=True):
            st.markdown("### 📊 Sales Forecasting") # Added emojis
            if st.button("Generate Future Projections", use_container_width=True, key="nav_sales_forecasting"):
                st.session_state.current_page = "Sales Forecasting"
                st.rerun()

def render_sales_performance(df):
    """Renders the detailed sales performance page."""
    back_to_main_dashboard_button() # ADDED: Back to main dashboard button
    st.header("Sales Performance: Detailed Analysis")

    available_years = df['Billing Year'].dropna().astype(int).unique()

    if len(available_years) > 0:
        selected_year = st.selectbox("Select Billing Year for Analysis", sorted(available_years, reverse=True))
        df_filtered_year = df[df['Billing Year'] == selected_year].copy()
        
        if df_filtered_year.empty:
            st.warning(f"No sales data for {selected_year}.")
            return
        else:
            st.success(f"Analysis for **Year {selected_year}** ({len(df_filtered_year):,} rows):")

            st.container(border=True).markdown("#### Key Performance Indicators")
            with st.container(border=True):
                col1, col2, col3 = st.columns(3)

                total_value_egp = df_filtered_year['Value EGP'].sum()
                col1.metric("Total Value (EGP)", format_value(total_value_egp, 'Value EGP'))

                # --- USD Conversion for Sales Performance KPIs ---
                total_value_usd_from_data = df_filtered_year['Value USD'].sum()
                
                # Check if Value USD is largely zero or NaN for this year or if user wants to override
                if total_value_usd_from_data < 0.001:
                    st.warning("USD sales data for this year is unavailable or negligible. Please provide the EGP to USD exchange rate for equivalence calculation.")
                    exchange_rate_input = st.number_input(
                        "Enter EGP to USD Exchange Rate (e.g., 1 USD = X EGP) for KPI conversion:",
                        min_value=1.0, value=30.0, step=0.1, format="%.2f", key="performance_exchange_rate_kpi"
                    )
                    calculated_usd_value = total_value_egp / exchange_rate_input if exchange_rate_input > 0 else 0
                    col2.metric("Total Value (USD)", format_value(calculated_usd_value, 'Value USD'))
                    st.info(f"USD value calculated using EGP sales and provided exchange rate: 1 USD = {exchange_rate_input:.2f} EGP.")
                else:
                    col2.metric("Total Value (USD)", format_value(total_value_usd_from_data, 'Value USD'))
                # --- End USD Conversion for Sales Performance KPIs ---

                total_quantity_sqm = df_filtered_year['Quantity SQM'].sum()
                col3.metric("Total Quantity (SQM)", format_value(total_quantity_sqm, 'Quantity SQM'))

            st.header("Detailed Sales Breakdowns")
            
            with st.expander("Creel Sales Analysis", expanded=True):
                show_creel_charts = st.checkbox("Show Charts for Creel Analysis", key="creel_charts_perf")
                col_creels1, col_creels2 = st.columns(2)
                with col_creels1:
                    display_top_n(df_filtered_year, 'Creel', 'Quantity SQM', "Top Selling Creels (SQM)", show_chart=show_creel_charts)
                with col_creels2:
                    display_top_n(df_filtered_year, 'Creel', 'Quantity SQM', "Least Selling Creels ( SQM)", ascending=True, show_chart=show_creel_charts)

            with st.expander("Customer & Region Sales Analysis", expanded=True):
                show_customer_region_charts = st.checkbox("Show Charts for Customer/Region Analysis", key="customer_region_charts_perf")
                col_customers_regions1, col_customers_regions2 = st.columns(2)
                with col_customers_regions1:
                    display_top_n(df_filtered_year, 'Customer', 'Value EGP', "Top Selling Customers ( EGP)", show_chart=show_customer_region_charts)
                with col_customers_regions2:
                    display_top_n(df_filtered_year, 'Region', 'Value EGP', "Top Selling Regions ( EGP)", show_chart=show_customer_region_charts)
                
            with st.expander("Quality Sales Analysis", expanded=False):
                show_quality_charts = st.checkbox("Show Charts for Quality Analysis", key="quality_charts_perf")
                display_top_n(df_filtered_year, 'Quality', 'Value EGP', "Top Selling Qualities ( EGP)", show_chart=show_quality_charts)


    else:
        st.error("No valid 'Billing Year' data found. Cannot proceed with year-based analysis.")

def render_suggested_delist_candidates(df_main_full, business_sector_filter):
    """
    Renders a list of suggested creels for future delisting based on specific criteria.
    Criteria: Zero sales in 2024 & 2025 (in selected sector) AND <= 1 customer overall.
    """
    st.header("Suggested Delist Candidates")
    st.info("This list suggests creels for potential future delisting based on the following criteria:"
            "\n- Zero sales (Quantity SQM) in **both 2024 and 2025** (within the selected sector)."
            "\n- Sold to **no more than 1 customer** ever (globally).")

    # 1. Filter data based on business_sector for sales analysis (relevant for 0 sales definition)
    df_sales_analysis = df_main_full.copy()
    if business_sector_filter == "Egypt":
        df_sales_analysis = df_sales_analysis[df_sales_analysis['Is_Egypt_Sector'] == True].copy()
    elif business_sector_filter == "International":
        df_sales_analysis = df_sales_analysis[df_sales_analysis['Is_Egypt_Sector'] == False].copy()

    # Get all unique creels present in the selected sector's data (even if they had 0 sales in target years)
    all_creels_in_sector_data = df_sales_analysis['Creel'].dropna().unique().tolist()
    if 'nan' in all_creels_in_sector_data:
        all_creels_in_sector_data.remove('nan')

    # Calculate sales for 2024 and 2025 for these creels within the current sector filter
    sales_summary_24_25 = df_sales_analysis[
        df_sales_analysis['Billing Year'].isin([2024, 2025])
    ].groupby(['Creel', 'Billing Year'])['Quantity SQM'].sum().unstack(fill_value=0)

    # Identify creels with zero sales in both 2024 and 2025 within the selected sector
    creels_zero_sales_24_25_in_sector = []
    for creel_name in all_creels_in_sector_data:
        qty_2024 = sales_summary_24_25.loc[creel_name, 2024] if creel_name in sales_summary_24_25.index and 2024 in sales_summary_24_25.columns else 0
        qty_2025 = sales_summary_24_25.loc[creel_name, 2025] if creel_name in sales_summary_24_25.index and 2025 in sales_summary_24_25.columns else 0

        if qty_2024 == 0 and qty_2025 == 0:
            creels_zero_sales_24_25_in_sector.append(creel_name)

    # 2. Identify creels with <= 1 unique customer globally (from the entire df_main_full)
    customer_counts_overall = df_main_full.groupby('Creel')['Customer'].nunique()
    creels_single_customer_overall = customer_counts_overall[customer_counts_overall <= 1].index.tolist()

    # 3. Find intersection of these two lists to get final candidates
    candidate_creels_for_delist = list(set(creels_zero_sales_24_25_in_sector) & set(creels_single_customer_overall))

    if not candidate_creels_for_delist:
        st.success("No creels meet all criteria for suggested delisting in the selected sector. Good job!")
        return

    # 4. Prepare data for display for the identified candidates
    # Get total historical sales (Value EGP and Quantity SQM) for these candidates
    # Filter original df_main_full by candidates and selected business sector
    df_candidates_historical = df_sales_analysis[
        df_sales_analysis['Creel'].isin(candidate_creels_for_delist)
    ].copy()

    historical_summary_candidates = df_candidates_historical.groupby('Creel').agg(
        Total_Historical_Value_EGP=('Value EGP', 'sum'),
        Total_Historical_Quantity_SQM=('Quantity SQM', 'sum')
    ).reset_index()

    # Get sales by year for 2023, 2024, 2025 for context (within the selected sector)
    sales_by_year_candidates_context = df_candidates_historical[
        df_candidates_historical['Billing Year'].isin([2023, 2024, 2025])
    ].groupby(['Creel', 'Billing Year'])['Quantity SQM'].sum().unstack(fill_value=0).reindex(columns=[2023, 2024, 2025], fill_value=0)
    sales_by_year_candidates_context.columns = [f'Qty SQM ({col})' for col in sales_by_year_candidates_context.columns]

    # Merge all information into a single DataFrame for display
    delist_summary_df = historical_summary_candidates.merge(
        sales_by_year_candidates_context.reset_index(), on='Creel', how='left'
    )

    # Add unique customer count (overall, from df_main_full)
    unique_customer_counts_for_candidates = customer_counts_overall[
        customer_counts_overall.index.isin(candidate_creels_for_delist)
    ].reset_index(name='Unique Customers (Overall)')
    delist_summary_df = delist_summary_df.merge(unique_customer_counts_for_candidates, on='Creel', how='left')

    # Ensure all columns exist and fill NaNs
    required_qty_cols = [f'Qty SQM ({y})' for y in [2023, 2024, 2025]]
    for col in required_qty_cols:
        if col not in delist_summary_df.columns:
            delist_summary_df[col] = 0
    delist_summary_df[['Total_Historical_Value_EGP', 'Total_Historical_Quantity_SQM', 'Unique Customers (Overall)']] = \
        delist_summary_df[['Total_Historical_Value_EGP', 'Total_Historical_Quantity_SQM', 'Unique Customers (Overall)']].fillna(0)
    
    delist_summary_df['Unique Customers (Overall)'] = delist_summary_df['Unique Customers (Overall)'].astype(int)

    # Sort by least historical quantity, then least historical value
    delist_summary_df = delist_summary_df.sort_values(
        by=['Total_Historical_Quantity_SQM', 'Total_Historical_Value_EGP'],
        ascending=[True, True]
    ).reset_index(drop=True)

    # Format columns for display
    formatted_delist_df = delist_summary_df.copy()
    formatted_delist_df['Total_Historical_Value_EGP'] = formatted_delist_df['Total_Historical_Value_EGP'].apply(lambda x: format_value(x, 'Value EGP'))
    formatted_delist_df['Total_Historical_Quantity_SQM'] = formatted_delist_df['Total_Historical_Quantity_SQM'].apply(lambda x: format_value(x, 'Quantity SQM'))
    for year_col in [f'Qty SQM ({y})' for y in [2023, 2024, 2025]]:
        formatted_delist_df[year_col] = formatted_delist_df[year_col].apply(lambda x: format_value(x, 'Quantity SQM'))
    
    st.dataframe(formatted_delist_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Summary of Delisting Candidates")
    st.markdown(f"- **Total Candidates Identified**: {len(candidate_creels_for_delist):,}")
    st.markdown(f"- **Total Historical Quantity (SQM) of Candidates**: {format_value(delist_summary_df['Total_Historical_Quantity_SQM'].sum(), 'Quantity SQM')}")
    st.markdown(f"- **Total Historical Value (EGP) of Candidates**: {format_value(delist_summary_df['Total_Historical_Value_EGP'].sum(), 'Value EGP')}")
    st.markdown("---")
    st.warning("This list provides potential delisting candidates based on defined criteria. Further strategic review and cross-functional discussion are highly recommended before making final decisions.")

def render_creel_rationalization(df):
    """Renders the creel rationalization page with tabs."""
    back_to_main_dashboard_button()
    st.header("Creel Rationalization")
    
    business_sector = st.selectbox(
        "Select Business Sector for Analysis:",
        ["Both", "Egypt", "International"],
        help="Filter sales data by business sector for both analysis tabs."
    )

    tab1, tab2 = st.tabs(["Current Tail Analysis (2025)", "Suggested Delist Candidates"])

    with tab1:
        st.subheader("Current Tail Analysis: Year 2025")
        st.markdown("""
        Creels with low sales contribution in **Year 2025** based on a cumulative percentage threshold.
        """)

        df_2025_filtered_sector = df[df['Billing Year'] == 2025].copy()

        if business_sector == "Egypt":
            df_2025_filtered_sector = df_2025_filtered_sector[df_2025_filtered_sector['Is_Egypt_Sector'] == True].copy()
        elif business_sector == "International":
            df_2025_filtered_sector = df_2025_filtered_sector[df_2025_filtered_sector['Is_Egypt_Sector'] == False].copy()

        if df_2025_filtered_sector.empty:
            st.warning(f"No sales data for **2025** and **{business_sector}** sector. Tail analysis not available.")
            return
        else:
            all_unique_creels_in_overall_df = df['Creel'].dropna().unique().tolist()
            if 'nan' in all_unique_creels_in_overall_df:
                all_unique_creels_in_overall_df.remove('nan')

            current_year_sales_summary = df_2025_filtered_sector.groupby('Creel')['Quantity SQM'].sum().reset_index()

            sales_by_creel_2025_df = pd.DataFrame({'Creel': all_unique_creels_in_overall_df})
            sales_by_creel_2025_df = sales_by_creel_2025_df.merge(
                current_year_sales_summary, on='Creel', how='left'
            ).fillna(0)
            sales_by_creel_2025 = sales_by_creel_2025_df.set_index('Creel')['Quantity SQM'].sort_values(ascending=True)

            if not sales_by_creel_2025.empty:
                total_sales_quantity_2025 = sales_by_creel_2025.sum()
                if total_sales_quantity_2025 == 0:
                    st.info("Total sales quantity for 2025 in the selected sector is zero. No percentage-based rationalization.")
                else:
                    raw_percentage_col_2025 = (sales_by_creel_2025 / total_sales_quantity_2025 * 100).round(2)

                    sales_by_creel_with_percentage_2025 = pd.DataFrame({
                        'Total Quantity SQM (2025)': sales_by_creel_2025,
                        'Percentage of Total Quantity (2025)': raw_percentage_col_2025
                    })

                    sales_by_creel_with_percentage_2025 = sales_by_creel_with_percentage_2025.sort_values(
                        by='Percentage of Total Quantity (2025)', ascending=True
                    ).copy()

                    sales_by_creel_with_percentage_2025['Cumulative Percentage (2025)'] = \
                        sales_by_creel_with_percentage_2025['Percentage of Total Quantity (2025)'].cumsum()

                    tail_threshold_percentage = st.slider(
                        "Cumulative Percentage Threshold for Tail Analysis (Year 2025):",
                        min_value=0.1, max_value=20.0, value=5.0, step=0.1, format="%.1f%%",
                        help="Creels whose cumulative contribution to total quantity in 2025 is below this percentage."
                    )

                    creels_for_rationalization_summary = sales_by_creel_with_percentage_2025[
                        (sales_by_creel_with_percentage_2025['Cumulative Percentage (2025)'] <= tail_threshold_percentage) &
                        (sales_by_creel_with_percentage_2025.index != 'nan')
                    ].copy()

                    if not creels_for_rationalization_summary.empty:
                        st.subheader(f"Tail Creels for Rationalization (Cumulative up to {tail_threshold_percentage:.1f}% of Total Quantity in 2025 - {business_sector} Sector):")
                        st.warning("Review these Creels for potential discontinuation or strategic adjustments.")
                        
                        rationalized_creels = creels_for_rationalization_summary.index.tolist()
                        
                        df_historical_tail_data = df[
                            (df['Creel'].isin(rationalized_creels)) &
                            (df['Billing Year'] >= 2023)
                        ].copy()

                        if business_sector == "Egypt":
                            df_historical_tail_data = df_historical_tail_data[df_historical_tail_data['Is_Egypt_Sector'] == True].copy()
                        elif business_sector == "International":
                            df_historical_tail_data = df_historical_tail_data[df_historical_tail_data['Is_Egypt_Sector'] == False].copy()

                        historical_qty_pivot = df_historical_tail_data.groupby(['Creel', 'Billing Year'])['Quantity SQM'].sum().unstack(fill_value=0)
                        historical_qty_pivot = historical_qty_pivot.reindex(columns=[2023, 2024, 2025], fill_value=0)
                        historical_qty_pivot.columns = [f'Qty SQM ({col})' for col in historical_qty_pivot.columns]

                        unique_customers_since_2023 = df_historical_tail_data.groupby('Creel')['Customer'].nunique().reset_index(name='Unique Customers (Since 2023)')
                        total_transactions_since_2023 = df_historical_tail_data.groupby('Creel').size().reset_index(name='Total Transactions (Since 2023)')

                        consolidated_summary_display = creels_for_rationalization_summary.reset_index().merge(
                            historical_qty_pivot.reset_index(), on='Creel', how='left'
                        ).merge(
                            unique_customers_since_2023, on='Creel', how='left' 
                        ).merge(
                            total_transactions_since_2023, on='Creel', how='left'
                        ).set_index('Creel')

                        consolidated_summary_display[['Unique Customers (Since 2023)', 'Total Transactions (Since 2023)']] = \
                            consolidated_summary_display[['Unique Customers (Since 2023)', 'Total Transactions (Since 2023)']].fillna(0).astype(int)
                        for year_col in [f'Qty SQM ({y})' for y in [2023, 2024, 2025]]:
                            consolidated_summary_display[year_col] = consolidated_summary_display[year_col].fillna(0)
                        
                        consolidated_summary_display = consolidated_summary_display.sort_values(by='Total Quantity SQM (2025)', ascending=True).copy()

                        if not df_historical_tail_data.empty:
                            customers_per_creel_year = df_historical_tail_data.groupby(['Creel', 'Billing Year'])['Customer'].nunique().reset_index(name='Unique Customer Count')
                            multi_customer_creels_years = customers_per_creel_year[
                                customers_per_creel_year['Unique Customer Count'] > 1
                            ][['Creel', 'Billing Year']].apply(tuple, axis=1).tolist()
                        else:
                            multi_customer_creels_years = []

                        def highlight_multi_customer_in_summary(row):
                            creel = row.name
                            is_multi_customer_across_years = False
                            for year_check in [2023, 2024, 2025]:
                                if (creel, year_check) in multi_customer_creels_years:
                                    is_multi_customer_across_years = True
                                    break 
                            
                            if is_multi_customer_across_years:
                                return ['background-color: orange'] * len(row)
                            return [''] * len(row)

                        formatted_consolidated_summary = consolidated_summary_display.copy()
                        formatted_consolidated_summary['Total Quantity SQM (2025)'] = formatted_consolidated_summary['Total Quantity SQM (2025)'].apply(lambda x: format_value(x, 'Quantity SQM'))
                        formatted_consolidated_summary['Percentage of Total Quantity (2025)'] = formatted_consolidated_summary['Percentage of Total Quantity (2025)'].apply(lambda x: format_value(x, 'Percentage'))
                        formatted_consolidated_summary['Cumulative Percentage (2025)'] = formatted_consolidated_summary['Cumulative Percentage (2025)'].apply(lambda x: format_value(x, 'Percentage'))
                        for year_col in [f'Qty SQM ({y})' for y in [2023, 2024, 2025]]:
                            formatted_consolidated_summary[year_col] = formatted_consolidated_summary[year_col].apply(lambda x: format_value(x, 'Quantity SQM'))
                        
                        st.dataframe(
                            formatted_consolidated_summary.style.apply(highlight_multi_customer_in_summary, axis=1),
                            use_container_width=True
                        )
                        st.markdown("---")
                        st.info("💡 Rows highlighted in orange indicate creels that had sales to more than one customer in at least one of the years (2023, 2024, or 2025) within the selected sector, suggesting they might still have some market relevance despite low overall contribution.")
                    else:
                        st.success(f"No creels found below the {tail_threshold_percentage:.1f}% cumulative quantity threshold for 2025 in the {business_sector} sector. All creels are contributing significantly!")
            else:
                st.info("No sales data available to perform tail analysis for 2025 in the selected sector.")

    with tab2:
        # Pass the full DataFrame to the delist candidates function,
        # as it handles its own filtering based on the selected business sector.
        render_suggested_delist_candidates(df, business_sector)

def render_sales_forecasting(df):
    """Renders the sales forecasting page."""
    back_to_main_dashboard_button()
    st.header("Sales Forecasting")
    st.info("This tool provides a sales forecast based on historical data and user-defined annual growth rates.")

    latest_year_in_data = df['Billing Year'].dropna().astype(int).max()
    st.subheader(f"Forecast from Year {latest_year_in_data + 1}")

    col_forecast1, col_forecast2 = st.columns(2)

    with col_forecast1:
        forecast_years = st.slider(
            "Number of Years to Forecast",
            min_value=1, max_value=5, value=1, step=1,
            help="Select how many years into the future you want to forecast."
        )

    with col_forecast2:
        annual_growth_rate_egp = st.number_input(
            "Annual Growth Rate for EGP Sales (%)",
            min_value=-50.0, max_value=50.0, value=5.0, step=0.1, format="%.1f",
            help="Expected annual growth rate for EGP sales."
        )
        annual_growth_rate_sqm = st.number_input(
            "Annual Growth Rate for Quantity (SQM) Sales (%)",
            min_value=-50.0, max_value=50.0, value=3.0, step=0.1, format="%.1f",
            help="Expected annual growth rate for Quantity (SQM) sales."
        )

    growth_factor_egp = 1 + (annual_growth_rate_egp / 100)
    growth_factor_sqm = 1 + (annual_growth_rate_sqm / 100)

    # Calculate current year's (latest_year_in_data) total sales by sector
    df_current_year = df[df['Billing Year'] == latest_year_in_data].copy()

    current_year_sales_by_sector = df_current_year.groupby('Sector').agg(
        Total_Value_EGP=('Value EGP', 'sum'),
        Total_Quantity_SQM=('Quantity SQM', 'sum')
    ).reset_index()

    if current_year_sales_by_sector.empty:
        st.warning(f"No sales data found for the latest year ({latest_year_in_data}). Cannot generate a forecast.")
        return

    st.markdown("---")
    st.subheader("Forecasted Sales by Sector")

    forecast_results = []

    for year_offset in range(1, forecast_years + 1):
        forecast_year = latest_year_in_data + year_offset
        
        for index, row in current_year_sales_by_sector.iterrows():
            sector = row['Sector']
            base_egp = row['Total_Value_EGP']
            base_sqm = row['Total_Quantity_SQM']

            # Apply growth cumulatively from the latest_year_in_data
            forecasted_egp = base_egp * (growth_factor_egp ** year_offset)
            forecasted_sqm = base_sqm * (growth_factor_sqm ** year_offset)

            # Apply monthly percentages
            monthly_percentages = EGYPT_MONTHLY_PERCENTAGES if sector == 'Egypt' else INTERNATIONAL_MONTHLY_PERCENTAGES
            
            for month_num, percentage in monthly_percentages.items():
                month_name = calendar.month_abbr[month_num]
                forecast_results.append({
                    'Year': forecast_year,
                    'Month': month_name,
                    'Sector': sector,
                    'Forecasted Value EGP': forecasted_egp * percentage,
                    'Forecasted Quantity SQM': forecasted_sqm * percentage
                })

    if forecast_results:
        df_forecast = pd.DataFrame(forecast_results)
        
        # Aggregate by year and sector for summary table
        df_forecast_summary_by_year_sector = df_forecast.groupby(['Year', 'Sector']).agg(
            Total_Forecasted_Value_EGP=('Forecasted Value EGP', 'sum'),
            Total_Forecasted_Quantity_SQM=('Forecasted Quantity SQM', 'sum')
        ).reset_index()

        # Format for display
        df_forecast_summary_by_year_sector['Total_Forecasted_Value_EGP'] = \
            df_forecast_summary_by_year_sector['Total_Forecasted_Value_EGP'].apply(lambda x: format_value(x, 'Value EGP'))
        df_forecast_summary_by_year_sector['Total_Forecasted_Quantity_SQM'] = \
            df_forecast_summary_by_year_sector['Total_Forecasted_Quantity_SQM'].apply(lambda x: format_value(x, 'Quantity SQM'))

        st.dataframe(df_forecast_summary_by_year_sector, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Monthly Breakdown of Forecasted Sales")

        # Aggregate by month and sector for monthly table
        df_forecast_monthly_summary = df_forecast.groupby(['Year', 'Month', 'Sector']).agg(
            Monthly_Forecasted_Value_EGP=('Forecasted Value EGP', 'sum'),
            Monthly_Forecasted_Quantity_SQM=('Forecasted Quantity SQM', 'sum')
        ).reset_index()

        # Order months correctly
        month_order = [calendar.month_abbr[i] for i in range(1, 13)]
        df_forecast_monthly_summary['Month'] = pd.Categorical(df_forecast_monthly_summary['Month'], categories=month_order, ordered=True)
        df_forecast_monthly_summary = df_forecast_monthly_summary.sort_values(by=['Year', 'Month', 'Sector']).reset_index(drop=True)

        # Format for display
        df_forecast_monthly_summary['Monthly_Forecasted_Value_EGP'] = \
            df_forecast_monthly_summary['Monthly_Forecasted_Value_EGP'].apply(lambda x: format_value(x, 'Value EGP'))
        df_forecast_monthly_summary['Monthly_Forecasted_Quantity_SQM'] = \
            df_forecast_monthly_summary['Monthly_Forecasted_Quantity_SQM'].apply(lambda x: format_value(x, 'Quantity SQM'))

        st.dataframe(df_forecast_monthly_summary, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Forecasted Sales Trends")

        # Plotting yearly trends
        df_yearly_plot = df_forecast.groupby(['Year', 'Sector']).agg(
            Total_Value_EGP=('Forecasted Value EGP', 'sum'),
            Total_Quantity_SQM=('Forecasted Quantity SQM', 'sum')
        ).reset_index()

        fig_egp = px.line(df_yearly_plot, x='Year', y='Total_Value_EGP', color='Sector',
                          title='Forecasted Total Value (EGP) by Year and Sector',
                          labels={'Total_Value_EGP': 'Total Value (EGP)', 'Year': 'Year'},
                          markers=True)
        fig_egp.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_egp, use_container_width=True)

        fig_sqm = px.line(df_yearly_plot, x='Year', y='Total_Quantity_SQM', color='Sector',
                          title='Forecasted Total Quantity (SQM) by Year and Sector',
                          labels={'Total_Quantity_SQM': 'Total Quantity (SQM)', 'Year': 'Year'},
                          markers=True)
        fig_sqm.update_layout(xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig_sqm, use_container_width=True)

    else:
        st.warning("No forecast could be generated. Please check your input data and parameters.")


# --- Main Application Logic ---
def main():
    st.set_page_config(layout="wide", page_title="Demand Planning Dashboard")

    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard Overview"

    df_main = load_and_preprocess_data()

    if df_main.empty:
        st.error("Data could not be loaded. Please check the file paths and data integrity.")
        return

    if st.session_state.current_page == "Dashboard Overview":
        render_dashboard_overview(df_main)
    elif st.session_state.current_page == "Sales Performance":
        render_sales_performance(df_main)
    elif st.session_state.current_page == "Creel Rationalization":
        render_creel_rationalization(df_main)
    elif st.session_state.current_page == "Sales Forecasting":
        render_sales_forecasting(df_main)

if __name__ == "__main__":
    main()
