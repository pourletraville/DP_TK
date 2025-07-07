import pandas as pd
import streamlit as st
import os
import plotly.express as px
import datetime
import calendar

# --- Set Pandas Styler max_elements option to handle large DataFrames ---
pd.set_option("styler.render.max_elements", 10_000_000)

# --- Configuration and File Paths ---
# IMPORTANT: Update these paths to match your actual file locations.
# Ensure these paths are accessible by the Streamlit app.
# For example, if running locally, use absolute paths or paths relative to the script's directory.
FILE1_PATH = r"CREEL3YRS-MOD.csv"
FILE2_PATH = r"Creel_arabic_quality.csv"
HEADER_IMAGE_PATH = r"C:\Users\t_helmy\OWICON.png"

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
                        
                        for year_col in [f'Qty SQM ({y})' for y in [2023, 2024, 2025]]:
                            formatted_consolidated_summary[year_col] = formatted_consolidated_summary[year_col].apply(lambda x: format_value(x, 'Quantity SQM'))

                        MAX_CELLS_FOR_STYLING_SUMMARY = 500_000
                        if formatted_consolidated_summary.size > MAX_CELLS_FOR_STYLING_SUMMARY:
                            st.warning(f"Summary table is very large ({formatted_consolidated_summary.size:,} cells). Displaying unstyled data. Adjust threshold for a smaller list.")
                            st.dataframe(formatted_consolidated_summary, use_container_width=True, hide_index=True)
                        else:
                            st.dataframe(
                                formatted_consolidated_summary.style.apply(highlight_multi_customer_in_summary, axis=1),
                                use_container_width=True,
                                hide_index=True
                            )
                            st.info(":orange[**Orange rows**]: Creels sold to multiple customers (since 2023).")

                        st.markdown("---")
                        st.subheader("Summary of Identified Tail Creels")

                        total_creels_shown = consolidated_summary_display.shape[0]
                        total_percentage_of_2025_sales = creels_for_rationalization_summary['Percentage of Total Quantity (2025)'].sum()
                        creels_zero_sales_2025_count = consolidated_summary_display[
                            consolidated_summary_display['Qty SQM (2025)'] == 0
                        ].shape[0]

                        st.markdown(f"- **Total Tail Creels**: {total_creels_shown:,}")
                        st.markdown(f"- **Total 2025 Sales Percentage (Quantity SQM)**: {total_percentage_of_2025_sales:,.2f}%")
                        st.markdown(f"- **Tail Creels with Zero Sales in 2025**: {creels_zero_sales_2025_count:,}")

                    else:
                        st.info(f"No Creels meet the {tail_threshold_percentage:.1f}% cumulative tail criteria for the **{business_sector}** sector in 2025.")
            else:
                st.info(f"No sales data for Creels in 2025 for the **{business_sector}** sector for tail analysis.")

    with tab2:
        render_suggested_delist_candidates(df, business_sector) # Pass the full df and the selected sector

def render_sales_forecasting(df):
    """
    Renders the sales forecasting page.
    Forecasts sales for a target period based on user selections
    and predefined monthly distribution rules.
    """
    back_to_main_dashboard_button() # ADDED: Back to main dashboard button
    st.header("Sales Forecasting")
    
    col_inputs1, col_inputs2 = st.columns(2)

    with col_inputs1:
        available_years = sorted(df['Billing Year'].dropna().astype(int).unique(), reverse=True)
        
        selected_base_years = st.multiselect(
            "1. Select Base Year(s) for Forecasting:",
            options=available_years,
            default=available_years[0] if available_years else [],
            help=" Will be used as the base for the forecast."
        )

        if not selected_base_years:
            st.warning("Please select at least one base year to generate a forecast.")
            return

        # Filtering base data for selected years
        df_base_years = df[df['Billing Year'].isin(selected_base_years)].copy()

        if df_base_years.empty:
            st.warning(f"No sales data for the selected base year(s) {selected_base_years}. Cannot perform forecasting.")
            return
            
        # User selection for Business Sector (Filter 2)
        selected_sector = st.selectbox(
            "2. Filter by Business Sector:",
            ["All", "Egypt", "International"],
            help="Filter business sector before generating the forecast."
        )

    with col_inputs2:
        num_months_to_forecast = st.slider(
            "4. Number of Months to Forecast (from next month):",
            min_value=1,
            max_value=24, # Max 24 months from current
            value=6, # Default to 6 months
            step=1,
            help="starting from the next month."
        )
        
        # Dynamic filter options based on sector (Filter 3 & 4)
        selected_regions = []
        selected_customers = []

        if selected_sector == "All":
            # Populate with all available regions/customers from df_base_years
            all_regions = sorted(df_base_years['Region'].dropna().unique().tolist())
            all_regions = [r for r in all_regions if r != 'nan']
            
            selected_regions = st.multiselect("3. Filter by Specific Region(s) (Optional):", options=["All"] + all_regions, default=["All"])
            
            # For "All" sector, customer list depends on selected regions
            df_for_customer_options = df_base_years.copy()
            if "All" not in selected_regions and selected_regions:
                df_for_customer_options = df_for_customer_options[df_for_customer_options['Region'].isin(selected_regions)]
            
            customer_options_for_multiselect = sorted(df_for_customer_options['Customer'].dropna().unique().tolist())
            customer_options_for_multiselect = [c for c in customer_options_for_multiselect if c != 'nan']

            selected_customers = st.multiselect("3. Filter by Specific Customer(s) (Optional):", options=["All"] + customer_options_for_multiselect, default=["All"])
            st.info("Apply Needed Filters.")

        elif selected_sector == "Egypt":
            egypt_regions = sorted(df_base_years[df_base_years['Is_Egypt_Sector'] == True]['Region'].dropna().unique().tolist())
            egypt_regions = [r for r in egypt_regions if r != 'nan']
            
            selected_regions = st.multiselect("3. Filter by Specific Region(s) (Egypt Sector):", options=["All"] + egypt_regions, default=["All"])

            # Filter customers based on Egypt sector AND selected regions
            df_for_customer_options = df_base_years[df_base_years['Is_Egypt_Sector'] == True].copy()
            if "All" not in selected_regions and selected_regions:
                df_for_customer_options = df_for_customer_options[df_for_customer_options['Region'].isin(selected_regions)]

            egypt_customers = sorted(df_for_customer_options['Customer'].dropna().unique().tolist())
            egypt_customers = [c for c in egypt_customers if c != 'nan']

            selected_customers = st.multiselect("3. Filter by Specific Customer(s) (Egypt Sector):", options=["All"] + egypt_customers, default=["All"])
            st.info("Filters for the 'Egypt' sector are applied. Customer list dynamically updates based on selected regions.")
            
        elif selected_sector == "International":
            international_regions = sorted(df_base_years[df_base_years['Is_Egypt_Sector'] == False]['Region'].dropna().unique().tolist())
            international_regions = [r for r in international_regions if r != 'nan']
            
            selected_regions = st.multiselect("3. Filter by Specific Region(s) (International Sector):", options=["All"] + international_regions, default=["All"])

            # Filter customers based on International sector AND selected regions
            df_for_customer_options = df_base_years[df_base_years['Is_Egypt_Sector'] == False].copy()
            if "All" not in selected_regions and selected_regions:
                df_for_customer_options = df_for_customer_options[df_for_customer_options['Region'].isin(selected_regions)]

            international_customers = sorted(df_for_customer_options['Customer'].dropna().unique().tolist())
            international_customers = [c for c in international_customers if c != 'nan']

            selected_customers = st.multiselect("3. Filter by Specific Customer(s) (International Sector):", options=["All"] + international_customers, default=["All"])
            st.info("Filters for the 'International' sector are applied. Customer list dynamically updates based on selected regions.")
    
    # --- Apply Filters to Base Data ---
    df_filtered_forecast = df_base_years.copy()

    if selected_sector == "Egypt":
        df_filtered_forecast = df_filtered_forecast[df_filtered_forecast['Is_Egypt_Sector'] == True]
    elif selected_sector == "International":
        df_filtered_forecast = df_filtered_forecast[df_filtered_forecast['Is_Egypt_Sector'] == False]

    if "All" not in selected_regions and selected_regions:
        df_filtered_forecast = df_filtered_forecast[df_filtered_forecast['Region'].isin(selected_regions)]

    if "All" not in selected_customers and selected_customers:
        df_filtered_forecast = df_filtered_forecast[df_filtered_forecast['Customer'].isin(selected_customers)]

    if df_filtered_forecast.empty:
        st.warning("No data matches the selected filters for forecasting.")
        return
    
    # --- Core Forecasting Logic ---
    total_quantity_sqm_base = df_filtered_forecast['Quantity SQM'].sum()
    st.subheader(f"Base Quantity (Selected Years & Filters): {format_value(total_quantity_sqm_base, 'Quantity SQM')}")

    # Determine monthly percentages based on selected sector
    monthly_percentages_to_use = {}
    if selected_sector == "Egypt":
        monthly_percentages_to_use = EGYPT_MONTHLY_PERCENTAGES
    elif selected_sector == "International":
        monthly_percentages_to_use = INTERNATIONAL_MONTHLY_PERCENTAGES
    else: # If 'All' or a mix, default to Egypt's for now, or could average/sum
        st.warning("For 'All' sectors, monthly distribution defaults to Egypt's percentages. Consider filtering by specific sector for more accurate distribution.")
        monthly_percentages_to_use = EGYPT_MONTHLY_PERCENTAGES # Or provide a mixed/average set

    st.markdown("---")
    st.subheader("Forecast Parameters")
    col_growth1, col_growth2 = st.columns(2)
    with col_growth1:
        st.write("Current Exchange Rate (for USD conversion):")
        # Ensure a unique key for the exchange rate input in this section
        forecast_exchange_rate = st.number_input(
            "1 USD = X EGP",
            min_value=1.0, value=30.0, step=0.1, format="%.2f", key="forecast_exchange_rate"
        )
    with col_growth2:
        growth_rate = st.slider(
            "Annual Growth Rate for Forecast (%):",
            min_value=-20.0, max_value=50.0, value=10.0, step=0.5, format="%.1f%%"
        ) / 100

    # Get current month and year to determine starting point for forecast
    current_date = datetime.date.today()
    start_month = current_date.month + 1 # Forecast starts from next month
    start_year = current_date.year
    
    if start_month > 12: # If current month is December, forecast starts next year January
        start_month = 1
        start_year += 1

    forecast_data = []
    
    # Initialize forecasted annual total for the first forecast year
    latest_base_year = df_base_years['Billing Year'].max()
    current_forecast_year_annual_sqm = total_quantity_sqm_base

    # If the forecast starts in a year beyond the latest base year, apply initial growth
    if start_year > latest_base_year:
        years_diff = start_year - latest_base_year
        current_forecast_year_annual_sqm *= ((1 + growth_rate) ** years_diff)


    for i in range(num_months_to_forecast):
        month = (start_month + i - 1) % 12 + 1
        year = start_year + (start_month + i - 1) // 12
        
        month_name = calendar.month_name[month]
        
        # If the year changes during the forecast period, apply the annual growth
        if i > 0 and month == 1: # If it's January of a new year in the forecast
            current_forecast_year_annual_sqm *= (1 + growth_rate)

        # Calculate month's share of the current forecast year's total
        monthly_percentage = monthly_percentages_to_use.get(month, 0)
        
        forecasted_monthly_sqm = current_forecast_year_annual_sqm * monthly_percentage
        
        # Calculate EGP and USD based on the average value per SQM from the base data
        avg_egp_per_sqm = df_filtered_forecast['Value EGP'].sum() / total_quantity_sqm_base if total_quantity_sqm_base > 0 else 0
        avg_usd_per_sqm = df_filtered_forecast['Value USD'].sum() / total_quantity_sqm_base if total_quantity_sqm_base > 0 else 0

        forecasted_monthly_egp = forecasted_monthly_sqm * avg_egp_per_sqm
        
        # If USD data is largely missing in base, use EGP conversion for USD forecast
        if avg_usd_per_sqm < 0.001:
            forecasted_monthly_usd = forecasted_monthly_egp / forecast_exchange_rate if forecast_exchange_rate > 0 else 0
        else:
            forecasted_monthly_usd = forecasted_monthly_sqm * avg_usd_per_sqm


        forecast_data.append({
            "Year": year,
            "Month": month_name,
            "Forecasted Quantity (SQM)": forecasted_monthly_sqm,
            "Forecasted Value (EGP)": forecasted_monthly_egp,
            "Forecasted Value (USD)": forecasted_monthly_usd
        })

    df_forecast = pd.DataFrame(forecast_data)

    st.markdown("---")
    st.subheader("Monthly Forecast Details")
    
    # --- Apply 1-decimal point formatting to the DataFrame columns ---
    st.dataframe(df_forecast, use_container_width=True, hide_index=True,
        column_config={
            "Forecasted Quantity (SQM)": st.column_config.NumberColumn(
                "Forecasted Quantity (SQM)", format="%.1f"
            ),
            "Forecasted Value (EGP)": st.column_config.NumberColumn(
                "Forecasted Value (EGP)", format="EGP %.1f"
            ),
            "Forecasted Value (USD)": st.column_config.NumberColumn(
                "Forecasted Value (USD)", format="USD %.1f"
            ),
        }
    )

    st.subheader("Forecast Visualizations")

    if not df_forecast.empty:
        # Plotly chart for forecasted quantity
        fig_sqm = px.bar(df_forecast,
                         x="Month",
                         y="Forecasted Quantity (SQM)",
                         color="Year",
                         title="Monthly Forecasted Quantity (SQM)",
                         labels={"Forecasted Quantity (SQM)": "Quantity (SQM)"},
                         barmode='group') # Group bars by month for different years
        st.plotly_chart(fig_sqm, use_container_width=True)

        # Plotly chart for forecasted value (EGP)
        fig_egp = px.bar(df_forecast,
                         x="Month",
                         y="Forecasted Value (EGP)",
                         color="Year",
                         title="Monthly Forecasted Value (EGP)",
                         labels={"Forecasted Value (EGP)": "Value (EGP)"},
                         barmode='group')
        st.plotly_chart(fig_egp, use_container_width=True)

        # Plotly chart for forecasted value (USD)
        fig_usd = px.bar(df_forecast,
                         x="Month",
                         y="Forecasted Value (USD)",
                         color="Year",
                         title="Monthly Forecasted Value (USD)",
                         labels={"Forecasted Value (USD)": "Value (USD)"},
                         barmode='group')
        st.plotly_chart(fig_usd, use_container_width=True)
    else:
        st.info("No forecast data generated. Adjust your selections.")


# --- Main Application Logic ---
def main():
    st.set_page_config(
        page_title="Demand Planning Dashboard",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for a more appealing sidebar and general look
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6; /* Light gray background */
        }
        .css-vk32pt { /* Targets the main content block */
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        .stSidebar {
            background-color: #e0e6ed; /* Slightly darker gray for sidebar */
            padding-top: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50; /* Green */
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }
        .stButton>button:active {
            background-color: #3e8e41;
            transform: translateY(0);
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50; /* Darker text for headers */
        }
        .stMetric {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.2rem;
        }
        /* Style for containers with border=True */
        .st-emotion-cache-1jm6gsa { /* Adjust this class if Streamlit's internal class names change */
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: #ffffff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard Overview"

    # Display header image
    if os.path.exists(HEADER_IMAGE_PATH):
        # Use use_container_width instead of use_column_width to remove deprecation warning
        st.sidebar.image(HEADER_IMAGE_PATH, use_container_width=True) 
    else:
        st.sidebar.warning(f"Header image not found at {HEADER_IMAGE_PATH}")

    st.sidebar.title("Navigation")
    
    # Navigation buttons in the sidebar
    if st.sidebar.button("Dashboard Overview", key="sidebar_dashboard_overview"):
        st.session_state.current_page = "Dashboard Overview"
    if st.sidebar.button("Sales Performance", key="sidebar_sales_performance"):
        st.session_state.current_page = "Sales Performance"
    if st.sidebar.button("Creel Rationalization", key="sidebar_creel_rationalization"):
        st.session_state.current_page = "Creel Rationalization"
    if st.sidebar.button("Sales Forecasting", key="sidebar_sales_forecasting"):
        st.session_state.current_page = "Sales Forecasting"

    df_main = load_and_preprocess_data()

    if df_main.empty:
        st.error("Cannot load or process data. Please check file paths and data integrity.")
        return

    # Render selected page
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
