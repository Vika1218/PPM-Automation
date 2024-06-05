# Import packages
import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import pingouin as pg
from datetime import datetime

# Import customized packages
from data_processing_module import connect_to_database, fetch_data_from_database, process_orders, process_detail_views, process_country_cost, merge_and_fill, rev_per_dv_model, cr_model, rev_per_dv_anova, cr_anova, rev_per_dv_model_dma, cr_model_dma, rev_per_dv_anova_dma, cr_anova_dma
from output_format_module import number_format, rename_column, output_format

# Streamlit app
def main():
    
    st.set_page_config(page_title="PPM Analysis - Basic",layout='wide')

    st.title("Product Feature Preference Model - Basic")

    st.markdown('''### Step 1: Define the Overall Analysis Scope''')
    st.write('''Which **products/features** are more preferred in the **selected region** vs the **baseline**? Measured by which **metric**?''')

    col1, col2, col3 = st.columns(3)
    with col1:
        market = st.selectbox("Select Market to Analyze", ["US",'AU'], 
                                       help = '''Market you need to analyse''',
                                       index=0)
    with col2:
        if market == 'US':
            analysed_level = st.selectbox("Select Granularity for Analysed Region", ["Regional Level",'DMA Level'], 
                                       help = '''Granularity for region analysed''',
                                       index=0)
        elif market == 'AU':
            analysed_level = st.selectbox("Select Granularity for Analysed Region", ["Regional Level"], 
                                       help = '''Granularity for region analysed''',
                                       index=0)
    with col3:
        if analysed_level == 'DMA Level':
            baseline_level = st.selectbox("Select Granularity for Baseline Region", ["Country & Regional Level",'DMA Level'], 
                                       help = '''Granularity for region baseline''',
                                       index=0)
        else:
            pass

    #market, region_analysed level, region_baseline level
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        metric_analysed = st.selectbox("Select Metric to Analyze", ["Rev per DV", "CR"], 
                                       help = '''Metric you need to analyse  \n -> Rev per DV: Revenue/Detailviews in the analysed region (= Total Revenue/Total Detailviews)  \n -> CR: Conversion rate in the analysed region (= Total Order/Total Detailviews)''',
                                       index=0)
    with col5:
        if analysed_level == 'Regional Level':
            if market == 'US':
                region_analysed = st.selectbox("Select Analysed Region", ["US West", "US East", "US Southeast", "US Northwest"], 
                                       help = '''Region you need to analyse''',
                                       index=0)
            elif market == 'AU':
                region_analysed = st.selectbox("Select Analysed Region", ['AU Sydney','AU Melbourne','AU Perth'], 
                                       help = '''Region you need to analyse''',
                                       index=0)
        elif analysed_level == 'DMA Level':
            region_analysed = st.selectbox("Select Analysed Region", ["New York, NY","Los Angeles CA","Washington DC (Hagerstown MD)","San Francisco-Oakland-San Jose CA","Seattle-Tacoma WA"], 
                                       help = ''' Region you need to analyse (Top 5 US DMAs available)''',
                                       index=0)
    with col6:
        if analysed_level == 'Regional Level':
            if market == 'US':
                region_baseline = st.selectbox("Select Baseline Region", ["US All", "US West", "US East", "US Southeast", "US Northwest"], 
                                       help = '''Region as the baseline in the comparison process  \n -> US All means comparing your analysed region to the overall US market'''
                                       ,index=0)
            elif market == 'AU':
                region_baseline = st.selectbox("Select Baseline Region", ["AU All", 'AU Sydney','AU Melbourne','AU Perth'], 
                                       help = '''Region as the baseline in the comparison process  \n -> US All means comparing your analysed region to the overall US market'''
                                       ,index=0)
        elif analysed_level == 'DMA Level':
            if baseline_level == 'Country & Regional Level':
                region_baseline = st.selectbox("Select Baseline Region", ["US All", "US West", "US East", "US Southeast", "US Northwest"], 
                                       help = '''Region as the baseline in the comparison process  \n -> US All means comparing your analysed region to the overall US market'''
                                       ,index=0)
            elif baseline_level == 'DMA Level':
                region_baseline = st.selectbox("Select Baseline Region", ["New York, NY","Los Angeles CA","Washington DC (Hagerstown MD)","San Francisco-Oakland-San Jose CA","Seattle-Tacoma WA"], 
                                       help = '''Region as the baseline in the comparison process (Top 5 US DMAs available)'''
                                       ,index=0)

    with col7:
        feature = st.selectbox("Select Feature Dimension", ["market_sku", "market_spu", "category", "subcategory", "collection", "color_tone", "material_helper"], 
                               help = '''Product feature you need to analyse''',
                               index=0)
        
    # Date range
    col8, col9, col10, col11 = st.columns(4)
    with col8:
        start_date = st.date_input("Select Start Date", datetime(datetime.now().year, 1, 1), help = '''State the start date for your analysis''')
    with col9:
        end_date = st.date_input("Select End Date", help = '''State the end date for your analysis''')
    if end_date < start_date:
        st.error("End date must be later than start date. Please select a valid end date.")
    else:
        params = {'market': market,'start_date': start_date, 'end_date': end_date}
    with col10:
        rank = st.radio("Select Rank", ["Top", "Bottom"], 
                        help = '''Rank the output of model  \n -> Select 'Top' to find products with better performance in analysed region  \n -> Select 'Bottom' to find products with worse performance in analysed region''',
                        index=0)
    with col11:
        output_limit = st.slider("Select Output Limit", min_value=1, max_value=100, value=30,
                                 help = '''Select the number of records to display in the output.  \n -> Keep in mind that the final output may contain fewer records depending on the threshold and date range you set''')
    
    st.divider()

    st.markdown('''### Step 2: Set Thresholds for Control Metrics''')
    st.write('''To ensure a robust analysis, set minimum thresholds for various control metrics to filter products based on their data size and performance.''')

    col12,col13, col14, col15 = st.columns(4)
    with col12:
        metric_control_1 = 'country_total_cost_per_sku'
        metric_threshold_1 = st.number_input(f"Threshold for {market} Ave. Cost per SKU >=", min_value = 0, value=100, step=1,
                                             help = f'''Set the minimum cost per sku in the {market} market  \n -> E.G., if analyzing by category, this threshold represents the average costs per SKU spent on your seleted category at {market} country level within your specified date range  \n Adjust this value based on your selected features and date range:  \n -> Example reference: at least $100 per sku per month
                                             ''',
                                             key="country_total_cost_threshold_per_sku")
    with col13:
        if metric_analysed == 'Rev per DV':
            metric_control_2 = 'rev_per_dv_analysed'
            metric_threshold_2 = st.number_input(f"Threshold for {metric_analysed}_Analysed >=", min_value = 0.0, value=0.0, step=0.1, 
                                                 help = '''Set the minimum Rev per DV for products in the analysed region  \n -> Example reference: Rev per DV >= 2.00''',
                                                 key="metric_analyzed_threshold")
        elif metric_analysed == 'CR':
            metric_control_2 = 'CR_analysed'
            metric_threshold_2 = st.number_input(f"Threshold for {metric_analysed}_Analysed >=(%)", min_value = 0.00, value=0.00, step=0.01, 
                                                 help = '''Set the minimum CR for products in the analysed region  \n -> The format is already in percentage  \n -> Example reference: CR >= 0.08% --> Enter 0.08 here''', 
                                                 key="metric_analyzed_threshold")/ 100.0
    with col14:
        metric_control_3 = 'total_order'
        metric_threshold_3 = st.number_input("Threshold for Total Order >=", min_value = 0, value=10, step=1, 
                                             help = '''Set the minimum orders for products in the analysed region  \n -> Example reference: Total Order >= 10''',
                                             key="total_order_threshold")
    with col15:
        metric_control_4 = 'average_price_analysed'
        metric_threshold_4 = st.number_input("Threshold for Average Price >=", min_value = 0, value=100, step=1, 
                                             help = '''Set the minimum orders for products in the analysed region  \n -> Average Price = Total Revenue/Total Quantity  \n -> Example reference: Total Order >= 100''',
                                             key="average_price_analysed_threshold")

    st.divider()

    st.markdown('''### Step 3: Further Specify the Analysis Scope (If Necessary)''')
    st.write('''Filter products to deep dive into specific category/subcategory/collection, default is blank (no filter). Refer to the sidebar for filtered dimensions.''')
    st.write(''':exclamation: If the filtered data is too small, it might result in 0 model output.''')
    
    # Multi-select function
    def multiselect_customized(df,filter_name):
        option = st.multiselect(f"Select {filter_name.capitalize()} to Analyze",df[filter_name].unique(),
                                placeholder="Choose one ore more options; Leaving blank here for choosing all options",
                                help = f'''Select to see the output only within one or multiple {filter_name} option(s)  \n -> Leaving the box blank to select all options  \n -> See the sidebar for the details of selected options''')
        if not option:
            option = df[filter_name].unique()
        st.sidebar.write(f"Selected {filter_name}:")
        st.sidebar.dataframe(option,width=300, height=120)
        return option

    # Multi-select section
    df_cateoption = pd.read_excel('Product_Info for MultiSelect.xlsx', sheet_name = 'category_subcategory')
    df_collection = pd.read_excel('Product_Info for MultiSelect.xlsx', sheet_name = 'collection')
    st.sidebar.subheader('''Basic Model - Filtered Dimensions in Step 3''')
    category_option = multiselect_customized(df_cateoption,"category")
    subcategory_option = multiselect_customized(df_cateoption,"subcategory")
    collection_option = multiselect_customized(df_collection,"collection")
    
    if feature == 'color_tone':
        df_color_tone = pd.read_excel('Product_Info for MultiSelect.xlsx', sheet_name='color_tone')
        color_tone_option = multiselect_customized(df_color_tone,"color_tone")
    elif feature == 'material_helper':
        df_material_helper = pd.read_excel('Product_Info for MultiSelect.xlsx', sheet_name='material_helper')
        material_helper_option = multiselect_customized(df_material_helper,"material_helper")                                                   

    # Trigger data processing on user input
    if st.button("Process Data",help = '''Press button to proceed the analysis'''):
       # change the according info to connect to the database
       host = 'dw-prod.cfujfnms1rth.ap-southeast-1.redshift.amazonaws.com'
       database = 'dwd_prod'
       username = 'yunyi_cheng' #change if needed
       password = 'A00JbR&3' #change if needed
       port = 5439 #change if needed
       
       # order record: change the query if needed
       order_query_region = """
                      SELECT
                      fs3.market_sku || '_' ||fs2.region || '_' ||date(fs2.payment_completion_time)::text AS unique_id,
                      date(fs2.payment_completion_time) as order_date,
                      fs2.market,
                      fs2.region,
                      fs3.market_sku,
                      fs3.sku_name,
                      fs3.market_spu,
                      fs3.spu_name,
                      ds.master_category,
                      ds.category,
                      ds.subcategory,
                      ds.collection,
                      ds.color_tone,
                      COALESCE(dskp_material_helper.value, dspp_material_helper.value) AS material_helper,
                      sum(fs3.sale_amount) as total_revenue,
                      sum(fs3.quantity) as total_quantity,
                      count(distinct fs2.spree_so_id) as total_order
                      from fact_saleorder fs2
                      LEFT JOIN fact_saleorderline fs3 ON fs2.spree_so_id = fs3.spree_so_id
                      LEFT JOIN dim_sku ds ON fs3.market_sku = ds.market_sku
                      LEFT JOIN ( SELECT market_sku, value
                      FROM dim_sku_property
                      WHERE property_type = 'Material Helper') dskp_material_helper ON ds.market_sku = dskp_material_helper.market_sku
                      LEFT JOIN ( SELECT market_spu, value
                      FROM dim_spu_property
                      WHERE property_type = 'Material Helper') dspp_material_helper ON ds.market_spu = dspp_material_helper.market_spu
                      where fs2.spree_channel = 'web'
                      AND fs2.market = %s
                      AND fs2.classification = 'complete'
                      AND fs2.order_type = 'Goods'
                      AND fs2.pure_swatch_order_flag = 'N'
                      AND ds.category != 'Swatch'
                      AND ds.category IS NOT NULL
                      AND ds.subcategory != 'Accident Protection Plan'
                      and DATE(fs2.payment_completion_time) BETWEEN %s AND %s
                      group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14
                      order by 1,3
                      """
       dv_query_region = """
                      SELECT 
                      frpdom.market_sku || '_' || frpdom.region || '_' || frpdom.date::TEXT AS unique_id,
                      frpdom.date,
                      frpdom.region,
                      frpdom.market_sku,
                      ds.sku_name,
                      ds.market_spu,
                      ds.spu_name,
                      ds.master_category,
                      ds.category,
                      ds.subcategory,
                      ds.collection,
                      ds.color_tone,
                      COALESCE(dskp_material_helper.value, dspp_material_helper.value) AS material_helper,
                      SUM(frpdom.product_view) AS total_detailview
                      FROM fact_region_product_daily_onsite_metric frpdom
                      LEFT JOIN 
                          dim_sku ds ON frpdom.market_sku = ds.market_sku
                      LEFT JOIN 
                          (SELECT market_sku, value FROM dim_sku_property WHERE property_type = 'Material Helper') dskp_material_helper ON ds.market_sku = dskp_material_helper.market_sku
                      LEFT JOIN 
                          (SELECT market_spu, value FROM dim_spu_property WHERE property_type = 'Material Helper') dspp_material_helper ON ds.market_spu = dspp_material_helper.market_spu
                      WHERE frpdom.market = %s
                         AND frpdom.date BETWEEN %s AND %s
                         and ds.category != 'Swatch'
                         and ds.category IS NOT NULL
                         and frpdom.region != 'Overseas'
                      GROUP BY 1, 2, 3, 4,5,6,7,8,9,10,11,12,13
                      """
           
       order_query_dma = """
                      SELECT
                      fs3.market_sku || '_' ||dd.ga_dma || '_' ||date(fs2.payment_completion_time)::text AS unique_id,
                      date(fs2.payment_completion_time) as order_date,
                      fs2.market,
                      dd.ga_dma as dma, 
                      fs3.market_sku,
                      fs3.sku_name,
                      fs3.market_spu,
                      fs3.spu_name,
                      ds.master_category,
                      ds.category,
                      ds.subcategory,
                      ds.collection,
                      ds.color_tone,
                      COALESCE(dskp_material_helper.value, dspp_material_helper.value) AS material_helper,
                      sum(fs3.sale_amount) as total_revenue,
                      sum(fs3.quantity) as total_quantity,
                      count(distinct fs2.spree_so_id) as total_order
                      from fact_saleorder fs2
                      LEFT JOIN fact_saleorderline fs3 ON fs2.spree_so_id = fs3.spree_so_id
                      left join helper_us_census huc on huc.zip = fs2.shipping_zip_code 
                      left join dim_dma dd on dd.dma_code = huc.dma_code 
                      LEFT JOIN dim_sku ds ON fs3.market_sku = ds.market_sku
                      LEFT JOIN ( SELECT market_sku, value
                                  FROM dim_sku_property
                                  WHERE property_type = 'Material Helper') dskp_material_helper ON ds.market_sku = dskp_material_helper.market_sku
                      LEFT JOIN ( SELECT market_spu, value
                                  FROM dim_spu_property
                                  WHERE property_type = 'Material Helper') dspp_material_helper ON ds.market_spu = dspp_material_helper.market_spu
                      where fs2.spree_channel = 'web'
                      AND fs2.market = %s
                      AND fs2.classification = 'complete'
                      AND fs2.order_type = 'Goods'
                      AND fs2.pure_swatch_order_flag = 'N'
                      AND ds.category != 'Swatch'
                      AND ds.category IS NOT NULL
                      AND ds.subcategory != 'Accident Protection Plan'
                      and DATE(fs2.payment_completion_time) between %s AND %s
                      and dd.ga_dma in ('New York, NY','Los Angeles CA','Washington DC (Hagerstown MD)','San Francisco-Oakland-San Jose CA','Seattle-Tacoma WA')
                      group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14
                      order by 1,3
                      """
       dv_query_dma = """
                      SELECT
                          frpdom.market_sku || '_' || frpdom.metro || '_' || frpdom.date ::text AS unique_id,
                          frpdom.date,
                          frpdom.metro AS dma,
                          frpdom.market_sku,
                          ds.sku_name,
                          ds.market_spu,
                          ds.spu_name,
                          ds.master_category,
                          ds.category,
                          ds.subcategory,
                          ds.collection,
                          ds.color_tone,
                          COALESCE(dskp_material_helper.value, dspp_material_helper.value) AS material_helper,
                          SUM(frpdom.product_view) AS total_detailview
                      FROM
                          fact_dma_product_daily_onsite_metric frpdom
                      LEFT JOIN dim_sku ds ON frpdom.market_sku = ds.market_sku
                      LEFT JOIN (
                          SELECT
                              market_sku,
                              value
                          FROM
                              dim_sku_property
                          WHERE
                              property_type = 'Material Helper'
                      ) dskp_material_helper ON ds.market_sku = dskp_material_helper.market_sku
                      LEFT JOIN (
                          SELECT
                              market_spu,
                              value
                          FROM
                              dim_spu_property
                          WHERE
                              property_type = 'Material Helper'
                      ) dspp_material_helper ON ds.market_spu = dspp_material_helper.market_spu
                      WHERE
                          frpdom.market = %s
                          AND frpdom.date BETWEEN %s AND %s
                          AND ds.category != 'Swatch'
                          AND ds.category IS NOT NULL
                          AND frpdom.metro in ('New York, NY','Los Angeles CA','Washington DC (Hagerstown MD)','San Francisco-Oakland-San Jose CA','Seattle-Tacoma WA')
                      GROUP BY
                          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
                      """
        
       country_cost_query = """
                      select 
                      fpam.market_sku,
                      ds.sku_name,
                      ds.market_spu,
                      ds.spu_name,
                      ds.master_category,
                      ds.category,
                      ds.subcategory,
                      ds.collection,
                      ds.color_tone,
                      COALESCE(dskp_material_helper.value, dspp_material_helper.value) AS material_helper,
                      sum(fpam.cost) as country_total_cost
                      from fact_product_ads_metric fpam
                      LEFT JOIN dim_sku ds ON fpam.market_sku = ds.market_sku
                      LEFT JOIN ( SELECT market_sku, value
                      FROM dim_sku_property
                      WHERE property_type = 'Material Helper') dskp_material_helper ON ds.market_sku = dskp_material_helper.market_sku
                      LEFT JOIN ( SELECT market_spu, value
                      FROM dim_spu_property
                      WHERE property_type = 'Material Helper') dspp_material_helper ON ds.market_spu = dspp_material_helper.market_spu
                      where fpam.market = %s
                      and fpam.date between %s AND %s
                      group by 1,2,3,4,5,6,7,8,9,10
                      """
       
       # Connect to the database & extract data & merge dataset
       conn = connect_to_database(host, database, username, password, port)
       try:
           conn = connect_to_database(host, database, username, password, port)
       except psycopg2.OperationalError as e:
           st.error("Error connecting to the database: {}. Try it later.".format(e))
           # Stop program execution
           raise SystemExit("Error connecting to the database")
       if analysed_level == 'Regional Level':
           # fetch data
           df_order = fetch_data_from_database(conn, order_query_region, params)
           df_dv = fetch_data_from_database(conn, dv_query_region, params)
           df_country_cost = fetch_data_from_database(conn, country_cost_query, params)
           conn.close()
           df_merge = merge_and_fill(df_order,df_dv,market,analysed_level)

           #create key
           df_merge['cate-feature'] = df_merge['category'] + ": " + df_merge[feature]

           # Calculate order percent
           df_merge['order_percent_of_total'] = df_merge.groupby('region')['total_order'].transform(lambda x: x / x.sum())
           df_merge['order_percent_of_category'] = df_merge.groupby(['region','category'])['total_order'].transform(lambda x: x / x.sum())
           
           # Filter out required data
           if feature == 'color_tone':
               df_merge = df_merge[(df_merge['category'].isin(category_option)) &
                               (df_merge['subcategory'].isin(subcategory_option)) &
                               (df_merge['collection'].isin(collection_option)) &
                                (df_merge['color_tone'].isin(color_tone_option))]
           elif feature == 'material_helper':
               df_merge = df_merge[(df_merge['category'].isin(category_option)) &
                               (df_merge['subcategory'].isin(subcategory_option)) &
                               (df_merge['collection'].isin(collection_option)) &
                                (df_merge['material_helper'].isin(material_helper_option))]
           else:
               df_merge = df_merge[(df_merge['category'].isin(category_option)) &
                               (df_merge['subcategory'].isin(subcategory_option)) &
                               (df_merge['collection'].isin(collection_option))]
           # raise error message if needed
           if df_merge.empty == True:
               st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1 or adding more options in step 3.")
               # Stop program execution
               raise SystemExit("Program halted due to empty DataFrame.")
           else:
               pass
           
       elif analysed_level == 'DMA Level' and baseline_level == 'Country & Regional Level':
           # fetch data
           df_order_analysed = fetch_data_from_database(conn, order_query_dma, params)
           df_order_baseline = fetch_data_from_database(conn, order_query_region, params)
           df_dv_analysed = fetch_data_from_database(conn, dv_query_dma, params)
           df_dv_baseline = fetch_data_from_database(conn, dv_query_region, params)
           df_country_cost = fetch_data_from_database(conn, country_cost_query, params)
           conn.close()
           df_merge_analysed = merge_and_fill(df_order_analysed,df_dv_analysed,market,analysed_level,include=0)
           df_merge_baseline = merge_and_fill(df_order_baseline,df_dv_baseline,market,analysed_level,include=1)

           #create key
           df_merge_analysed['cate-feature'] = df_merge_analysed['category'] + ": " + df_merge_analysed[feature]
           df_merge_baseline['cate-feature'] = df_merge_baseline['category'] + ": " + df_merge_baseline[feature]

           # Calculate order percent
           df_merge_analysed['order_percent_of_total'] = df_merge_analysed.groupby('dma')['total_order'].transform(lambda x: x / x.sum())
           df_merge_analysed['order_percent_of_category'] = df_merge_analysed.groupby(['dma','category'])['total_order'].transform(lambda x: x / x.sum())

           # Filter out required data
           if feature == 'color_tone':
            df_merge_analysed = df_merge_analysed[(df_merge_analysed['category'].isin(category_option)) &
                                                  (df_merge_analysed['subcategory'].isin(subcategory_option)) &
                                                  (df_merge_analysed['collection'].isin(collection_option)) &
                                                  (df_merge_analysed['color_tone'].isin(color_tone_option))]
            df_merge_baseline = df_merge_baseline[(df_merge_baseline['category'].isin(category_option)) &
                                                  (df_merge_baseline['subcategory'].isin(subcategory_option)) &
                                                  (df_merge_baseline['collection'].isin(collection_option)) &
                                                  (df_merge_baseline['color_tone'].isin(color_tone_option))]
           elif feature == 'material_helper':
            df_merge_analysed = df_merge_analysed[(df_merge_analysed['category'].isin(category_option)) &
                                                  (df_merge_analysed['subcategory'].isin(subcategory_option)) &
                                                  (df_merge_analysed['collection'].isin(collection_option)) &
                                                  (df_merge_analysed['material_helper'].isin(material_helper_option))]
            df_merge_baseline = df_merge_baseline[(df_merge_baseline['category'].isin(category_option)) &
                                                  (df_merge_baseline['subcategory'].isin(subcategory_option)) &
                                                  (df_merge_baseline['collection'].isin(collection_option)) &
                                                  (df_merge_baseline['material_helper'].isin(material_helper_option))]
           else:
            df_merge_analysed = df_merge_analysed[(df_merge_analysed['category'].isin(category_option)) &
                                                  (df_merge_analysed['subcategory'].isin(subcategory_option)) &
                                                  (df_merge_analysed['collection'].isin(collection_option))]
            df_merge_baseline = df_merge_baseline[(df_merge_baseline['category'].isin(category_option)) &
                                                  (df_merge_baseline['subcategory'].isin(subcategory_option)) &
                                                  (df_merge_baseline['collection'].isin(collection_option))]
            
            # raise error message if needed
           if df_merge_analysed.empty == True or df_merge_baseline.empty == True:
               st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1 or adding more options in step 3.")
               # Stop program execution
               raise SystemExit("Program halted due to empty DataFrame.")
           else:
               pass

       elif analysed_level == 'DMA Level' and baseline_level == 'DMA Level':
           # fetch data
           df_order = fetch_data_from_database(conn, order_query_dma, params)
           df_dv = fetch_data_from_database(conn, dv_query_dma, params)
           df_country_cost = fetch_data_from_database(conn, country_cost_query, params)
           conn.close()
           df_merge = merge_and_fill(df_order,df_dv,market,analysed_level,include=0)

           #create key
           df_merge['cate-feature'] = df_merge['category'] + ": " + df_merge[feature]

           # Calculate order percent
           df_merge['order_percent_of_total'] = df_merge.groupby('dma')['total_order'].transform(lambda x: x / x.sum())
           df_merge['order_percent_of_category'] = df_merge.groupby(['dma','category'])['total_order'].transform(lambda x: x / x.sum())
           
           # Filter out required data
           if feature == 'color_tone':
               df_merge = df_merge[(df_merge['category'].isin(category_option)) &
                               (df_merge['subcategory'].isin(subcategory_option)) &
                               (df_merge['collection'].isin(collection_option)) &
                                (df_merge['color_tone'].isin(color_tone_option))]
           elif feature == 'material_helper':
               df_merge = df_merge[(df_merge['category'].isin(category_option)) &
                               (df_merge['subcategory'].isin(subcategory_option)) &
                               (df_merge['collection'].isin(collection_option)) &
                                (df_merge['material_helper'].isin(material_helper_option))]
           else:
               df_merge = df_merge[(df_merge['category'].isin(category_option)) &
                               (df_merge['subcategory'].isin(subcategory_option)) &
                               (df_merge['collection'].isin(collection_option))]   
           # raise error message if needed
           if df_merge.empty == True:
               st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1 or adding more options in step 3.")
               # Stop program execution
               raise SystemExit("Program halted due to empty DataFrame.")
           else:
               pass
        
       if metric_analysed == 'Rev per DV':
            
            # Call comparison model & ANOVA
            if analysed_level == 'Regional Level':
                output_rev_per_dv = rev_per_dv_model(df_merge, df_country_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2, metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4)
            elif analysed_level == 'DMA Level':
                if baseline_level == 'Country & Regional Level':
                    output_rev_per_dv = rev_per_dv_model_dma(df_country_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2, metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4, 
                            baseline_level, df_merge_analysed=df_merge_analysed,df_merge_baseline=df_merge_baseline)
                elif baseline_level == 'DMA Level':
                    output_rev_per_dv = rev_per_dv_model_dma(df_country_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2, metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4, 
                            baseline_level, df_merge=df_merge)

            if output_rev_per_dv.empty == True:
                st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1, adjusting thresholds in step 2, or adding more options in step 3.")
                # Stop program execution
                raise SystemExit("Program halted due to empty DataFrame.")
            else:
                pass

            if analysed_level == 'Regional Level':
                anova_each_rev_per_dv = rev_per_dv_anova(df_merge, output_rev_per_dv, region_analysed, region_baseline, feature)
            elif analysed_level == 'DMA Level':
                if baseline_level == 'Country & Regional Level':
                    anova_each_rev_per_dv = rev_per_dv_anova_dma(output_rev_per_dv, region_analysed, region_baseline, feature,
                                                             baseline_level, df_merge_analysed=df_merge_analysed, df_merge_baseline=df_merge_baseline)
                elif baseline_level == 'DMA Level':
                    anova_each_rev_per_dv = rev_per_dv_anova_dma(output_rev_per_dv, region_analysed, region_baseline, feature,
                                                             baseline_level, df_merge=df_merge)
            
            # Get final output
            output_rev_per_dv = pd.merge(output_rev_per_dv, anova_each_rev_per_dv[['cate-feature', 'p_value']], on='cate-feature', how = 'left')
            output_rev_per_dv = output_format(output_rev_per_dv, feature)
            st.table(output_rev_per_dv)

            # Allow users to download the data as CSV
            csv_data = output_rev_per_dv.to_csv(index=False)
            st.download_button(label="Download Data as CSV", data=csv_data, 
                               file_name=f"{region_analysed} vs. {region_baseline}_{feature}_Rev per DV_{start_date} to {end_date}.csv",
                               help = '''Press button to download output as csv file''',
                               key="download_data")
        
       elif metric_analysed == 'CR':
            
            # Call comparison model & ANOVA
            if analysed_level == 'Regional Level':
                output_cr = cr_model(df_merge, df_country_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2, metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4)
            elif analysed_level == 'DMA Level':
                if analysed_level == 'Regional Level':
                    output_cr = cr_model_dma(df_country_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2, metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4, 
                            baseline_level, df_merge_analysed=df_merge_analysed,df_merge_baseline=df_merge_baseline)
                elif baseline_level == 'DMA Level':
                    output_cr = cr_model_dma(df_country_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2, metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4, 
                            baseline_level, df_merge=df_merge)
            
            if output_cr.empty == True:
                st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1, adjusting thresholds in step 2, or adding more options in step 3.")
                # Stop program execution
                raise SystemExit("Program halted due to empty DataFrame.")
            else:
                pass

            if analysed_level == 'Regional Level':
                anova_each_cr = cr_anova(df_merge, output_cr, region_analysed, region_baseline, feature)
            elif analysed_level == 'DMA Level':
                if analysed_level == 'Regional Level':
                    anova_each_cr = cr_anova_dma(output_cr, region_analysed, region_baseline, feature,
                                                 baseline_level, df_merge_analysed=df_merge_analysed, df_merge_baseline=df_merge_baseline)
                elif baseline_level == 'DMA Level':
                    anova_each_cr = cr_anova_dma(output_cr, region_analysed, region_baseline, feature,
                                                 baseline_level, df_merge=df_merge)
            
            # Get final output
            output_cr = pd.merge(output_cr, anova_each_cr[['cate-feature', 'p_value']], on='cate-feature', how = 'left')
            output_cr = output_format(output_cr, feature)
            st.table(output_cr)

            # Allow users to download the data as CSV
            csv_data = output_cr.to_csv(index=False)
            st.download_button(label="Download Data as CSV", data=csv_data, 
                               file_name=f"{region_analysed} vs. {region_baseline}_{feature}_CR_{start_date} to {end_date}.csv", 
                               help = '''Press button to download output as csv file''',
                               key="download_data")


if __name__ == "__main__":
    main()