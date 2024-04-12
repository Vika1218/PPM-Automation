# Import packages
import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import pingouin as pg
from datetime import datetime

# Import customized packages
from reverse_process import connect_to_database, fetch_data_from_database, process_orders, process_detail_views, process_us_cost, merge_and_fill, rev_per_dv_model, cr_model, rev_per_dv_anova, cr_anova, rev_per_dv_model_dma, cr_model_dma, rev_per_dv_anova_dma, cr_anova_dma
from output_format_module import number_format, rename_column, output_format_reverse

# Streamlit app
def main():

    st.set_page_config(page_title="PPM Analysis -Reverse",layout='wide')

    st.title('Product Feature Preference Model - Reverse')

    st.markdown('''### Step 1: Define the Overall Analysis Scope''')
    st.write('''**Which region** shows a higher preference for the selected **Products/Features**? Measured by which **metric**?''')

    col1, col2, col3 = st.columns(3)
    with col1:
        market = st.selectbox("Select Market to Analyze", ["US"], 
                                       help = '''Market you need to analyse''',
                                       index=0)
    with col2:
        analysed_level = st.selectbox("Select Granularity for Region", ["Regional Level",'DMA Level'], 
                                       help = '''Granularity for region''',
                                       index=0)
    with col3:
        metric_analysed = st.selectbox("Select Metric to Analyze", ["Rev per DV", "CR"], 
                                       help = '''Metric you need to analyse  \n -> Rev per DV: Revenue/Detailviews in the analysed region (= Total Revenue/Total Detailviews)  \n -> CR: Conversion rate in the analysed region (= Total Order/Total Detailviews)''',
                                       index=0)
  
    col4, col5 = st.columns(2)
    with col4:
        start_date = st.date_input("Select Start Date", datetime(datetime.now().year, 1, 1), help = '''State the start date for your analysis''')
    with col5:
        end_date = st.date_input("Select End Date", help = '''State the end date for your analysis''')
    if end_date < start_date:
        st.error("End date must be later than start date. Please select a valid end date.")
    else:
        params = {'start_date': start_date, 'end_date': end_date}

    col6, col7 = st.columns(2)
    with col6:
        feature1 = st.selectbox("Select Feature Dimension 1", ["category", "subcategory", "collection","market_spu", "market_sku", "color_tone", "material_helper"], 
                               help = '''Product feature you need to analyse''',
                               index=0)
    with col7:
        if feature1 == "category":
            feature2 = st.selectbox("Select Feature Dimension 2 (Optional)", ["No Selection", "market_sku", "market_spu", "subcategory", "collection", "color_tone", "material_helper"], 
                               help = '''Product feature you need to analyse''',
                               index=0)
        elif feature1 == "subcategory":
            feature2 = st.selectbox("Select Feature Dimension 2 (Optional)", ["No Selection", "market_sku", "market_spu", "collection", "color_tone", "material_helper"], 
                               help = '''Feature dimension 1 you need to analyse''',
                               index=0)
        elif feature1 == "collection":
            feature2 = st.selectbox("Select Feature Dimension 2 (Optional)", ["No Selection", "category", "subcategory" ,"market_sku", "market_spu", "color_tone", "material_helper"], 
                               help = '''Feature dimension 2 you need to analyse''',
                               index=0)
        elif feature1 == "market_spu":
            feature2 = st.selectbox("Select Feature Dimension 2 (Optional)", ['No Selection', 'market_sku'], 
                               help = '''Product feature you need to analyse''',
                               index=0)
        elif feature1 in ["color_tone","material_helper"]:
            feature2 = st.selectbox("Select Feature Dimension 2 (Optional)", ['No Selection','market_sku'], 
                               help = '''Product feature you need to analyse''',
                               index=0)
        elif feature1 == "market_sku":
            feature2 = st.selectbox("Select Feature Dimension 2 (Optional)", ["No Selection"], 
                               help = '''Product feature you need to analyse''',
                               index=0)
    
    col8, col9 = st.columns(2)
    with col8:
        info = pd.read_csv('Product_Info.csv')
        us_info = info[info['market_sku'].str.contains("US-")]
        au_info = info[info['market_sku'].str.contains("AU-")]
        if market == 'US':
            specify1 = st.multiselect("Specify the Feature for Dimension 1", us_info[feature1].unique(),
                                       help = '''The specific feature you need to analyse for product dimension 1''',
                                       placeholder = "Leaving blank here for choosing all options.")
            if not specify1:
                specify1 = us_info[feature1].unique()
        elif market == 'AU':
            specify1 = st.multiselect("Specify the Feature for Dimension 1", au_info[feature1].unique(),
                                       help = '''The specific feature you need to analyse for product dimension 1''',
                                       placeholder = "Leaving blank here for choosing all options.")
            if not specify1:
                specify1 = au_info[feature1].unique()

    with col9:
        if feature2 == "No Selection":
            specify2 = st.multiselect("Specify the Feature for Dimension 2 (Optional)", ["No Selection"],
                                       help = '''The specific feature you need to analyse for product dimension 2''')
        else:
            if market == 'US':
                specify2 = st.multiselect("Specify the Feature for Dimension 2 (Optional)", us_info[feature2].unique(),
                                       help = '''The specific feature you need to analyse for product dimension 2''',
                                       placeholder = "Leaving blank here for choosing all options.")
                if not specify2:
                    specify2 = us_info[feature2].unique()
            if market == 'AU':
                specify2 = st.multiselect("Specify the Feature for Dimension 2 (Optional)", au_info[feature2].unique(),
                                       help = '''The specific feature you need to analyse for product dimension 2''',
                                       placeholder = "Leaving blank here for choosing all options.")
                if not specify2:
                    specify2 = au_info[feature2].unique()
    
    st.divider()

    st.markdown('''### Step 2: Set Thresholds for Control Metrics''')
    st.write('''To ensure a robust analysis, set minimum thresholds for various control metrics to filter products based on their data size and performance.''')

    col7,col8, col9, col10 = st.columns(4)
    with col7:
        metric_control_1 = 'us_total_cost_per_sku'
        metric_threshold_1 = st.number_input("Threshold for US Average Cost per SKU >=", min_value = 0, value=100, step=1,
                                             help = '''Set the minimum cost per sku in the US market  \n -> E.G., if analyzing by category, this threshold represents the average costs per SKU spent on your seleted category at US country level within your specified date range  \n Adjust this value based on your selected features and date range:  \n -> Example reference: at least $100 per sku per month
                                             ''',
                                             key="us_total_cost_threshold_per_sku")
    with col8:
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
    with col9:
        metric_control_3 = 'total_order'
        metric_threshold_3 = st.number_input("Threshold for Total Order >=", min_value = 0, value=10, step=1, 
                                             help = '''Set the minimum orders for products in the analysed region  \n -> Example reference: Total Order >= 10''',
                                             key="total_order_threshold")
    with col10:
        metric_control_4 = 'average_price_analysed'
        metric_threshold_4 = st.number_input("Threshold for Average Price >=", min_value = 0, value=100, step=1, 
                                             help = '''Set the minimum orders for products in the analysed region  \n -> Average Price = Total Revenue/Total Quantity  \n -> Example reference: Total Order >= 100''',
                                             key="average_price_analysed_threshold")
                                                   

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
                      AND fs2.market = 'US'
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
                      WHERE frpdom.market = 'US'
                         AND frpdom.date BETWEEN %s AND %s
                         and ds.category != 'Swatch'
                         and ds.category IS NOT NULL
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
                      AND fs2.market = 'US'
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
                          frpdom.market = 'US'
                          AND frpdom.date BETWEEN %s AND %s
                          AND ds.category != 'Swatch'
                          AND ds.category IS NOT NULL
                          AND frpdom.metro in ('New York, NY','Los Angeles CA','Washington DC (Hagerstown MD)','San Francisco-Oakland-San Jose CA','Seattle-Tacoma WA')
                      GROUP BY
                          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13
                      """
        
       us_cost_query = """
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
                      sum(fpam.cost) as us_total_cost
                      from fact_product_ads_metric fpam
                      LEFT JOIN dim_sku ds ON fpam.market_sku = ds.market_sku
                      LEFT JOIN ( SELECT market_sku, value
                      FROM dim_sku_property
                      WHERE property_type = 'Material Helper') dskp_material_helper ON ds.market_sku = dskp_material_helper.market_sku
                      LEFT JOIN ( SELECT market_spu, value
                      FROM dim_spu_property
                      WHERE property_type = 'Material Helper') dspp_material_helper ON ds.market_spu = dspp_material_helper.market_spu
                      where fpam.date between %s AND %s
                      and fpam.market = 'US'
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
       if analysed_level == 'Regional Level' and feature2 == 'No Selection':
           # fetch data
           df_order = fetch_data_from_database(conn, order_query_region, params)
           df_dv = fetch_data_from_database(conn, dv_query_region, params)
           df_us_cost = fetch_data_from_database(conn, us_cost_query, params)
           conn.close()
           df_merge = merge_and_fill(df_order,df_dv,market,analysed_level)

           #create key
           df_merge['cate-feature'] = df_merge['category'] + ": " + df_merge[feature1]

           # Calculate order percent
           df_merge['order_percent_of_total'] = df_merge.groupby('region')['total_order'].transform(lambda x: x / x.sum())
           df_merge['order_percent_of_category'] = df_merge.groupby(['region','category'])['total_order'].transform(lambda x: x / x.sum())
           
           # Filter out required data
           df_merge = df_merge[(df_merge[feature1].isin(specify1))]

           # raise error message if needed
           if df_merge.empty == True:
               st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1.")
               # Stop program execution
               raise SystemExit("Program halted due to empty DataFrame.")
           else:
               pass
       elif analysed_level == 'Regional Level' and feature2 is not 'No Selection':
           # fetch data
           df_order = fetch_data_from_database(conn, order_query_region, params)
           df_dv = fetch_data_from_database(conn, dv_query_region, params)
           df_us_cost = fetch_data_from_database(conn, us_cost_query, params)
           conn.close()
           df_merge = merge_and_fill(df_order,df_dv,market,analysed_level)

           #create key
           df_merge['cate-feature'] = df_merge['category'] + ": " + df_merge[feature2]

           # Calculate order percent
           df_merge['order_percent_of_total'] = df_merge.groupby('region')['total_order'].transform(lambda x: x / x.sum())
           df_merge['order_percent_of_category'] = df_merge.groupby(['region','category'])['total_order'].transform(lambda x: x / x.sum())
           
           # Filter out required data
           df_merge = df_merge[(df_merge[feature1].isin(specify1)) &
                               (df_merge[feature2].isin(specify2)) ]

           # raise error message if needed
           if df_merge.empty == True:
               st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1.")
               # Stop program execution
               raise SystemExit("Program halted due to empty DataFrame.")
           else:
               pass
           
       elif analysed_level == 'DMA Level' and feature2 == 'No Selection':
           # fetch data
           df_order_analysed = fetch_data_from_database(conn, order_query_dma, params)
           df_order_baseline = fetch_data_from_database(conn, order_query_region, params)
           df_dv_analysed = fetch_data_from_database(conn, dv_query_dma, params)
           df_dv_baseline = fetch_data_from_database(conn, dv_query_region, params)
           df_us_cost = fetch_data_from_database(conn, us_cost_query, params)
           conn.close()
           df_merge_analysed = merge_and_fill(df_order_analysed,df_dv_analysed,market,analysed_level,include=0)
           df_merge_baseline = merge_and_fill(df_order_baseline,df_dv_baseline,market,analysed_level,include=1)

           #create key
           df_merge_analysed['cate-feature'] = df_merge_analysed['category'] + ": " + df_merge_analysed[feature1]
           df_merge_baseline['cate-feature'] = df_merge_baseline['category'] + ": " + df_merge_baseline[feature1]

           # Calculate order percent
           df_merge_analysed['order_percent_of_total'] = df_merge_analysed.groupby('dma')['total_order'].transform(lambda x: x / x.sum())
           df_merge_analysed['order_percent_of_category'] = df_merge_analysed.groupby(['dma','category'])['total_order'].transform(lambda x: x / x.sum())

           # Filter out required data
           df_merge_analysed = df_merge_analysed[(df_merge_analysed[feature1].isin(specify1))]
           df_merge_baseline = df_merge_baseline[(df_merge_baseline[feature1].isin(specify1))]
            
            # raise error message if needed
           if df_merge_analysed.empty == True or df_merge_baseline.empty == True:
               st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1.")
               # Stop program execution
               raise SystemExit("Program halted due to empty DataFrame.")
           else:
               pass
       elif analysed_level == 'DMA Level' and feature2 is not 'No Selection':
           # fetch data
           df_order_analysed = fetch_data_from_database(conn, order_query_dma, params)
           df_order_baseline = fetch_data_from_database(conn, order_query_region, params)
           df_dv_analysed = fetch_data_from_database(conn, dv_query_dma, params)
           df_dv_baseline = fetch_data_from_database(conn, dv_query_region, params)
           df_us_cost = fetch_data_from_database(conn, us_cost_query, params)
           conn.close()
           df_merge_analysed = merge_and_fill(df_order_analysed,df_dv_analysed,market,analysed_level,include=0)
           df_merge_baseline = merge_and_fill(df_order_baseline,df_dv_baseline,market,analysed_level,include=1)

           #create key
           df_merge_analysed['cate-feature'] = df_merge_analysed['category'] + ": " + df_merge_analysed[feature2]
           df_merge_baseline['cate-feature'] = df_merge_baseline['category'] + ": " + df_merge_baseline[feature2]

           # Calculate order percent
           df_merge_analysed['order_percent_of_total'] = df_merge_analysed.groupby('dma')['total_order'].transform(lambda x: x / x.sum())
           df_merge_analysed['order_percent_of_category'] = df_merge_analysed.groupby(['dma','category'])['total_order'].transform(lambda x: x / x.sum())

           # Filter out required data
           df_merge_analysed = df_merge_analysed[(df_merge_analysed[feature1].isin(specify1)) &
                               (df_merge_analysed[feature2].isin(specify2))]
           df_merge_baseline = df_merge_baseline[(df_merge_baseline[feature1].isin(specify1)) &
                               (df_merge_baseline[feature2].isin(specify2)) ]
            
            # raise error message if needed
           if df_merge_analysed.empty == True or df_merge_baseline.empty == True:
               st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1.")
               # Stop program execution
               raise SystemExit("Program halted due to empty DataFrame.")
           else:
               pass
        
       if metric_analysed == 'Rev per DV':
            
            results_df = pd.DataFrame()
            anova_df = pd.DataFrame()

            # Call comparison model
            if analysed_level == 'Regional Level':
                region_list = ["US West", "US East", "US Southeast", "US Northwest"]
                for region_analysed in region_list:
                    output_rev_per_dv = rev_per_dv_model(df_merge, df_us_cost, region_analysed, feature1, feature2,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2,metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4)
                    # Add region info
                    output_rev_per_dv['Region'] = region_analysed
                    #expand result to dataframe
                    results_df = pd.concat([results_df, output_rev_per_dv], ignore_index=True)

            elif analysed_level == 'DMA Level':
                region_list = ["New York, NY","Los Angeles CA","Washington DC (Hagerstown MD)","San Francisco-Oakland-San Jose CA","Seattle-Tacoma WA"]
                for region_analysed in region_list:
                    output_rev_per_dv = rev_per_dv_model_dma(df_us_cost, region_analysed, feature1,feature2,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2,metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4,
                            df_merge_analysed, df_merge_baseline)
                    # Add region info
                    output_rev_per_dv['DMA'] = region_analysed
                    #append result to dataframe
                    results_df = pd.concat([results_df, output_rev_per_dv], ignore_index=True)

            if results_df.empty == True:
                st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1, or adjusting thresholds in step 2.")
                # Stop program execution
                raise SystemExit("Program halted due to empty DataFrame.")
            else:
                pass

            #ANOVA
            if analysed_level == 'Regional Level':
                region_list = ["US West", "US East", "US Southeast", "US Northwest"]
                for region_analysed in region_list:
                    anova_each_rev_per_dv = rev_per_dv_anova(df_merge, results_df, region_analysed)
                    # Add region info
                    anova_each_rev_per_dv['Region'] = region_analysed
                    #append result to dataframe
                    anova_df = pd.concat([anova_df,anova_each_rev_per_dv],ignore_index = True)

            elif analysed_level == 'DMA Level':
                region_list = ["New York, NY","Los Angeles CA","Washington DC (Hagerstown MD)","San Francisco-Oakland-San Jose CA","Seattle-Tacoma WA"]
                for region_analysed in region_list:
                    anova_each_rev_per_dv = rev_per_dv_anova_dma(results_df, region_analysed, df_merge_analysed, df_merge_baseline)
                    # Add region info
                    anova_each_rev_per_dv['DMA'] = region_analysed
                    #append result to dataframe
                    anova_df = pd.concat([anova_df, anova_each_rev_per_dv],ignore_index = True)
            
            # Get final output
            if analysed_level == 'Regional Level':
                final_df = pd.merge(results_df, anova_df[['cate-feature','Region', 'p_value']], on=['cate-feature','Region'], how = 'left')
            elif analysed_level == 'DMA Level':
                final_df = pd.merge(results_df, anova_df[['cate-feature','DMA', 'p_value']], on=['cate-feature','DMA'], how = 'left')
            final_df = final_df.sort_values(by=['cate-feature','weighted_score'], ascending= False)
            final_df = output_format_reverse(final_df, feature1, feature2)
            st.table(final_df)

            # Allow users to download the data as CSV
            csv_data = final_df.to_csv(index=False)
            st.download_button(label="Download Data as CSV", data=csv_data, 
                               file_name=f"Feature1:{feature1}_Feature2:{feature2}_{analysed_level}_Rev per DV_{start_date} to {end_date}.csv",
                               help = '''Press button to download output as csv file''',
                               key="download_data")
        
       elif metric_analysed == 'CR':
            
            results_df = pd.DataFrame()
            anova_df = pd.DataFrame()

            # Call comparison model
            if analysed_level == 'Regional Level':
                region_list = ["US West", "US East", "US Southeast", "US Northwest"]
                for region_analysed in region_list:
                    output_cr = cr_model(df_merge, df_us_cost, region_analysed, feature1, feature2,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2,metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4)
                    # Add region info
                    output_cr['Region'] = region_analysed
                    #expand result to dataframe
                    results_df = pd.concat([results_df, output_cr], ignore_index=True)

            elif analysed_level == 'DMA Level':
                region_list = ["New York, NY","Los Angeles CA","Washington DC (Hagerstown MD)","San Francisco-Oakland-San Jose CA","Seattle-Tacoma WA"]
                for region_analysed in region_list:
                    output_cr = cr_model_dma(df_us_cost, region_analysed, feature1,feature2,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2,metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4,
                            df_merge_analysed, df_merge_baseline)
                    # Add region info
                    output_cr['DMA'] = region_analysed
                    #append result to dataframe
                    results_df = pd.concat([results_df, output_cr], ignore_index=True)

            if results_df.empty == True:
                st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1, or adjusting thresholds in step 2.")
                # Stop program execution
                raise SystemExit("Program halted due to empty DataFrame.")
            else:
                pass

            #ANOVA
            if analysed_level == 'Regional Level':
                region_list = ["US West", "US East", "US Southeast", "US Northwest"]
                for region_analysed in region_list:
                    anova_each_cr = cr_anova(df_merge, results_df, region_analysed)
                    # Add region info
                    anova_each_cr['Region'] = region_analysed
                    #append result to dataframe
                    anova_df = pd.concat([anova_df, anova_each_cr],ignore_index = True)

            elif analysed_level == 'DMA Level':
                region_list = ["New York, NY","Los Angeles CA","Washington DC (Hagerstown MD)","San Francisco-Oakland-San Jose CA","Seattle-Tacoma WA"]
                for region_analysed in region_list:
                    anova_each_cr = cr_anova_dma(results_df, region_analysed, df_merge_analysed, df_merge_baseline)
                    # Add region info
                    anova_each_cr['DMA'] = region_analysed
                    #append result to dataframe
                    anova_df = pd.concat([anova_df, anova_each_cr],ignore_index = True)
            
            # Get final output
            if analysed_level == 'Regional Level':
                final_df = pd.merge(results_df, anova_df[['cate-feature','Region', 'p_value']], on=['cate-feature','Region'], how = 'left')
            elif analysed_level == 'DMA Level':
                final_df = pd.merge(results_df, anova_df[['cate-feature','DMA', 'p_value']], on=['cate-feature','DMA'], how = 'left')
            final_df = final_df.sort_values(by=['cate-feature','weighted_score'], ascending= False)
            final_df = output_format_reverse(final_df, feature1, feature2)
            st.table(final_df)

            # Allow users to download the data as CSV
            csv_data = final_df.to_csv(index=False)
            st.download_button(label="Download Data as CSV", data=csv_data, 
                               file_name=f"Feature1:{feature1}_Feature2:{feature2}_{analysed_level}_CR_{start_date} to {end_date}.csv",
                               help = '''Press button to download output as csv file''',
                               key="download_data")


if __name__ == "__main__":
    main()