# Import packages
import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import pingouin as pg
from datetime import datetime

# Import customized packages
from country_fact_process import connect_to_database, fetch_data_from_database, process_orders, process_detail_views, process_country_cost, merge_and_fill, output_model
from output_format_module import number_format, rename_column, output_format_reverse

# Streamlit app
def main():

    st.set_page_config(page_title="Country Fact Table",layout='wide')

    st.title('Country Fact Table')

    st.markdown('''### Step 1: Define the Overall Analysis Scope''')
    st.write('''What's the **performance** of the product at **country level**? Ranked by which **metric**?''')

    col1, col2 = st.columns(2)
    with col1:
        market = st.selectbox("Select Market to Analyze", ["US",'AU'], 
                                       help = '''Market you need to analyse''',
                                       index=0)
    with col2:
        metric = st.selectbox("Select Metric to Rank Output", ["Rev per DV", "CR"], 
                                       help = '''Rank the output by metric''',
                                       index=0)
  
    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input("Select Start Date", datetime(datetime.now().year, 1, 1), help = '''State the start date for your analysis''')
    with col4:
        end_date = st.date_input("Select End Date", help = '''State the end date for your analysis''')
    if end_date < start_date:
        st.error("End date must be later than start date. Please select a valid end date.")
    else:
        params = {'market': market,'start_date': start_date, 'end_date': end_date}

    col5, col6 = st.columns(2)
    with col5:
        feature1 = st.selectbox("Select Feature Dimension 1", ["category", "subcategory", "collection","market_spu", "market_sku", "color_tone", "material_helper"], 
                               help = '''Product feature you need to analyse''',
                               index=0)
    with col6:
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
    
    col7, col8 = st.columns(2)
    with col7:
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

    with col8:
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
    st.write('''Set minimum thresholds for various control metrics to filter products based on their data size and performance.''')

    col7,col8, col9, col10 = st.columns(4)
    with col7:
        metric_control_1 = 'country_total_cost_per_sku'
        metric_threshold_1 = st.number_input(f"Threshold for {market} Ave. Cost per SKU >=", min_value = 0, value=100, step=1,
                                             help = f'''Set the minimum cost per sku in the {market} market  \n -> E.G., if analyzing by category, this threshold represents the average costs per SKU spent on your seleted category at {market} country level within your specified date range  \n Adjust this value based on your selected features and date range:  \n -> Example reference: at least $100 per sku per month
                                             ''',
                                             key="country_total_cost_threshold_per_sku")
    with col8:
        if metric == 'Rev per DV':
            metric_control_2 = 'rev_per_dv'
            metric_threshold_2 = st.number_input(f"Threshold for {metric} >=", min_value = 0.0, value=0.0, step=0.1, 
                                                 help = '''Set the minimum Rev per DV for products  \n -> Example reference: Rev per DV >= 2.00''',
                                                 key="metric_analyzed_threshold")
        elif metric == 'CR':
            metric_control_2 = 'CR'
            metric_threshold_2 = st.number_input(f"Threshold for {metric} >=(%)", min_value = 0.00, value=0.00, step=0.01, 
                                                 help = '''Set the minimum CR for products  \n -> The format is already in percentage  \n -> Example reference: CR >= 0.08% --> Enter 0.08 here''', 
                                                 key="metric_analyzed_threshold")/ 100.0
    with col9:
        metric_control_3 = 'total_order'
        metric_threshold_3 = st.number_input("Threshold for Total Order >=", min_value = 0, value=10, step=1, 
                                             help = '''Set the minimum orders for products  \n -> Example reference: Total Order >= 10''',
                                             key="total_order_threshold")
    with col10:
        metric_control_4 = 'average_price'
        metric_threshold_4 = st.number_input("Threshold for Average Price >=", min_value = 0, value=100, step=1, 
                                             help = '''Set the minimum orders for products  \n -> Average Price = Total Revenue/Total Quantity  \n -> Example reference: Total Order >= 100''',
                                             key="average_price_threshold")
                                                   

    # Trigger data processing on user input
    if st.button("Process Data",help = '''Press button to proceed the analysis'''):
       # change the according info to connect to the database
       host = 'dw-prod.cfujfnms1rth.ap-southeast-1.redshift.amazonaws.com'
       database = 'dwd_prod'
       username = 'yunyi_cheng' #change if needed
       password = 'A00JbR&3' #change if needed
       port = 5439 #change if needed
       
       # order record: change the query if needed
       order_query = """
                      SELECT
                      fs3.market_sku || '_' ||date(fs2.payment_completion_time)::text AS unique_id,
                      date(fs2.payment_completion_time) as order_date,
                      fs2.market,
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
                      group by 1,2,3,4,5,6,7,8,9,10,11,12,13
                      order by 1,2
                      """
       dv_query = """
                      SELECT 
                      frpdom.market_sku || '_' || frpdom.date::TEXT AS unique_id,
                      frpdom.date,
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
                      SUM(frpdom.product_view) AS total_detailview,
                      SUM(frpdom.add_to_cart) AS atc
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
                      GROUP BY 1, 2, 3, 4,5,6,7,8,9,10,11,12
                      """
        
       country_cost_query = """
                      select 
                      fpam.market_sku || '_' || fpam.date::TEXT AS unique_id,
                      fpam.date,
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
                      group by 1,2,3,4,5,6,7,8,9,10,11,12
                      """
       
       # Connect to the database & extract data & merge dataset
       conn = connect_to_database(host, database, username, password, port)
       try:
           conn = connect_to_database(host, database, username, password, port)
       except psycopg2.OperationalError as e:
           st.error("Error connecting to the database: {}. Try it later.".format(e))
           # Stop program execution
           raise SystemExit("Error connecting to the database")
       
       if feature2 == 'No Selection':
           # fetch data
           df_order = fetch_data_from_database(conn, order_query, params)
           df_dv = fetch_data_from_database(conn, dv_query, params)
           df_country_cost = fetch_data_from_database(conn, country_cost_query, params)
           conn.close()
           df_merge = merge_and_fill(df_order,df_dv,df_country_cost, market)

           #create key
           df_merge['cate-feature'] = df_merge['category'] + ": " + df_merge[feature1]

           # Calculate order percent
           df_merge['order_percent_of_total'] = df_merge['total_order'].transform(lambda x: x / x.sum())
           df_merge['order_percent_of_category'] = df_merge.groupby('category')['total_order'].transform(lambda x: x / x.sum())
           
           # Filter out required data
           df_merge = df_merge[(df_merge[feature1].isin(specify1))]

           # raise error message if needed
           if df_merge.empty == True:
               st.error("Your current selections did not yield any data matches. Please consider expanding the date range in step 1.")
               # Stop program execution
               raise SystemExit("Program halted due to empty DataFrame.")
           else:
               pass
       elif feature2 is not 'No Selection':
           # fetch data
           df_order = fetch_data_from_database(conn, order_query, params)
           df_dv = fetch_data_from_database(conn, dv_query, params)
           df_country_cost = fetch_data_from_database(conn, country_cost_query, params)
           conn.close()
           df_merge = merge_and_fill(df_order,df_dv,df_country_cost, market)

           #create key
           df_merge['cate-feature'] = df_merge['category'] + ": " + df_merge[feature2]

           # Calculate order percent
           df_merge['order_percent_of_total'] = df_merge['total_order'].transform(lambda x: x / x.sum())
           df_merge['order_percent_of_category'] = df_merge.groupby('category')['total_order'].transform(lambda x: x / x.sum())
           
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
           
       # Final output
       output = output_model(df_merge, metric, feature1, feature2,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2,metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4)
       final_output = output_format_reverse(output, feature1, feature2)
       
       st.table(final_output)

       # Allow users to download the data as CSV
       csv_data = final_output.to_csv(index=False)
       st.download_button(label="Download Data as CSV", data=csv_data, 
                               file_name=f"{feature1}_{feature2}_{start_date} to {end_date}.csv",
                               help = '''Press button to download output as csv file''',
                               key="download_data")
       
if __name__ == "__main__":
    main()

       