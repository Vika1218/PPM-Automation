# Import packages
import streamlit as st
import psycopg2
import pandas as pd
import numpy as np
import pingouin as pg

# Import customized packages
from data_processing_module import connect_to_database, fetch_data_from_database, process_orders, process_detail_views, process_us_cost, rev_per_dv_model, cr_model, rev_per_dv_anova, cr_anova

# Streamlit app
def main():
    
    st.title("Product Feature MVP Automation_test")

    # User input fields
    metric_analysed = st.selectbox("Select Metric to Analyze", ["Rev per DV", "CR"], index=0)
    region_analysed = st.selectbox("Select Region to Analyze", ["US West", "US East", "US Southeast", "US Northwest"], index=0)
    region_baseline = st.selectbox("Select Baseline Region", ["US All", "US West", "US East", "US Southeast", "US Northwest"], index=0)
    feature = st.selectbox("Select Feature Dimension", ["market_sku", "market_spu", "category", "subcategory", "collection", "color_tone", "material_helper"], index=0)
    rank = st.radio("Select Rank", ["Top", "Bottom"], index=0)
    output_limit = st.slider("Select Output Limit", min_value=1, max_value=100, value=30)
    
    # Threshold for us_total_costs
    metric_control_1 = 'us_total_cost'
    metric_threshold_1 = st.number_input("Enter Threshold for US Total Cost >=", value=300.00, step=0.01,key="us_total_cost_threshold")

    # Metric analyzed threshold based on the user's selection
    if metric_analysed == 'Rev per DV':
        metric_control_2 = 'rev_per_dv_analysed'
        metric_threshold_2 = st.number_input(f"Enter Threshold for {metric_analysed}_analysed >=", value=0.00, step=0.01, key="metric_analyzed_threshold")
    elif metric_analysed == 'CR':
        metric_control_2 = 'CR_analysed'
        metric_threshold_2 = st.number_input(f"Enter Threshold for {metric_analysed}_analysed >=(%)    E.g.,For CR >= 0.08% -> Input number 0.08", value=0.00, step=0.01, format="%f", key="metric_analyzed_threshold")/ 100.0
    
    # Threshold for total_order
    metric_control_3 = 'total_order'
    metric_threshold_3 = st.number_input("Enter Threshold for Total Order >=", value=10, step=1, key="total_order_threshold")

    # Date range
    start_date = st.date_input("Select Start Date", value=pd.to_datetime("2023-10-01"))
    end_date = st.date_input("Select End Date", value=pd.to_datetime("2023-12-31"))
    params = {'start_date': start_date, 'end_date': end_date}

    # Trigger data processing on user input
    if st.button("Process Data"):
       # change the according info to connect to the database
       host = 'dw-prod.cfujfnms1rth.ap-southeast-1.redshift.amazonaws.com'
       database = 'dwd_prod'
       username = 'yunyi_cheng' #change if needed
       password = 'A00JbR&3' #change if needed
       port = 5439 #change if needed
       
       # order record: change the query if needed
       order_query = """
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
                      AND ds.category != 'Furniture Sets'
                      AND ds.category IS NOT NULL
                      AND ds.subcategory != 'Accident Protection Plan'
                      and DATE(fs2.payment_completion_time) BETWEEN %s AND %s
                      group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14
                      order by 1,3
                      """
       dv_query = """
                      select 
                      frpdom.market_sku || '_' || frpdom.region || '_' || frpdom.date ::text AS unique_id,
                      frpdom.date,
                      frpdom.market_sku,
                      frpdom.region,
                      sum(frpdom.product_view) as total_detailview
                      from fact_region_product_daily_onsite_metric frpdom 
                      where frpdom.market = 'US'
                      and frpdom.date between %s AND %s
                      group by 1,2,3,4
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
                      and ds.category != 'Furniture Sets'
                      group by 1,2,3,4,5,6,7,8,9,10
                      """
       
       # Connect to the database & extract data & merge dataset
       conn = connect_to_database(host, database, username, password, port)
       df_order = fetch_data_from_database(conn, order_query, params)
       df_dv = fetch_data_from_database(conn, dv_query, params)
       df_us_cost = fetch_data_from_database(conn, us_cost_query, params)
       conn.close()
       df_merge = pd.merge(df_order, df_dv[['unique_id', 'total_detailview']], on='unique_id', how='left')
       df_merge['cate-feature'] = df_merge['category'] + ": " + df_merge[feature]
        
       if metric_analysed == 'Rev per DV':
            
            # Call comparison model & ANOVA
            output_rev_per_dv = rev_per_dv_model(df_merge, df_us_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2, metric_control_3, metric_threshold_3)
            anova_each_rev_per_dv = rev_per_dv_anova(df_merge, output_rev_per_dv, region_analysed, region_baseline, feature)
            # Get final output
            output_rev_per_dv = pd.merge(output_rev_per_dv, anova_each_rev_per_dv[['cate-feature', 'p_value']], on='cate-feature', how = 'left')
            st.table(output_rev_per_dv)

            # Allow users to download the data as CSV
            csv_data = output_rev_per_dv.to_csv(index=False)
            st.download_button(label="Download Data as CSV", data=csv_data, file_name="output_rev_per_dv.csv", key="download_data")
        
       elif metric_analysed == 'CR':
            
            # Call comparison model & ANOVA
            output_cr = cr_model(df_merge, df_us_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2, metric_control_3, metric_threshold_3)
            anova_each_cr = cr_anova(df_merge, output_cr, region_analysed, region_baseline, feature)
            
            # Get final output
            output_cr = pd.merge(output_cr, anova_each_cr[['cate-feature', 'p_value']], on='cate-feature', how = 'left')
            st.table(output_cr)

            # Allow users to download the data as CSV
            csv_data = output_cr.to_csv(index=False)
            st.download_button(label="Download Data as CSV", data=csv_data, file_name="output_cr.csv", key="download_data")


if __name__ == "__main__":
    main()