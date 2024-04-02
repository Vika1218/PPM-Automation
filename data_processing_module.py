import psycopg2
import pandas as pd
import numpy as np
import pingouin as pg


def connect_to_database(host, database, username, password, port):
    # Connect to the database and return the connection
    conn = psycopg2.connect(host=host, database=database, user=username, password=password, port=port)
    return conn

def fetch_data_from_database(conn, query, params):
    # Execute the query and fetch the data
    cursor = conn.cursor()
    cursor.execute(query, tuple(params.values()))
    raw_data = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(raw_data, columns=column_names)
    return df

def process_orders(conn, order_query, params):
    # Process order data
    df_order = fetch_data_from_database(conn, order_query, params)
    return df_order

def process_detail_views(conn, dv_query, params):
    # Process detail views data
    df_dv = fetch_data_from_database(conn, dv_query, params)
    return df_dv

def process_us_cost(conn, us_cost_query, params):
    # Process US cost data
    df_us_cost = fetch_data_from_database(conn, us_cost_query, params)
    return df_us_cost

def merge_and_fill(df_order, df_dv, market, analysed_level,include=None):
    # Get unique values of 'unique_id'
    id = pd.concat([df_order['unique_id'], df_dv['unique_id']]).unique()
    
    # Create a new DataFrame with unique 'unique_id'
    df_merge = pd.DataFrame({'unique_id': id})
    
    # Merge 'df_merge' with 'df_order' and 'df_dv'
    if analysed_level == 'Regional Level':
        df_merge = pd.merge(df_merge, df_order[['unique_id', 'order_date', 'market', 'region', 'market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper', 'total_revenue', 'total_quantity', 'total_order']], on='unique_id', how='left')
        df_merge = pd.merge(df_merge, df_dv[['unique_id', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper', 'total_detailview']], on='unique_id', how='left')
        columns_to_fill = ['sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']
    elif analysed_level == 'DMA Level':
        if include == 0: #no region in column
            df_merge = pd.merge(df_merge, df_order[['unique_id', 'order_date', 'market', 'dma', 'market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper', 'total_revenue', 'total_quantity', 'total_order']], on='unique_id', how='left')
            df_merge = pd.merge(df_merge, df_dv[['unique_id','sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper', 'total_detailview']], on='unique_id', how='left')
            columns_to_fill = ['sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']
        elif include == 1: #have region in column (for baseline is regional level)
            df_merge = pd.merge(df_merge, df_order[['unique_id', 'order_date', 'market', 'region', 'market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper', 'total_revenue', 'total_quantity', 'total_order']], on='unique_id', how='left')
            df_merge = pd.merge(df_merge, df_dv[['unique_id','sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper', 'total_detailview']], on='unique_id', how='left')
            columns_to_fill = ['sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']    
    
    # Fill missing values
    for column in columns_to_fill:
        df_merge[column] = df_merge[column+ '_x'].fillna(df_merge[column + '_y'])
    
    # Drop redundant columns '_x' and '_y' after filling missing values
    if analysed_level == 'Regional Level':
        df_merge.drop(columns=['sku_name_x', 'market_spu_x', 'spu_name_x', 'master_category_x', 'category_x', 'subcategory_x', 'collection_x', 'color_tone_x', 'material_helper_x', 'sku_name_y', 'market_spu_y', 'spu_name_y', 'master_category_y', 'category_y', 'subcategory_y', 'collection_y', 'color_tone_y', 'material_helper_y'], inplace=True)
        # Fill in order date, sku, region
        df_merge.loc[df_merge['order_date'].isnull(), 'order_date'] = df_merge.loc[df_merge['order_date'].isnull(), 'unique_id'].str.split('_').str[2]
        df_merge.loc[df_merge['region'].isnull(), 'region'] = df_merge.loc[df_merge['region'].isnull(), 'unique_id'].str.split('_').str[1]
        df_merge.loc[df_merge['market_sku'].isnull(), 'market_sku'] = df_merge.loc[df_merge['market_sku'].isnull(), 'unique_id'].str.split('_').str[0]
    elif analysed_level == 'DMA Level':
        if include == 0:#no region in column
            df_merge.drop(columns=['sku_name_x', 'market_spu_x', 'spu_name_x', 'master_category_x', 'category_x', 'subcategory_x', 'collection_x', 'color_tone_x', 'material_helper_x', 'sku_name_y', 'market_spu_y', 'spu_name_y', 'master_category_y', 'category_y', 'subcategory_y', 'collection_y', 'color_tone_y', 'material_helper_y'], inplace=True)
            # Fill in order date, sku, dma
            df_merge.loc[df_merge['order_date'].isnull(), 'order_date'] = df_merge.loc[df_merge['order_date'].isnull(), 'unique_id'].str.split('_').str[2]
            df_merge.loc[df_merge['dma'].isnull(), 'dma'] = df_merge.loc[df_merge['dma'].isnull(), 'unique_id'].str.split('_').str[1]
            df_merge.loc[df_merge['market_sku'].isnull(), 'market_sku'] = df_merge.loc[df_merge['market_sku'].isnull(), 'unique_id'].str.split('_').str[0]
        elif include == 1: #have region in column (for baseline is regional level)
            df_merge.drop(columns=['sku_name_x', 'market_spu_x', 'spu_name_x', 'master_category_x', 'category_x', 'subcategory_x', 'collection_x', 'color_tone_x', 'material_helper_x', 'sku_name_y', 'market_spu_y', 'spu_name_y', 'master_category_y', 'category_y', 'subcategory_y', 'collection_y', 'color_tone_y', 'material_helper_y'], inplace=True)
            # Fill in order date, sku, dma
            df_merge.loc[df_merge['order_date'].isnull(), 'order_date'] = df_merge.loc[df_merge['order_date'].isnull(), 'unique_id'].str.split('_').str[2]
            df_merge.loc[df_merge['region'].isnull(), 'region'] = df_merge.loc[df_merge['region'].isnull(), 'unique_id'].str.split('_').str[1]
            df_merge.loc[df_merge['market_sku'].isnull(), 'market_sku'] = df_merge.loc[df_merge['market_sku'].isnull(), 'unique_id'].str.split('_').str[0]

    # Fill in missing values for 'market', 'total_detailview', 'total_revenue', 'total_quantity', 'total_order'
    df_merge['market'] = df_merge['market'].fillna(market)
    df_merge['total_detailview'] = df_merge['total_detailview'].fillna(0)
    df_merge['total_revenue'] = df_merge['total_revenue'].fillna(0)
    df_merge['total_quantity'] = df_merge['total_quantity'].fillna(0)
    df_merge['total_order'] = df_merge['total_order'].fillna(0)

    return df_merge

def rev_per_dv_model(df_merge, df_us_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2,metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4):
    
    # Extract subset of dataset
    if region_baseline == 'US All':
        data_analysed = df_merge[df_merge['region'] == region_analysed]
        data_baseline = df_merge.copy()
    else:
        data_analysed = df_merge[df_merge['region'] == region_analysed]
        data_baseline = df_merge[df_merge['region'] == region_baseline]
    
    # Group and aggregate data for data_analysed
    if feature == 'category':
        grouped_data_analysed = data_analysed.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
        'order_percent_of_total':'sum'
        }).reset_index()
        
        grouped_data_analysed = grouped_data_analysed.rename(columns={'order_percent_of_total': 'order_percent'})
    else:
        grouped_data_analysed = data_analysed.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
        'order_percent_of_category':'sum'
        }).reset_index()
        
        grouped_data_analysed = grouped_data_analysed.rename(columns={'order_percent_of_category': 'order_percent'})

    grouped_data_analysed = grouped_data_analysed.rename(columns={'market_sku': 'sku_count'})
    grouped_data_analysed['category'] = grouped_data_analysed['cate-feature'].str.split(': ').str[0]
    grouped_data_analysed[feature] = grouped_data_analysed['cate-feature'].str.split(': ').str[1]

    # Group and aggregate data for data_baseline
    grouped_data_baseline = data_baseline.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
    }).reset_index()

    grouped_data_baseline = grouped_data_baseline.rename(columns={'market_sku': 'sku_count'})
    grouped_data_baseline['category'] = grouped_data_baseline['cate-feature'].str.split(': ').str[0]
    grouped_data_baseline[feature] = grouped_data_baseline['cate-feature'].str.split(': ').str[1]
    
    # Calculate the metrics & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['rev_per_dv_analysed'] = grouped_data_analysed['total_revenue'] / grouped_data_analysed['total_detailview']
        grouped_data_baseline['rev_per_dv_baseline'] = grouped_data_baseline['total_revenue'] / grouped_data_baseline['total_detailview']

    # Calculate average price & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['average_price_analysed'] = grouped_data_analysed['total_revenue'] / grouped_data_analysed['total_quantity']
        grouped_data_baseline['average_price_baseline'] = grouped_data_baseline['total_revenue'] / grouped_data_baseline['total_quantity']

    # Merge baseline grouped dataset's rev_per_dv column into analysed grouped dataset
    merged_data = pd.merge(grouped_data_analysed, grouped_data_baseline[['cate-feature', 'rev_per_dv_baseline']],
                           on='cate-feature', how='left')

    # Replace NaN and inf with 0
    merged_data.fillna(0, inplace=True)
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Calculate metric diff
    merged_data['metric_diff_percent'] = np.where(
    merged_data['rev_per_dv_baseline'] == 0,
    np.nan,  # Replace division by zero with NaN
    (merged_data['rev_per_dv_analysed'] / merged_data['rev_per_dv_baseline']) - 1)
    
    # Calculate weighted score
    merged_data['weighted_score'] = merged_data['metric_diff_percent'] * merged_data['order_percent']
    
    # Group & Merge products' US cost
    df_us_cost['cate-feature'] = df_us_cost['category'] + ": " + df_us_cost[feature]
    grouped_us_cost = df_us_cost.groupby('cate-feature').agg({
        'us_total_cost': 'sum'
    }).reset_index()
    
    merged_data = pd.merge(merged_data, grouped_us_cost[['cate-feature','us_total_cost']],
                           on='cate-feature', how='left')

    # Filter the output according to the conditions & output limit
    filtered_output = merged_data[
        (merged_data[metric_control_1] >= metric_threshold_1) &
        (merged_data[metric_control_2] >= metric_threshold_2) &
        (merged_data[metric_control_3] >= metric_threshold_3) &
        (merged_data[metric_control_4] >= metric_threshold_4)
    ]
    
    # Rank the output
    if rank == 'Top':
        ranked_output = filtered_output.sort_values(by='weighted_score', ascending= False)
    elif rank == 'Bottom':
        ranked_output = filtered_output.sort_values(by='weighted_score', ascending= True)

    # Get the sku info (in case need to add product name or more features in the output)
    sku_info = df_merge[['market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']]
    sku_info = sku_info.drop_duplicates()
    
    # Format the output
    if feature == 'market_sku':
        ranked_output = pd.merge(ranked_output, sku_info[[feature, 'sku_name']], on=feature, how='left')
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'sku_name','sku_count', 'category', 'total_revenue', 'total_quantity', 'average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'rev_per_dv_analysed', 'rev_per_dv_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    elif feature == 'market_spu':
        ranked_output = pd.merge(ranked_output, sku_info[[feature, 'spu_name']], on=feature, how='left')
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'spu_name', 'sku_count', 'category', 'total_revenue', 'total_quantity', 'average_price_analysed','total_order', 'total_detailview', 'us_total_cost',
                        'rev_per_dv_analysed', 'rev_per_dv_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    elif feature == 'category':
        ranked_output = ranked_output.loc[:, ~ranked_output.columns.duplicated(keep='first')]
        ranked_output = ranked_output.reindex(columns=['cate-feature', 'category', 'sku_count', 'total_revenue', 'total_quantity', 'average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'rev_per_dv_analysed', 'rev_per_dv_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    else:
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'sku_count', 'category',  'total_revenue', 'total_quantity', 'average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'rev_per_dv_analysed', 'rev_per_dv_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    
    return ranked_output.head(output_limit)

def cr_model(df_merge, df_us_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2,metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4):
    
    # Extract subset of dataset
    if region_baseline == 'US All':
        data_analysed = df_merge[df_merge['region'] == region_analysed]
        data_baseline = df_merge.copy()
    else:
        data_analysed = df_merge[df_merge['region'] == region_analysed]
        data_baseline = df_merge[df_merge['region'] == region_baseline]
    
    # Group and aggregate data for data_analysed
    if feature == 'category':
        grouped_data_analysed = data_analysed.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
        'order_percent_of_total':'sum'
        }).reset_index()
        
        grouped_data_analysed = grouped_data_analysed.rename(columns={'order_percent_of_total': 'order_percent'})
    else:
        grouped_data_analysed = data_analysed.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
        'order_percent_of_category':'sum'
        }).reset_index()
        
        grouped_data_analysed = grouped_data_analysed.rename(columns={'order_percent_of_category': 'order_percent'})

    grouped_data_analysed = grouped_data_analysed.rename(columns={'market_sku': 'sku_count'})
    grouped_data_analysed['category'] = grouped_data_analysed['cate-feature'].str.split(': ').str[0]
    grouped_data_analysed[feature] = grouped_data_analysed['cate-feature'].str.split(': ').str[1]

    # Group and aggregate data for data_baseline
    grouped_data_baseline = data_baseline.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
    }).reset_index()

    grouped_data_baseline = grouped_data_baseline.rename(columns={'market_sku': 'sku_count'})
    grouped_data_baseline['category'] = grouped_data_baseline['cate-feature'].str.split(': ').str[0]
    grouped_data_baseline[feature] = grouped_data_baseline['cate-feature'].str.split(': ').str[1]
    
    # Calculate the metrics & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['CR_analysed'] = grouped_data_analysed['total_order'] / grouped_data_analysed['total_detailview']
        grouped_data_baseline['CR_baseline'] = grouped_data_baseline['total_order'] / grouped_data_baseline['total_detailview']

    # Calculate average price & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['average_price_analysed'] = grouped_data_analysed['total_revenue'] / grouped_data_analysed['total_quantity']
        grouped_data_baseline['average_price_baseline'] = grouped_data_baseline['total_revenue'] / grouped_data_baseline['total_quantity']

    # Merge baseline grouped dataset's rev_per_dv column into analysed grouped dataset
    merged_data = pd.merge(grouped_data_analysed, grouped_data_baseline[['cate-feature', 'CR_baseline']],
                           on='cate-feature', how='left')

    # Replace NaN and inf with 0
    merged_data.fillna(0, inplace=True)
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Calculate metric diff
    merged_data['metric_diff_percent'] = np.where(
    merged_data['CR_baseline'] == 0,
    np.nan,  # Replace division by zero with NaN
    (merged_data['CR_analysed'] / merged_data['CR_baseline']) - 1)
    
    # Calculate weighted score
    merged_data['weighted_score'] = merged_data['metric_diff_percent'] * merged_data['order_percent']
    
    # Group & Merge products' US cost
    df_us_cost['cate-feature'] = df_us_cost['category'] + ": " + df_us_cost[feature]
    grouped_us_cost = df_us_cost.groupby('cate-feature').agg({
        'us_total_cost': 'sum'
    }).reset_index()
    
    merged_data = pd.merge(merged_data, grouped_us_cost[['cate-feature', 'us_total_cost']],
                           on='cate-feature', how='left')

    # Filter the output according to the conditions & output limit
    filtered_output = merged_data[
        (merged_data[metric_control_1] >= metric_threshold_1) &
        (merged_data[metric_control_2] >= metric_threshold_2) &
        (merged_data[metric_control_3] >= metric_threshold_3) &
        (merged_data[metric_control_4] >= metric_threshold_4)
    ]
    
    # Rank the output
    if rank == 'Top':
        ranked_output = filtered_output.sort_values(by='weighted_score', ascending= False)
    elif rank == 'Bottom':
        ranked_output = filtered_output.sort_values(by='weighted_score', ascending= True)
    
    # Get the sku info (in case need to add product name or more features in the output)
    sku_info = df_merge[['market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']]
    sku_info = sku_info.drop_duplicates()
    
    # Format the output
    if feature == 'market_sku':
        ranked_output = pd.merge(ranked_output, sku_info[[feature, 'sku_name']], on=feature, how='left')
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'sku_name','sku_count', 'category', 'total_revenue', 'total_quantity','average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'CR_analysed', 'CR_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    elif feature == 'market_spu':
        ranked_output = pd.merge(ranked_output, sku_info[[feature, 'spu_name']], on=feature, how='left')
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'spu_name','sku_count', 'category', 'total_revenue', 'total_quantity', 'average_price_analysed','total_order', 'total_detailview', 'us_total_cost',
                        'CR_analysed', 'CR_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    elif feature == 'category':
        ranked_output = ranked_output.loc[:, ~ranked_output.columns.duplicated(keep='first')]
        ranked_output = ranked_output.reindex(columns=['cate-feature', 'category', 'sku_count', 'total_revenue', 'total_quantity', 'average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'CR_analysed', 'CR_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    else:
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'sku_count', 'category', 'total_revenue', 'total_quantity','average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'CR_analysed', 'CR_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    
    return ranked_output.head(output_limit)

def rev_per_dv_anova(df_merge, output_rev_per_dv, region_analysed, region_baseline, feature):
    
    # Extract subset of dataset
    if region_baseline == 'US All':
        data_analysed = df_merge[df_merge['region'] == region_analysed]
        data_baseline = df_merge.copy()
    else:
        data_analysed = df_merge[df_merge['region'] == region_analysed]
        data_baseline = df_merge[df_merge['region'] == region_baseline]
    
    # See ANOVA for the feature_list get from the comparison model
    feature_list = output_rev_per_dv['cate-feature'].unique()
    
    # Create a dictionary to store the ANOVA results
    # Now conduct Welch's ANOVA to know: for each feature in output list, does their daily metric are significantly different between regions? 
    p_value_each = {}
    
    # Group and aggregate data for data_analysed
    grouped_data_analysed = data_analysed.groupby(['cate-feature','order_date']).agg({
        'total_revenue': 'sum',
        'total_detailview': 'sum'
    }).reset_index()
    
    # Group and aggregate data for data_baseline
    grouped_data_baseline = data_baseline.groupby(['cate-feature','order_date']).agg({
        'total_revenue': 'sum',
        'total_detailview': 'sum'
    }).reset_index()

    # Calculate the metrics & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['rev_per_dv_analysed'] = grouped_data_analysed['total_revenue'] / grouped_data_analysed['total_detailview']
        grouped_data_baseline['rev_per_dv_baseline'] = grouped_data_baseline['total_revenue'] / grouped_data_baseline['total_detailview']
        
    # Concatenate the datasets vertically
    merged_data = pd.concat([grouped_data_analysed, grouped_data_baseline], keys=['region_analysed', 'region_baseline'])
    merged_data['value'] = np.where(
        ~merged_data['rev_per_dv_analysed'].isna(),
        merged_data['rev_per_dv_analysed'],
        np.where(
            ~merged_data['rev_per_dv_baseline'].isna(),
            merged_data['rev_per_dv_baseline'],
            0))
    
    # Add a column to indicate the source dataset
    merged_data['group'] = merged_data.index.get_level_values(0)
    
    # Reset the index for a cleaner output
    merged_data.reset_index(drop=True, inplace=True)
    
    # Replace infinite value with 0
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Initialize an empty list to store dictionaries for each feature
    anova_each_list = []

    # Conduct Welch's ANOVA for each in the list
    for feature_value in feature_list:
        # Filter 'df_merge' to include only rows where 'feature' matches the current value
        subset_data = merged_data[merged_data['cate-feature'] == feature_value]

        # Perform Welch's ANOVA
        result = pg.welch_anova(data=subset_data,
                                dv='value',
                                between='group')
        
        # Store the feature value and p-value in a dictionary
        anova_each_dict = {'cate-feature': feature_value, 'p_value': result['p-unc'].values[0]}
        # Append the dictionary to the list
        anova_each_list.append(anova_each_dict)

    # Convert the list of dictionaries to a DataFrame
    anova_each = pd.DataFrame(anova_each_list)
    
    # Replace Nan by 'Sample size too small to conduct ANOVA'
    anova_each['p_value'].fillna('Sample size too small to conduct ANOVA', inplace=True)
        
    return anova_each

def cr_anova(df_merge, output_cr, region_analysed, region_baseline, feature):
    
    # Extract subset of dataset
    if region_baseline == 'US All':
        data_analysed = df_merge[df_merge['region'] == region_analysed]
        data_baseline = df_merge.copy()
    else:
        data_analysed = df_merge[df_merge['region'] == region_analysed]
        data_baseline = df_merge[df_merge['region'] == region_baseline]
    
    # See ANOVA for the feature_list get from the comparison model
    feature_list = output_cr['cate-feature'].unique()
    
    # Create a dictionary to store the ANOVA results
    # Now conduct Welch's ANOVA to know: for each feature in output list, does their daily metric are significantly different between regions? 
    p_value_each = {}
    
    # Group and aggregate data for data_analysed
    grouped_data_analysed = data_analysed.groupby(['cate-feature','order_date']).agg({
        'total_order': 'sum',
        'total_detailview': 'sum'
    }).reset_index()
    
    # Group and aggregate data for data_baseline
    grouped_data_baseline = data_baseline.groupby(['cate-feature','order_date']).agg({
        'total_order': 'sum',
        'total_detailview': 'sum'
    }).reset_index()

    # Calculate the metrics & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['CR_analysed'] = grouped_data_analysed['total_order'] / grouped_data_analysed['total_detailview']
        grouped_data_baseline['CR_baseline'] = grouped_data_baseline['total_order'] / grouped_data_baseline['total_detailview']
        
    # Concatenate the datasets vertically
    merged_data = pd.concat([grouped_data_analysed, grouped_data_baseline], keys=['region_analysed', 'region_baseline'])
    merged_data['value'] = np.where(
        ~merged_data['CR_analysed'].isna(),
        merged_data['CR_analysed'],
        np.where(
            ~merged_data['CR_baseline'].isna(),
            merged_data['CR_baseline'],
            0))
    
    # Add a column to indicate the source dataset
    merged_data['group'] = merged_data.index.get_level_values(0)
    
    # Reset the index for a cleaner output
    merged_data.reset_index(drop=True, inplace=True)
    
    # Replace infinite value with 0
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Initialize an empty list to store dictionaries for each feature
    anova_each_list = []

    # Conduct Welch's ANOVA for each in the list
    for feature_value in feature_list:
        # Filter 'df_merge' to include only rows where 'feature' matches the current value
        subset_data = merged_data[merged_data['cate-feature'] == feature_value]

        # Perform Welch's ANOVA
        result = pg.welch_anova(data=subset_data,
                                dv='value',
                                between='group')
        
        # Store the feature value and p-value in a dictionary
        anova_each_dict = {'cate-feature': feature_value, 'p_value': result['p-unc'].values[0]}
        # Append the dictionary to the list
        anova_each_list.append(anova_each_dict)

    # Convert the list of dictionaries to a DataFrame
    anova_each = pd.DataFrame(anova_each_list)
    
    # Replace Nan by 'Sample size too small to conduct ANOVA'
    anova_each['p_value'].fillna('Sample size too small to conduct ANOVA', inplace=True)
        
    return anova_each

def rev_per_dv_model_dma(df_us_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2,metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4,
                            baseline_level, df_merge=None, df_merge_analysed=None, df_merge_baseline=None):
    
    if baseline_level == 'DMA Level':
        # Extract subset of dataset
        data_analysed = df_merge[df_merge['dma'] == region_analysed]
        data_baseline = df_merge[df_merge['dma'] == region_baseline]
    elif baseline_level == 'Country & Regional Level':
        if region_baseline in ['US West', 'US East', 'US Southeast', 'US Northwest']:
            data_analysed = df_merge_analysed[df_merge_analysed['dma'] == region_analysed]
            data_baseline = df_merge_baseline[df_merge_baseline['region'] == region_baseline]
        elif region_baseline == 'US All':
            data_analysed = df_merge_analysed[df_merge_analysed['dma'] == region_analysed]
            data_baseline = df_merge_baseline.copy()
    
    # Group and aggregate data for data_analysed
    if feature == 'category':
        grouped_data_analysed = data_analysed.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
        'order_percent_of_total':'sum'
        }).reset_index()
        
        grouped_data_analysed = grouped_data_analysed.rename(columns={'order_percent_of_total': 'order_percent'})
    else:
        grouped_data_analysed = data_analysed.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
        'order_percent_of_category':'sum'
        }).reset_index()
        
        grouped_data_analysed = grouped_data_analysed.rename(columns={'order_percent_of_category': 'order_percent'})

    grouped_data_analysed = grouped_data_analysed.rename(columns={'market_sku': 'sku_count'})
    grouped_data_analysed['category'] = grouped_data_analysed['cate-feature'].str.split(': ').str[0]
    grouped_data_analysed[feature] = grouped_data_analysed['cate-feature'].str.split(': ').str[1]

    # Group and aggregate data for data_baseline
    grouped_data_baseline = data_baseline.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
    }).reset_index()

    grouped_data_baseline = grouped_data_baseline.rename(columns={'market_sku': 'sku_count'})
    grouped_data_baseline['category'] = grouped_data_baseline['cate-feature'].str.split(': ').str[0]
    grouped_data_baseline[feature] = grouped_data_baseline['cate-feature'].str.split(': ').str[1]
    
    # Calculate the metrics & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['rev_per_dv_analysed'] = grouped_data_analysed['total_revenue'] / grouped_data_analysed['total_detailview']
        grouped_data_baseline['rev_per_dv_baseline'] = grouped_data_baseline['total_revenue'] / grouped_data_baseline['total_detailview']

    # Calculate average price & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['average_price_analysed'] = grouped_data_analysed['total_revenue'] / grouped_data_analysed['total_quantity']
        grouped_data_baseline['average_price_baseline'] = grouped_data_baseline['total_revenue'] / grouped_data_baseline['total_quantity']

    # Merge baseline grouped dataset's rev_per_dv column into analysed grouped dataset
    merged_data = pd.merge(grouped_data_analysed, grouped_data_baseline[['cate-feature', 'rev_per_dv_baseline']],
                           on='cate-feature', how='left')

    # Replace NaN and inf with 0
    merged_data.fillna(0, inplace=True)
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Calculate metric diff
    merged_data['metric_diff_percent'] = np.where(
    merged_data['rev_per_dv_baseline'] == 0,
    np.nan,  # Replace division by zero with NaN
    (merged_data['rev_per_dv_analysed'] / merged_data['rev_per_dv_baseline']) - 1)
    
    # Calculate weighted score
    merged_data['weighted_score'] = merged_data['metric_diff_percent'] * merged_data['order_percent']
    
    # Group & Merge products' US cost
    df_us_cost['cate-feature'] = df_us_cost['category'] + ": " + df_us_cost[feature]
    grouped_us_cost = df_us_cost.groupby('cate-feature').agg({
        'us_total_cost': 'sum'
    }).reset_index()
    
    merged_data = pd.merge(merged_data, grouped_us_cost[['cate-feature','us_total_cost']],
                           on='cate-feature', how='left')

    # Filter the output according to the conditions & output limit
    filtered_output = merged_data[
        (merged_data[metric_control_1] >= metric_threshold_1) &
        (merged_data[metric_control_2] >= metric_threshold_2) &
        (merged_data[metric_control_3] >= metric_threshold_3) &
        (merged_data[metric_control_4] >= metric_threshold_4)
    ]
    
    # Rank the output
    if rank == 'Top':
        ranked_output = filtered_output.sort_values(by='weighted_score', ascending= False)
    elif rank == 'Bottom':
        ranked_output = filtered_output.sort_values(by='weighted_score', ascending= True)

    # Get the sku info (in case need to add product name or more features in the output)
    if baseline_level == 'DMA Level':
        sku_info = df_merge[['market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']]
        sku_info = sku_info.drop_duplicates()
    elif baseline_level == 'Country & Regional Level':
        sku_info = pd.concat([df_merge_analysed[['market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']], 
                                     df_merge_baseline[['market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']]])
        sku_info = sku_info.drop_duplicates()
    
    # Format the output
    if feature == 'market_sku':
        ranked_output = pd.merge(ranked_output, sku_info[[feature, 'sku_name']], on=feature, how='left')
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'sku_name','sku_count', 'category', 'total_revenue', 'total_quantity', 'average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'rev_per_dv_analysed', 'rev_per_dv_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    elif feature == 'market_spu':
        ranked_output = pd.merge(ranked_output, sku_info[[feature, 'spu_name']], on=feature, how='left')
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'spu_name', 'sku_count', 'category', 'total_revenue', 'total_quantity', 'average_price_analysed','total_order', 'total_detailview', 'us_total_cost',
                        'rev_per_dv_analysed', 'rev_per_dv_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    elif feature == 'category':
        ranked_output = ranked_output.loc[:, ~ranked_output.columns.duplicated(keep='first')]
        ranked_output = ranked_output.reindex(columns=['cate-feature', 'category', 'sku_count', 'total_revenue', 'total_quantity', 'average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'rev_per_dv_analysed', 'rev_per_dv_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    else:
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'sku_count', 'category',  'total_revenue', 'total_quantity', 'average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'rev_per_dv_analysed', 'rev_per_dv_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    
    return ranked_output.head(output_limit)

def rev_per_dv_anova_dma(output_rev_per_dv, region_analysed, region_baseline, feature,baseline_level, df_merge=None, df_merge_analysed=None, df_merge_baseline=None):
    
    if baseline_level == 'DMA Level':
        # Extract subset of dataset
        data_analysed = df_merge[df_merge['dma'] == region_analysed]
        data_baseline = df_merge[df_merge['dma'] == region_baseline]
    elif baseline_level == 'Country & Regional Level':
        if region_baseline in ['US West', 'US East', 'US Southeast', 'US Northwest']:
            data_analysed = df_merge_analysed[df_merge_analysed['dma'] == region_analysed]
            data_baseline = df_merge_baseline[df_merge_baseline['region'] == region_baseline]
        elif region_baseline == 'US All':
            data_analysed = df_merge_analysed[df_merge_analysed['dma'] == region_analysed]
            data_baseline = df_merge_baseline.copy()
    
    # See ANOVA for the feature_list get from the comparison model
    feature_list = output_rev_per_dv['cate-feature'].unique()
    
    # Create a dictionary to store the ANOVA results
    # Now conduct Welch's ANOVA to know: for each feature in output list, does their daily metric are significantly different between regions? 
    p_value_each = {}
    
    # Group and aggregate data for data_analysed
    grouped_data_analysed = data_analysed.groupby(['cate-feature','order_date']).agg({
        'total_revenue': 'sum',
        'total_detailview': 'sum'
    }).reset_index()
    
    # Group and aggregate data for data_baseline
    grouped_data_baseline = data_baseline.groupby(['cate-feature','order_date']).agg({
        'total_revenue': 'sum',
        'total_detailview': 'sum'
    }).reset_index()

    # Calculate the metrics & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['rev_per_dv_analysed'] = grouped_data_analysed['total_revenue'] / grouped_data_analysed['total_detailview']
        grouped_data_baseline['rev_per_dv_baseline'] = grouped_data_baseline['total_revenue'] / grouped_data_baseline['total_detailview']
        
    # Concatenate the datasets vertically
    merged_data = pd.concat([grouped_data_analysed, grouped_data_baseline], keys=['region_analysed', 'region_baseline'])
    merged_data['value'] = np.where(
        ~merged_data['rev_per_dv_analysed'].isna(),
        merged_data['rev_per_dv_analysed'],
        np.where(
            ~merged_data['rev_per_dv_baseline'].isna(),
            merged_data['rev_per_dv_baseline'],
            0))
    
    # Add a column to indicate the source dataset
    merged_data['group'] = merged_data.index.get_level_values(0)
    
    # Reset the index for a cleaner output
    merged_data.reset_index(drop=True, inplace=True)
    
    # Replace infinite value with 0
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Initialize an empty list to store dictionaries for each feature
    anova_each_list = []

    # Conduct Welch's ANOVA for each in the list
    for feature_value in feature_list:
        # Filter 'df_merge' to include only rows where 'feature' matches the current value
        subset_data = merged_data[merged_data['cate-feature'] == feature_value]

        # Perform Welch's ANOVA
        result = pg.welch_anova(data=subset_data,
                                dv='value',
                                between='group')
        
        # Store the feature value and p-value in a dictionary
        anova_each_dict = {'cate-feature': feature_value, 'p_value': result['p-unc'].values[0]}
        # Append the dictionary to the list
        anova_each_list.append(anova_each_dict)

    # Convert the list of dictionaries to a DataFrame
    anova_each = pd.DataFrame(anova_each_list)
    
    # Replace Nan by 'Sample size too small to conduct ANOVA'
    anova_each['p_value'].fillna('Sample size too small to conduct ANOVA', inplace=True)
        
    return anova_each

def cr_model_dma(df_us_cost, region_analysed, region_baseline, feature, rank, output_limit,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2,metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4,
                            baseline_level, df_merge=None, df_merge_analysed=None, df_merge_baseline=None):
    if baseline_level == 'DMA Level':
        # Extract subset of dataset
        data_analysed = df_merge[df_merge['dma'] == region_analysed]
        data_baseline = df_merge[df_merge['dma'] == region_baseline]
    elif baseline_level == 'Country & Regional Level':
        if region_baseline in ['US West', 'US East', 'US Southeast', 'US Northwest']:
            data_analysed = df_merge_analysed[df_merge_analysed['dma'] == region_analysed]
            data_baseline = df_merge_baseline[df_merge_baseline['region'] == region_baseline]
        elif region_baseline == 'US All':
            data_analysed = df_merge_analysed[df_merge_analysed['dma'] == region_analysed]
            data_baseline = df_merge_baseline.copy()
    
    # Group and aggregate data for data_analysed
    if feature == 'category':
        grouped_data_analysed = data_analysed.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
        'order_percent_of_total':'sum'
        }).reset_index()
        
        grouped_data_analysed = grouped_data_analysed.rename(columns={'order_percent_of_total': 'order_percent'})
    else:
        grouped_data_analysed = data_analysed.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
        'order_percent_of_category':'sum'
        }).reset_index()
        
        grouped_data_analysed = grouped_data_analysed.rename(columns={'order_percent_of_category': 'order_percent'})

    grouped_data_analysed = grouped_data_analysed.rename(columns={'market_sku': 'sku_count'})
    grouped_data_analysed['category'] = grouped_data_analysed['cate-feature'].str.split(': ').str[0]
    grouped_data_analysed[feature] = grouped_data_analysed['cate-feature'].str.split(': ').str[1]

    # Group and aggregate data for data_baseline
    grouped_data_baseline = data_baseline.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
    }).reset_index()

    grouped_data_baseline = grouped_data_baseline.rename(columns={'market_sku': 'sku_count'})
    grouped_data_baseline['category'] = grouped_data_baseline['cate-feature'].str.split(': ').str[0]
    grouped_data_baseline[feature] = grouped_data_baseline['cate-feature'].str.split(': ').str[1]
    
    # Calculate the metrics & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['CR_analysed'] = grouped_data_analysed['total_order'] / grouped_data_analysed['total_detailview']
        grouped_data_baseline['CR_baseline'] = grouped_data_baseline['total_order'] / grouped_data_baseline['total_detailview']

    # Calculate average price & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['average_price_analysed'] = grouped_data_analysed['total_revenue'] / grouped_data_analysed['total_quantity']
        grouped_data_baseline['average_price_baseline'] = grouped_data_baseline['total_revenue'] / grouped_data_baseline['total_quantity']

    # Merge baseline grouped dataset's rev_per_dv column into analysed grouped dataset
    merged_data = pd.merge(grouped_data_analysed, grouped_data_baseline[['cate-feature', 'CR_baseline']],
                           on='cate-feature', how='left')

    # Replace NaN and inf with 0
    merged_data.fillna(0, inplace=True)
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Calculate metric diff
    merged_data['metric_diff_percent'] = np.where(
    merged_data['CR_baseline'] == 0,
    np.nan,  # Replace division by zero with NaN
    (merged_data['CR_analysed'] / merged_data['CR_baseline']) - 1)
    
    # Calculate weighted score
    merged_data['weighted_score'] = merged_data['metric_diff_percent'] * merged_data['order_percent']
    
    # Group & Merge products' US cost
    df_us_cost['cate-feature'] = df_us_cost['category'] + ": " + df_us_cost[feature]
    grouped_us_cost = df_us_cost.groupby('cate-feature').agg({
        'us_total_cost': 'sum'
    }).reset_index()
    
    merged_data = pd.merge(merged_data, grouped_us_cost[['cate-feature', 'us_total_cost']],
                           on='cate-feature', how='left')

    # Filter the output according to the conditions & output limit
    filtered_output = merged_data[
        (merged_data[metric_control_1] >= metric_threshold_1) &
        (merged_data[metric_control_2] >= metric_threshold_2) &
        (merged_data[metric_control_3] >= metric_threshold_3) &
        (merged_data[metric_control_4] >= metric_threshold_4)
    ]
    
    # Rank the output
    if rank == 'Top':
        ranked_output = filtered_output.sort_values(by='weighted_score', ascending= False)
    elif rank == 'Bottom':
        ranked_output = filtered_output.sort_values(by='weighted_score', ascending= True)

    # Get the sku info (in case need to add product name or more features in the output)
    if baseline_level == 'DMA Level':
        sku_info = df_merge[['market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']]
        sku_info = sku_info.drop_duplicates()
    elif baseline_level == 'Country & Regional Level':
        sku_info = pd.concat([df_merge_analysed[['market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']], 
                                     df_merge_baseline[['market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']]])
        sku_info = sku_info.drop_duplicates()
    
    # Format the output
    if feature == 'market_sku':
        ranked_output = pd.merge(ranked_output, sku_info[[feature, 'sku_name']], on=feature, how='left')
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'sku_name','sku_count', 'category', 'total_revenue', 'total_quantity','average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'CR_analysed', 'CR_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    elif feature == 'market_spu':
        ranked_output = pd.merge(ranked_output, sku_info[[feature, 'spu_name']], on=feature, how='left')
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'spu_name','sku_count', 'category', 'total_revenue', 'total_quantity', 'average_price_analysed','total_order', 'total_detailview', 'us_total_cost',
                        'CR_analysed', 'CR_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    elif feature == 'category':
        ranked_output = ranked_output.loc[:, ~ranked_output.columns.duplicated(keep='first')]
        ranked_output = ranked_output.reindex(columns=['cate-feature', 'category', 'sku_count', 'total_revenue', 'total_quantity', 'average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'CR_analysed', 'CR_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    else:
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'sku_count', 'category', 'total_revenue', 'total_quantity','average_price_analysed', 'total_order', 'total_detailview', 'us_total_cost',
                        'CR_analysed', 'CR_baseline', 'metric_diff_percent', 'order_percent', 'weighted_score'])
    
    return ranked_output.head(output_limit)

def cr_anova_dma(output_cr, region_analysed, region_baseline, feature, baseline_level, df_merge=None, df_merge_analysed=None, df_merge_baseline=None):
    
    if baseline_level == 'DMA Level':
        # Extract subset of dataset
        data_analysed = df_merge[df_merge['dma'] == region_analysed]
        data_baseline = df_merge[df_merge['dma'] == region_baseline]
    elif baseline_level == 'Country & Regional Level':
        if region_baseline in ['US West', 'US East', 'US Southeast', 'US Northwest']:
            data_analysed = df_merge_analysed[df_merge_analysed['dma'] == region_analysed]
            data_baseline = df_merge_baseline[df_merge_baseline['region'] == region_baseline]
        elif region_baseline == 'US All':
            data_analysed = df_merge_analysed[df_merge_analysed['dma'] == region_analysed]
            data_baseline = df_merge_baseline.copy()
    
    # See ANOVA for the feature_list get from the comparison model
    feature_list = output_cr['cate-feature'].unique()
    
    # Create a dictionary to store the ANOVA results
    # Now conduct Welch's ANOVA to know: for each feature in output list, does their daily metric are significantly different between regions? 
    p_value_each = {}
    
    # Group and aggregate data for data_analysed
    grouped_data_analysed = data_analysed.groupby(['cate-feature','order_date']).agg({
        'total_order': 'sum',
        'total_detailview': 'sum'
    }).reset_index()
    
    # Group and aggregate data for data_baseline
    grouped_data_baseline = data_baseline.groupby(['cate-feature','order_date']).agg({
        'total_order': 'sum',
        'total_detailview': 'sum'
    }).reset_index()

    # Calculate the metrics & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['CR_analysed'] = grouped_data_analysed['total_order'] / grouped_data_analysed['total_detailview']
        grouped_data_baseline['CR_baseline'] = grouped_data_baseline['total_order'] / grouped_data_baseline['total_detailview']
        
    # Concatenate the datasets vertically
    merged_data = pd.concat([grouped_data_analysed, grouped_data_baseline], keys=['region_analysed', 'region_baseline'])
    merged_data['value'] = np.where(
        ~merged_data['CR_analysed'].isna(),
        merged_data['CR_analysed'],
        np.where(
            ~merged_data['CR_baseline'].isna(),
            merged_data['CR_baseline'],
            0))
    
    # Add a column to indicate the source dataset
    merged_data['group'] = merged_data.index.get_level_values(0)
    
    # Reset the index for a cleaner output
    merged_data.reset_index(drop=True, inplace=True)
    
    # Replace infinite value with 0
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Initialize an empty list to store dictionaries for each feature
    anova_each_list = []

    # Conduct Welch's ANOVA for each in the list
    for feature_value in feature_list:
        # Filter 'df_merge' to include only rows where 'feature' matches the current value
        subset_data = merged_data[merged_data['cate-feature'] == feature_value]

        # Perform Welch's ANOVA
        result = pg.welch_anova(data=subset_data,
                                dv='value',
                                between='group')
        
        # Store the feature value and p-value in a dictionary
        anova_each_dict = {'cate-feature': feature_value, 'p_value': result['p-unc'].values[0]}
        # Append the dictionary to the list
        anova_each_list.append(anova_each_dict)

    # Convert the list of dictionaries to a DataFrame
    anova_each = pd.DataFrame(anova_each_list)
    
    # Replace Nan by 'Sample size too small to conduct ANOVA'
    anova_each['p_value'].fillna('Sample size too small to conduct ANOVA', inplace=True)
        
    return anova_each
