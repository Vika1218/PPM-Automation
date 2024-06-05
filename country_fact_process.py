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

def process_country_cost(conn, country_cost_query, params):
    # Process US cost data
    df_country_cost = fetch_data_from_database(conn, country_cost_query, params)
    return df_country_cost

def merge_and_fill(df_order, df_dv, df_country_cost, market):
    # Get unique values of 'unique_id'
    id = pd.concat([df_order['unique_id'], df_dv['unique_id']]).unique()
    
    # Create a new DataFrame with unique 'unique_id'
    df_merge = pd.DataFrame({'unique_id': id})
    
    # Merge 'df_merge' with 'df_order','df_dv','df_country_cost'
    df_merge = pd.merge(df_merge, df_order[['unique_id', 'order_date', 'market', 'market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper', 'total_revenue', 'total_quantity', 'total_order']], on='unique_id', how='left')
    df_merge = pd.merge(df_merge, df_dv[['unique_id', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper', 'total_detailview', 'atc']], on='unique_id', how='left')
    df_merge = pd.merge(df_merge, df_country_cost[['unique_id', 'country_total_cost']], on='unique_id', how='left')
    columns_to_fill = ['sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']
    
    # Fill missing values
    for column in columns_to_fill:
        df_merge[column] = df_merge[column+ '_x'].fillna(df_merge[column + '_y'])
    
    # Drop redundant columns '_x' and '_y' after filling missing values
    df_merge.drop(columns=['sku_name_x', 'market_spu_x', 'spu_name_x', 'master_category_x', 'category_x', 'subcategory_x', 'collection_x', 'color_tone_x', 'material_helper_x', 'sku_name_y', 'market_spu_y', 'spu_name_y', 'master_category_y', 'category_y', 'subcategory_y', 'collection_y', 'color_tone_y', 'material_helper_y'], inplace=True)
    # Fill in order date, sku
    df_merge.loc[df_merge['order_date'].isnull(), 'order_date'] = df_merge.loc[df_merge['order_date'].isnull(), 'unique_id'].str.split('_').str[2]
    df_merge.loc[df_merge['market_sku'].isnull(), 'market_sku'] = df_merge.loc[df_merge['market_sku'].isnull(), 'unique_id'].str.split('_').str[0]

    # Fill in missing values for 'market', 'total_detailview', 'total_revenue', 'total_quantity', 'total_order'
    df_merge['market'] = df_merge['market'].fillna(market)
    df_merge['total_detailview'] = df_merge['total_detailview'].fillna(0)
    df_merge['total_revenue'] = df_merge['total_revenue'].fillna(0)
    df_merge['total_quantity'] = df_merge['total_quantity'].fillna(0)
    df_merge['total_order'] = df_merge['total_order'].fillna(0)
    df_merge['atc'] = df_merge['atc'].fillna(0)
    df_merge['country_total_cost'] = df_merge['country_total_cost'].fillna(0)

    return df_merge


def output_model(df_merge, metric, feature1, feature2,
                            metric_control_1, metric_threshold_1, metric_control_2, metric_threshold_2,metric_control_3, metric_threshold_3,
                            metric_control_4, metric_threshold_4):
    
    # Extract subset of dataset
    data_analysed = df_merge.copy()
    
    if feature2 == 'No Selection':
        feature = feature1
    else:
        feature = feature2

    # Group and aggregate data for data_analysed
    if feature == 'category':
        grouped_data_analysed = data_analysed.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
        'atc':'sum',
        'order_percent_of_total':'sum',
        'country_total_cost':'sum'
        }).reset_index()
        
        grouped_data_analysed = grouped_data_analysed.rename(columns={'order_percent_of_total': 'order_percent'})
    else:
        grouped_data_analysed = data_analysed.groupby('cate-feature').agg({
        'market_sku':'nunique',
        'total_revenue': 'sum',
        'total_quantity': 'sum',
        'total_order': 'sum',
        'total_detailview': 'sum',
        'atc':'sum',
        'order_percent_of_category':'sum',
        'country_total_cost':'sum'
        }).reset_index()
        
        grouped_data_analysed = grouped_data_analysed.rename(columns={'order_percent_of_category': 'order_percent'})

    grouped_data_analysed = grouped_data_analysed.rename(columns={'market_sku': 'sku_count'})
    grouped_data_analysed['category'] = grouped_data_analysed['cate-feature'].str.split(': ').str[0]
    grouped_data_analysed[feature] = grouped_data_analysed['cate-feature'].str.split(': ').str[1]
    
    # Calculate the metrics & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['rev_per_dv'] = grouped_data_analysed['total_revenue'] / grouped_data_analysed['total_detailview']
        grouped_data_analysed['CR'] = grouped_data_analysed['total_order'] / grouped_data_analysed['total_detailview']

    # Calculate average price & take care the error case
    with np.errstate(divide='ignore', invalid='ignore'):
        grouped_data_analysed['average_price'] = grouped_data_analysed['total_revenue'] / grouped_data_analysed['total_quantity']

    # Merge baseline grouped dataset's rev_per_dv column into analysed grouped dataset
    merged_data = grouped_data_analysed.copy()

    # Replace NaN and inf with 0
    merged_data.fillna(0, inplace=True)
    merged_data.replace([np.inf, -np.inf], 0, inplace=True)

    # Calculate cost per sku
    merged_data['country_total_cost_per_sku'] = merged_data['country_total_cost']/merged_data['sku_count']
    merged_data.drop(columns=['country_total_cost'], inplace=True)

    # Filter the output according to the conditions & output limit
    filtered_output = merged_data[
        (merged_data[metric_control_1] >= metric_threshold_1) &
        (merged_data[metric_control_2] >= metric_threshold_2) &
        (merged_data[metric_control_3] >= metric_threshold_3) &
        (merged_data[metric_control_4] >= metric_threshold_4)
    ]
    
    if metric == 'Rev per DV':
        ranked_output = filtered_output.sort_values(by='rev_per_dv', ascending= False)
    elif metric == 'CR':
        ranked_output = filtered_output.sort_values(by='CR', ascending= False)

    # Get the sku info (in case need to add product name or more features in the output)
    sku_info = df_merge[['market_sku', 'sku_name', 'market_spu', 'spu_name', 'master_category', 'category', 'subcategory', 'collection', 'color_tone', 'material_helper']]
    if feature == 'market_spu':
        sku_info = sku_info.drop_duplicates(subset=[feature])
    else:
        sku_info = sku_info.drop_duplicates()
    
    # Format the output
    if feature == 'market_sku':
        ranked_output = pd.merge(ranked_output, sku_info[[feature, 'sku_name']], on=feature, how='left')
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'sku_name','sku_count', 'category', 'total_revenue', 'total_quantity', 'average_price', 'total_order', 'total_detailview', 'atc','country_total_cost_per_sku',
                        'rev_per_dv', 'CR','order_percent'])
    elif feature == 'market_spu':
        ranked_output = pd.merge(ranked_output, sku_info[[feature, 'spu_name']], on=feature, how='left')
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'spu_name', 'sku_count', 'category', 'total_revenue', 'total_quantity', 'average_price','total_order', 'total_detailview', 'atc','country_total_cost_per_sku',
                        'rev_per_dv', 'CR', 'order_percent'])
    elif feature == 'category':
        ranked_output = ranked_output.loc[:, ~ranked_output.columns.duplicated(keep='first')]
        ranked_output = ranked_output.reindex(columns=['cate-feature', 'category', 'sku_count', 'total_revenue', 'total_quantity', 'average_price', 'total_order', 'total_detailview', 'atc','country_total_cost_per_sku',
                        'rev_per_dv', 'CR', 'order_percent'])
    else:
        ranked_output = ranked_output.reindex(columns=['cate-feature', feature, 'sku_count', 'category',  'total_revenue', 'total_quantity', 'average_price', 'total_order', 'total_detailview', 'atc', 'country_total_cost_per_sku',
                        'rev_per_dv', 'CR', 'order_percent'])

    return ranked_output