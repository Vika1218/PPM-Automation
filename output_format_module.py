import pandas as pd

def number_format(df,col_name):
    # in format 1,000
    if col_name in ['sku_count','total_quantity','total_order','total_detailview',]:
        return df[col_name].map('{:,.0f}'.format)
    # in format $1,000.00
    elif col_name in ['total_revenue','average_price_analysed','country_total_cost_per_sku']:
        return df[col_name].map('${:,.2f}'.format)
    # in format 1,000.00
    elif col_name in ['rev_per_dv_analysed','rev_per_dv_baseline']:
        return df[col_name].map('{:,.2f}'.format)
    elif col_name in ['p_value']:
        return df[col_name].apply(lambda x: '{:,.2f}'.format(x) if x != 'Sample size too small to conduct ANOVA' else x)
    elif col_name in ['CR_analysed','CR_baseline', 'metric_diff_percent', 'order_percent','weighted_score']:
        return df[col_name].map('{:.2%}'.format)
    else:
        return df[col_name]

def rename_column(df):
    column_mapping = {
        'market_sku': 'Market SKU',
        'sku_name': 'SKU Name',
        'market_spu': 'Market SPU',
        'spu_name': 'SPU Name',
        'category': 'Category',
        'collection': 'Collection',
        'subcategory': 'Subcategory',
        'color_tone': 'Color Tone',
        'material_helper': 'Material Helper',
        'cate-feature': 'Category: Feature',
        'sku_count': 'SKU Count',
        'total_revenue': 'Revenue_Analysed',
        'total_quantity': 'Quantity_Analysed',
        'average_price_analysed': 'Ave. Price_Analysed',
        'total_order': 'Order_Analysed',
        'total_detailview': 'Detailview_Analysed',
        'country_total_cost_per_sku': 'Country Average Cost per SKU',
        'rev_per_dv_analysed': 'Rev/DV_Analysed',
        'rev_per_dv_baseline': 'Rev/DV_Baseline',
        'CR_analysed': 'CR_Analysed',
        'CR_baseline': 'CR_Baseline',
        'metric_diff_percent': '%Metric Difference',
        'order_percent': '%Order',
        'weighted_score': 'Weighted Score',
        'p_value': 'P-Value'
    }
    df.rename(columns=column_mapping, inplace=True)

def output_format(df, feature):
    # Drop useless columns
    if feature in ['market_sku', 'market_spu', 'category', 'subcategory','collection']:
        df = df.drop(columns=['cate-feature'])

    for col in df.columns:
        df[col] = number_format(df, col)
    rename_column(df)
    return df

def output_format_reverse(df, feature1, feature2):
    if feature2 == 'No Selection':
        feature = feature1
    else:
        feature = feature2
    # Drop useless columns
    if feature in ['market_sku', 'market_spu', 'category', 'subcategory','collection']:
        df = df.drop(columns=['cate-feature'])

    for col in df.columns:
        df[col] = number_format(df, col)
    rename_column(df)
    return df