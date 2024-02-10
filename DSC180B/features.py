import pandas as pd
import numpy as np



def get_data():
    acct = pd.read_parquet('data/q2_acctDF_final.pqt')
    cons = pd.read_parquet('data/q2_consDF_final.pqt')
    inflows = pd.read_parquet('data/q2_inflows_final.pqt')
    outflows = pd.concat([
        pd.read_parquet('data/q2_outflows_1sthalf_final.pqt'),
        pd.read_parquet('data/q2_outflows_2ndhalf_final.pqt')
    ])
    return {
        'acct': acct,
        'cons': cons,
        'inflows': inflows,
        'outflows': outflows
    }
    
def get_dataframes(data):
    acct = data['acct']
    cons = data['cons']
    inflows = data['inflows']
    outflows = data['outflows']
    outflows['amount'] = -outflows['amount']
    total = pd.concat([inflows, outflows])
    return acct, cons, inflows, outflows, total

def categorical_features(data):
    acct, cons, inflows, outflows, total = get_dataframes(data)
    
    by_category = total[['prism_consumer_id', 'category_description', 'month', 'amount']].groupby(['prism_consumer_id', 'category_description', 'month']).sum()
    by_category = by_category.reset_index()
    by_category = by_category.merge(consumer_category_months, on=['prism_consumer_id', 'category_description', 'month'], how='right')
    by_category = by_category.fillna(0)
    by_category['diffs'] = by_category.groupby(['prism_consumer_id', 'category_description'])['amount'].transform(lambda x: x.diff()) 
    metrics = by_category.drop(columns='month').groupby(['prism_consumer_id', 'category_description']).agg(['mean', 'std'])
    metrics
    

def get_consumer_category_months(data):
    """
    Puts all permutations of consumers per category per month into a new DataFrame
    so that it can later be joined with.

    :param data: a dictionary containing relevant DataFrames
    :type data: dict[str, DataFrame]
    """
    acct, cons, inflows, outflows, total = get_dataframes(data)
    
    total['datetime'] = pd.to_datetime(total['posted_date'])
    total['month'] = total['datetime'].apply(lambda d: d.strftime('%Y-%m'))
    consumer_intervals = total[['prism_consumer_id', 'month']].groupby('prism_consumer_id').agg(['min', 'max'])
    consumer_intervals.columns = ['min', 'max']
    consumer_intervals = consumer_intervals.to_dict()
    categories = sorted(total['category_description'].unique())
    consumer_category_months = pd.DataFrame(columns = ['prism_consumer_id', 'month', 'category_description'])
    
    consumer_category_months = []
    for consumer in consumer_intervals['min'].keys():
        consumer_min = consumer_intervals['min'][consumer]
        consumer_max = consumer_intervals['max'][consumer]
        month_range = pd.date_range(consumer_min, consumer_max, freq='1M')
        month_range = [d.strftime('%Y-%m') for d in month_range] + [consumer_max]
        for category in categories:
            for month in month_range:
                consumer_category_months.append({
                    "prism_consumer_id": consumer,
                    "month": month,
                    "category_description": category,
                })
    consumer_category_months = pd.DataFrame(
    data=consumer_category_months, 
    columns = ['prism_consumer_id', 'category_description', 'month'])
