import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import binom_test


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

    
def _get_dataframes(data):
    acct = data['acct']
    cons = data['cons']
    inflows = data['inflows']
    outflows = data['outflows']
    outflows['amount'] = -outflows['amount']
    total = pd.concat([inflows, outflows])
    return acct, cons, inflows, outflows, total


def get_categorical_features(data):
    acct, cons, inflows, outflows, total = _get_dataframes(data)
    
    by_category = total[['prism_consumer_id', 'category_description', 'month', 'amount']].groupby(['prism_consumer_id', 'category_description', 'month']).sum()
    by_category = by_category.reset_index()

    consumer_category_months = _get_consumer_category_months(data)
    by_category = by_category.merge(consumer_category_months, on=['prism_consumer_id', 'category_description', 'month'], how='right')
    by_category = by_category.fillna(0)
    by_category['diffs'] = by_category.groupby(['prism_consumer_id', 'category_description'])['amount'].transform(lambda x: x.diff()) 
    metrics = by_category.drop(columns='month').groupby(['prism_consumer_id', 'category_description']).agg(['mean', 'std'])

    acct_on_cons = acct[['prism_consumer_id', 'balance']].groupby('prism_consumer_id').sum()

    pivot_df = metrics.pivot_table(index='prism_consumer_id', columns='category_description')
    pivot_df.columns = [f'{col[2]}_{col[0]}_{col[1]}' for col in pivot_df.columns]
    pivot_df = pivot_df.fillna(0)
    category_cols = list(pivot_df.columns)
    pivot_df = pivot_df.merge(cons, on='prism_consumer_id', how='left')
    pivot_df = pivot_df.merge(acct_on_cons, on='prism_consumer_id', how='left')
    pivot_df

    return pivot_df


def _get_consumer_category_months(data):
    """
    Puts all permutations of consumers per category per month into a new DataFrame
    so that it can later be joined with.

    :param data: a dictionary containing relevant DataFrames
    :type data: dict[str, DataFrame]
    """
    acct, cons, inflows, outflows, total = _get_dataframes(data)
    
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
        columns = ['prism_consumer_id', 'category_description', 'month']
    )
    return consumer_category_months


def get_income_features(data):
    acct, cons, inflows, outflows, total = _get_dataframes(data)
    merged_df = pd.merge(cons, inflows, on='prism_consumer_id', how='left')
    result_df = merged_df[merged_df['posted_date'] < merged_df['evaluation_date']]
    df = result_df.drop(
        columns=['evaluation_date', 'APPROVED', 'prism_account_id']
    )
    df['is_income'] = df['category_description'].apply(
        lambda cat: 1 if 'INCOME' in cat else 0
    )
    
    grouped_data = inflows.groupby(
        ['prism_consumer_id', 'prism_account_id', 'memo_clean']
    )
    reg_pays = _detect_similar_date_distances(grouped_data)

    for income in reg_pays:
        df.loc[(df['prism_consumer_id'] == income[0]) & (
            df['memo_clean'] == income[2]), df.columns[-1]] = 1
    
    feature_df = pd.DataFrame(df['prism_consumer_id'].unique(),
                    columns=['prism_consumer_id'])
    feature_df['avg_inc_perQ'] = feature_df['prism_consumer_id'].apply(
        lambda x: _feature_gen(x, df)[0])
    feature_df['avg_inc_pct_change_perQ'] = feature_df['prism_consumer_id'].apply(
        lambda x: _feature_gen(x, df)[1])

    feature_df['avg_inc_perQ'].fillna(0, inplace=True)
    feature_df['avg_inc_pct_change_perQ'].fillna(0, inplace=True)

    return feature_df


def _detect_similar_date_distances(grouped_data):
    def calculate_time_diff(dates):
        dates = dates.sort_values()
        diff = dates - dates.shift()
        diff = diff.dropna()
        diff = diff[diff != pd.Timedelta(0)]
        return diff

    def is_similar_distances(time_diffs, period=30):
        c_gt_low = 0
        c_lt_high = 0

        n = len(time_diffs)
        if n < 3: return False

        for d in time_diffs:
            if d.days > 20:
                c_gt_low += 1
            if d.days < 40:
                c_lt_high += 1
        if n < 10:
            p_value_gt_low = binom_test(
                c_gt_low, n, 0.5, alternative='greater')
            p_value_lt_high = binom_test(
                c_lt_high, n, 0.5, alternative='greater')
            if max(p_value_gt_low, p_value_lt_high) < 0.3:
                return True
        else:
            z_gt_low = (c_gt_low - n/2)/np.sqrt(n/4)
            z_lt_high = (c_lt_high - n/2)/np.sqrt(n/4)
            p_value_gt_low = 1 - norm.cdf(z_gt_low)
            p_value_lt_high = 1 - norm.cdf(z_lt_high)

            if max(p_value_gt_low, p_value_lt_high) < 0.3:
                return True
        return False

    results = {}
    reg_pays = set()

    for group_name, group_data in grouped_data:
        date_list = group_data['posted_date']
        time_diffs = calculate_time_diff(date_list)
        results[group_name] = is_similar_distances(time_diffs)

    for key, value in results.items():
        if value:
            reg_pays.add(key)

    return reg_pays


def _feature_gen(consumer_id, df):
    data = df[(df['prism_consumer_id'] == consumer_id)
              & (df['is_income'] == 1)]
    data = data.sort_values('posted_date')

    data['posted_date'] = pd.to_datetime(data['posted_date'])

    data['quarter'] = data['posted_date'].dt.to_period('Q')

    quarterly_income = data.groupby(['prism_consumer_id', 'quarter'])[
        'amount'].sum().reset_index()
    quarterly_income['pct_change'] = quarterly_income['amount'].pct_change()

    avg_income = quarterly_income['amount'].mean()
    avg_change = quarterly_income['pct_change'].mean()

    return avg_income, avg_change
