import pandas as pd


def generate_user_features(
    orders: pd.DataFrame, clicks: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate user features DataFrame from clicks and orders
    """
    
    orders['discount_percent'] = orders["discount_value"] / orders["initial_product_sum"]
    orders['fee_percent'] = (orders["delivery_fee"] + orders["service_fee"]) / orders["total_value"]

    user_features = pd.DataFrame(columns=['user_id'])
    user_features['user_id'] = list(set(
        orders["customer_id"].to_list() + clicks["user_id"].to_list()
    ))
    user_features = user_features.set_index('user_id')
    
    for delivery_type in orders["delivery_type"].unique():
        user_features[
            f"{str.lower(delivery_type)}_percent"
        ] = orders[orders["delivery_type"] == delivery_type].groupby(
            "customer_id", sort=False
        ).size().div(
            orders.groupby("customer_id", sort=False).size(),
            fill_value=0.
        )
        
    for status_id in orders["status_id"].unique():
        user_features[
            f"{status_id}_percent"
        ] = orders[orders["status_id"] == status_id].groupby(
            "customer_id", sort=False
        ).size().div(
            orders.groupby("customer_id", sort=False).size(),
            fill_value=0.
        )
    
    for col in [
        'expected_delivery_min',
        'products_count',
        'total_value',
        'discount_value',
        'delivery_fee',
        'backend_expected_delivery_time',
        'delivery_time',
        'delivery_distance',
        'star_rating',
        'discount_percent',
        'fee_percent'
    ]:
        user_features[f"{col}_mean"] = orders.groupby(
            "customer_id", sort=False
        )[col].mean() 
    
    user_features["order_cnt"] = orders.groupby(
            "customer_id", sort=False
    ).size()
    
    session_length = clicks.groupby(
        ['user_id', 'session_id'], sort=False, as_index=False
    ).size()

    user_features["session_length_mean"] = (
        session_length
        .groupby('user_id', sort=False)["size"].mean()
    )
    user_features["session_cnt"] = (
        session_length
        .groupby('user_id', sort=False)["session_id"].nunique()
    )    
    
    return user_features
