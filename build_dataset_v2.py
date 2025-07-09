import pandas as pd
import numpy as np

np.random.seed(42)

# -----------------------------
# PARAMETERS
# -----------------------------
num_customers = 30000
num_stores = 150
weeks = [1, 2, 3, 4]
campaigns = ['A', 'B', 'C']
locations = ['Urban', 'Suburban', 'Rural']
sizes = ['Small', 'Medium', 'Large']
gifts = ['Tote Bag', 'Socks', 'Keychain', 'Stickers']

# -----------------------------
# CAMPAIGN-SPECIFIC CONFIGURATION
# -----------------------------
campaign_config = {
    'A': {
        'purchase_prob': 0.7,
        'collection_prob': 0.3,
        'segment_probs': [0.4, 0.6],
        'avg_unit_price': 80
    },
    'B': {
        'purchase_prob': 0.5,
        'collection_prob': 0.14,
        'segment_probs': [0.25, 0.75],
        'avg_unit_price': 60
    },
    'C': {
        'purchase_prob': 0.3,
        'collection_prob': 0.07,
        'segment_probs': [0.15, 0.85],
        'avg_unit_price': 35
    }
}

# -----------------------------
# CREATE STORES
# -----------------------------
store_ids = [f"S{str(i).zfill(3)}" for i in range(1, num_stores + 1)]
store_df = pd.DataFrame({
    'store_id': store_ids,
    'store_location': np.random.choice(locations, num_stores, p=[0.4, 0.4, 0.2]),
    'store_size_category': np.random.choice(sizes, num_stores, p=[0.3, 0.5, 0.2]),
    'avg_revenue_last_3_months': np.random.normal(120000, 25000, num_stores).round(0),
    'monthly_active_customers': np.random.randint(500, 3000, num_stores),
    'campaign_group': np.random.choice(campaigns, size=num_stores, p=[0.33, 0.34, 0.33])
})

# -----------------------------
# ASSIGN CUSTOMERS TO STORES
# -----------------------------
customer_ids = np.arange(1, num_customers + 1)
customer_store_map = pd.DataFrame({
    'customer_id': customer_ids,
    'store_id': np.random.choice(store_df['store_id'], num_customers)
})
customer_store_map = customer_store_map.merge(store_df, on='store_id', how='left')

# -----------------------------
# GENERATE WEEKLY TRANSACTIONAL DATA
# -----------------------------
all_data = []
for week in weeks:
    temp = customer_store_map.copy()
    temp['week'] = week

    # Assign purchase_flag based on campaign
    temp['purchase_prob'] = temp['campaign_group'].apply(lambda x: campaign_config[x]['purchase_prob'])
    temp['purchase_flag'] = np.random.binomial(1, temp['purchase_prob'])
    temp = temp[temp['purchase_flag'] == 1].copy()

    # Visit & product orders
    temp['visit_count'] = np.random.randint(1, 3, size=len(temp))
    temp['products_ord'] = np.random.randint(1, 6, size=len(temp))

    # Assign collection rate per campaign
    temp['collection_prob'] = temp['campaign_group'].apply(lambda x: campaign_config[x]['collection_prob'])
    temp['new_collection_items'] = np.random.binomial(temp['products_ord'], temp['collection_prob'])
    temp['other_products_ord'] = temp['products_ord'] - temp['new_collection_items']

    # Apply campaign logic
    temp['discount'] = 0.0
    temp['gift_received'] = np.nan
    temp['used_loyalty_or_voucher'] = False

    for campaign in campaigns:
        idx = temp['campaign_group'] == campaign
        eligible = idx & (temp['new_collection_items'] > 0)

        if campaign == 'A':
            temp.loc[eligible, 'discount'] = 0.05
        elif campaign == 'B':
            temp.loc[eligible, 'gift_received'] = np.random.choice(gifts, size=eligible.sum())
        elif campaign == 'C':
            idx_c = eligible & (np.random.rand(len(temp)) < 0.5)
            temp.loc[idx_c, 'used_loyalty_or_voucher'] = True
            temp.loc[idx_c, 'discount'] = 0.10

    # Revenue Calculation with variable unit price
    temp['avg_unit_price'] = temp['campaign_group'].apply(lambda x: campaign_config[x]['avg_unit_price'])
    temp['unit_price'] = np.random.normal(temp['avg_unit_price'], 20).clip(10, 120)
    temp['revenue_before_discount'] = temp['products_ord'] * temp['unit_price']
    temp['revenue'] = (temp['revenue_before_discount'] * (1 - temp['discount'])).round(2)

    # Demographics for purchasers
    temp['age'] = np.random.normal(32, 15, len(temp)).clip(18, 65).round()
    temp['gender'] = np.random.choice(['Male', 'Female'], len(temp))

    # Assign customer segment based on campaign
    def assign_segment(row):
        if row['week'] == 1:
            p_new, p_returning = campaign_config[row['campaign_group']]['segment_probs']
            return np.random.choice(['New', 'Returning'], p=[p_new, p_returning])
        else:
            return 'Returning'

    temp['customer_segment'] = temp.apply(assign_segment, axis=1)

    # Promo used
    temp['promo_used'] = np.where(temp['discount'] > 0, 'Yes',
                                  np.where(temp['gift_received'].notna(), 'Yes', 'No'))

    # Collection flag
    temp['used_new_collection'] = temp['new_collection_items'] > 0

    all_data.append(temp)

# -----------------------------
# COMBINE AND FINALIZE
# -----------------------------
final_df = pd.concat(all_data, ignore_index=True)

final_df = final_df[[ 
    'customer_id', 'week', 'store_id', 'campaign_group', 'store_location', 'store_size_category',
    'avg_revenue_last_3_months', 'monthly_active_customers', 'customer_segment', 'age', 'gender',
    'visit_count', 'products_ord', 'new_collection_items', 'other_products_ord',
    'discount', 'gift_received', 'used_loyalty_or_voucher', 'promo_used',
    'revenue_before_discount', 'revenue', 'used_new_collection'
]]

# -----------------------------
# EXPORT TO CSV
# -----------------------------
final_df.to_csv("ab_marketing_campaign_updated.csv", index=False)
print("✅ Dataset saved as ab_marketing_campaign_updated.csv")
