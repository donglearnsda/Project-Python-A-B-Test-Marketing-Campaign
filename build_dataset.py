import pandas as pd
import numpy as np

np.random.seed(42)

# Parameters
num_customers = 30000
num_stores = 150
weeks = [1, 2, 3, 4]
campaigns = ['A', 'B', 'C']
locations = ['Urban', 'Suburban', 'Rural']
sizes = ['Small', 'Medium', 'Large']
gifts = ['Tote Bag', 'Socks', 'Keychain', 'Stickers']

# Create stores
store_ids = [f"S{str(i).zfill(3)}" for i in range(1, num_stores + 1)]
store_df = pd.DataFrame({
    'store_id': store_ids,
    'store_location': np.random.choice(locations, num_stores, p=[0.4, 0.4, 0.2]),
    'store_size_category': np.random.choice(sizes, num_stores, p=[0.3, 0.5, 0.2]),
    'avg_revenue_last_3_months': np.random.normal(120000, 25000, num_stores).round(0),
    'monthly_active_customers': np.random.randint(500, 3000, num_stores),
    'campaign_group': np.random.choice(campaigns, size=num_stores, p=[0.33, 0.34, 0.33])
})

# Assign customers to stores
customer_ids = np.arange(1, num_customers + 1)
customer_store_map = pd.DataFrame({
    'customer_id': customer_ids,
    'store_id': np.random.choice(store_df['store_id'], num_customers)
})
customer_store_map = customer_store_map.merge(store_df, on='store_id', how='left')

# Generate weekly transactional data
all_data = []
for week in weeks:
    temp = customer_store_map.copy()
    temp['week'] = week

    # Simulate purchases
    temp['purchase_flag'] = np.random.binomial(1, 0.8)  # 25% purchase probability
    temp = temp[temp['purchase_flag'] == 1].copy()  # Only keep rows with purchase

    temp['visit_count'] = np.random.randint(1, 3, size=len(temp))
    temp['products_ord'] = np.random.randint(1, 5, size=len(temp))
    temp['new_collection_items'] = np.random.binomial(temp['products_ord'], 0.6).clip(0, temp['products_ord'])

    # Apply campaign logic
    temp['discount'] = 0.0
    temp['gift_received'] = np.nan
    temp['used_loyalty_or_voucher'] = False

    for campaign in campaigns:
        idx = temp['campaign_group'] == campaign
        if campaign == 'A':
            temp.loc[idx & (temp['new_collection_items'] > 0), 'discount'] = 0.05
        elif campaign == 'B':
            gift_idx = idx & (temp['new_collection_items'] > 0)
            temp.loc[gift_idx, 'gift_received'] = np.random.choice(gifts, gift_idx.sum())
        elif campaign == 'C':
            idx_c = idx & (np.random.rand(len(temp)) < 0.5)  # 50% eligible
            temp.loc[idx_c, 'used_loyalty_or_voucher'] = True
            temp.loc[idx_c, 'discount'] = 0.10

    # Revenue calculation
    temp['unit_price'] = np.random.normal(45, 10, size=len(temp)).clip(10, 100)
    temp['revenue_before_discount'] = temp['products_ord'] * temp['unit_price']
    temp['revenue'] = (temp['revenue_before_discount'] * (1 - temp['discount'])).round(2)

    # Demographics for active purchasers only
    temp['age'] = np.random.normal(32, 10, len(temp)).clip(18, 60).round()
    temp['gender'] = np.random.choice(['Male', 'Female'], len(temp))

    # Segment by logic (week 1 → some New, others Returning)
    temp['customer_segment'] = np.where((temp['week'] == 1),
                                        np.random.choice(['New', 'Returning'], len(temp), p=[0.3, 0.7]),
                                        'Returning')

    temp['promo_used'] = np.where(temp['discount'] > 0, 'Yes',
                                  np.where(temp['gift_received'].notna(), 'Yes', 'No'))

    temp['other_products_ord'] = temp['products_ord'] - temp['new_collection_items']
    all_data.append(temp)

# Combine data
final_df = pd.concat(all_data, ignore_index=True)

# Finalize columns
final_df = final_df[[
    'customer_id', 'week', 'store_id', 'campaign_group', 'store_location', 'store_size_category',
    'avg_revenue_last_3_months', 'monthly_active_customers', 'customer_segment', 'age', 'gender',
    'visit_count', 'products_ord', 'new_collection_items', 'other_products_ord',
    'discount', 'gift_received', 'used_loyalty_or_voucher', 'promo_used',
    'revenue_before_discount', 'revenue'
]]

# Save to CSV
final_df.to_csv("ab_marketing_campaign_cleaned.csv", index=False)
print("✅ Dataset saved as ab_marketing_campaign_cleaned.csv")