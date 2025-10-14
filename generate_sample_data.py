#!/usr/bin/env python3
"""
Generate random sample data for RetentionRx testing
"""

import pandas as pd
import numpy as np
import random

def generate_sample_data(num_customers=100, seed=42):
    """Generate realistic customer churn data"""
    np.random.seed(seed)
    random.seed(seed)
    
    data = []
    
    for i in range(num_customers):
        customer_id = f"C{i+1:03d}"
        
        # Generate realistic tenure (some new, some long-term)
        if np.random.random() < 0.3:  # 30% new customers
            tenure_days = np.random.randint(1, 90)
        else:
            tenure_days = np.random.randint(90, 1000)
        
        # Monthly spend correlates with tenure and NPS
        base_spend = 20 + tenure_days * 0.1
        monthly_spend = base_spend + np.random.normal(0, 20)
        monthly_spend = max(10, monthly_spend)  # Minimum $10
        
        # Tickets correlate with churn risk
        if tenure_days < 60:  # New customers more likely to have issues
            num_tickets = np.random.poisson(2)
        else:
            num_tickets = np.random.poisson(0.5)
        
        # NPS correlates with tenure and tickets
        base_nps = 5
        if tenure_days > 180:
            base_nps += 2
        if num_tickets == 0:
            base_nps += 1
        elif num_tickets > 2:
            base_nps -= 2
        
        avg_nps = max(1, min(10, base_nps + np.random.normal(0, 1)))
        
        # Last purchase correlates with churn risk
        if avg_nps > 7 and tenure_days > 180:
            last_purchase_days_ago = np.random.randint(1, 30)
        else:
            last_purchase_days_ago = np.random.randint(30, 200)
        
        # Churn probability based on multiple factors
        churn_prob = 0.1
        if tenure_days < 60:
            churn_prob += 0.3
        if num_tickets > 3:
            churn_prob += 0.4
        if avg_nps < 4:
            churn_prob += 0.3
        if last_purchase_days_ago > 90:
            churn_prob += 0.2
        
        churn_prob = min(0.95, max(0.05, churn_prob))
        churn = 1 if np.random.random() < churn_prob else 0
        
        data.append({
            'customer_id': customer_id,
            'tenure_days': tenure_days,
            'monthly_spend': round(monthly_spend, 2),
            'num_tickets_30d': num_tickets,
            'avg_nps': round(avg_nps, 1),
            'last_purchase_days_ago': last_purchase_days_ago,
            'churn': churn
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate different datasets
    print("Generating sample data...")
    
    # Fixed dataset (same as current)
    df_fixed = generate_sample_data(50, seed=42)
    df_fixed.to_csv('sample_data_fixed.csv', index=False)
    print(f"✅ Fixed dataset: {len(df_fixed)} customers, {df_fixed['churn'].mean():.1%} churn rate")
    
    # Random dataset (changes each time)
    df_random = generate_sample_data(50, seed=None)
    df_random.to_csv('sample_data_random.csv', index=False)
    print(f"✅ Random dataset: {len(df_random)} customers, {df_random['churn'].mean():.1%} churn rate")
    
    # Larger dataset for testing
    df_large = generate_sample_data(200, seed=123)
    df_large.to_csv('sample_data_large.csv', index=False)
    print(f"✅ Large dataset: {len(df_large)} customers, {df_large['churn'].mean():.1%} churn rate")
    
    print("\n🎯 Use these files in RetentionRx:")
    print("  - sample_data_fixed.csv (same results every time)")
    print("  - sample_data_random.csv (different each time)")
    print("  - sample_data_large.csv (200 customers for testing)")
