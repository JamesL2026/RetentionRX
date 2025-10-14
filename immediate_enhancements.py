#!/usr/bin/env python3
"""
RetentionRx Immediate Enhancements
Quick wins to make the product more powerful and insightful
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_rfm_analysis(df):
    """Calculate RFM (Recency, Frequency, Monetary) analysis"""
    # Calculate RFM metrics
    rfm_df = df.copy()
    
    # Recency: Days since last purchase (lower is better)
    rfm_df['Recency'] = df['last_purchase_days_ago']
    
    # Frequency: Number of purchases (higher is better) - approximated by tenure/months
    rfm_df['Frequency'] = np.maximum(1, df['tenure_days'] / 30)  # Approximate monthly frequency
    
    # Monetary: Average monthly spend (higher is better)
    rfm_df['Monetary'] = df['monthly_spend']
    
    # Create RFM scores (1-5 scale)
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5,4,3,2,1])  # Inverted for recency
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1,2,3,4,5])
    
    # Create RFM segment
    rfm_df['RFM_Segment'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
    
    # Define customer segments
    segment_map = {
        '555': 'Champions', '554': 'Champions', '544': 'Champions', '545': 'Champions', '454': 'Champions',
        '455': 'Champions', '445': 'Champions', '444': 'Champions', '543': 'Loyal Customers',
        '444': 'Loyal Customers', '435': 'Loyal Customers', '355': 'Loyal Customers', '354': 'Loyal Customers',
        '345': 'Loyal Customers', '344': 'Loyal Customers', '335': 'Loyal Customers', '334': 'Loyal Customers',
        '325': 'Loyal Customers', '324': 'Loyal Customers', '234': 'Potential Loyalists',
        '233': 'Potential Loyalists', '232': 'Potential Loyalists', '223': 'Potential Loyalists',
        '222': 'Potential Loyalists', '221': 'Potential Loyalists', '212': 'Potential Loyalists',
        '211': 'Potential Loyalists', '111': 'New Customers', '112': 'New Customers',
        '121': 'New Customers', '122': 'New Customers', '123': 'New Customers',
        '132': 'New Customers', '213': 'New Customers', '214': 'New Customers',
        '215': 'New Customers', '314': 'New Customers', '313': 'New Customers',
        '312': 'New Customers', '311': 'New Customers', '411': 'New Customers',
        '412': 'New Customers', '413': 'New Customers', '414': 'New Customers',
        '415': 'New Customers', '511': 'New Customers', '512': 'New Customers',
        '513': 'New Customers', '514': 'New Customers', '515': 'New Customers',
        '521': 'At Risk', '522': 'At Risk', '523': 'At Risk', '524': 'At Risk',
        '525': 'At Risk', '531': 'At Risk', '532': 'At Risk', '533': 'At Risk',
        '534': 'At Risk', '535': 'At Risk', '541': 'At Risk', '542': 'At Risk',
        '543': 'At Risk', '544': 'At Risk', '545': 'At Risk', '421': 'Cannot Lose Them',
        '422': 'Cannot Lose Them', '423': 'Cannot Lose Them', '424': 'Cannot Lose Them',
        '425': 'Cannot Lose Them', '431': 'Cannot Lose Them', '432': 'Cannot Lose Them',
        '433': 'Cannot Lose Them', '434': 'Cannot Lose Them', '435': 'Cannot Lose Them',
        '441': 'Cannot Lose Them', '442': 'Cannot Lose Them', '443': 'Cannot Lose Them',
        '444': 'Cannot Lose Them', '445': 'Cannot Lose Them', '451': 'Cannot Lose Them',
        '452': 'Cannot Lose Them', '453': 'Cannot Lose Them', '454': 'Cannot Lose Them',
        '455': 'Cannot Lose Them', '521': 'At Risk', '522': 'At Risk', '523': 'At Risk',
        '524': 'At Risk', '525': 'At Risk', '531': 'At Risk', '532': 'At Risk',
        '533': 'At Risk', '534': 'At Risk', '535': 'At Risk', '541': 'At Risk',
        '542': 'At Risk', '543': 'At Risk', '544': 'At Risk', '545': 'At Risk',
        '321': 'Need Attention', '322': 'Need Attention', '323': 'Need Attention',
        '324': 'Need Attention', '325': 'Need Attention', '331': 'Need Attention',
        '332': 'Need Attention', '333': 'Need Attention', '334': 'Need Attention',
        '335': 'Need Attention', '341': 'Need Attention', '342': 'Need Attention',
        '343': 'Need Attention', '344': 'Need Attention', '345': 'Need Attention',
        '351': 'Need Attention', '352': 'Need Attention', '353': 'Need Attention',
        '354': 'Need Attention', '355': 'Need Attention', '411': 'Need Attention',
        '412': 'Need Attention', '413': 'Need Attention', '414': 'Need Attention',
        '415': 'Need Attention', '421': 'Need Attention', '422': 'Need Attention',
        '423': 'Need Attention', '424': 'Need Attention', '425': 'Need Attention',
        '431': 'Need Attention', '432': 'Need Attention', '433': 'Need Attention',
        '434': 'Need Attention', '435': 'Need Attention', '441': 'Need Attention',
        '442': 'Need Attention', '443': 'Need Attention', '444': 'Need Attention',
        '445': 'Need Attention', '451': 'Need Attention', '452': 'Need Attention',
        '453': 'Need Attention', '454': 'Need Attention', '455': 'Need Attention'
    }
    
    rfm_df['Customer_Segment'] = rfm_df['RFM_Segment'].map(segment_map).fillna('Others')
    
    return rfm_df

def calculate_customer_lifetime_value(df):
    """Calculate Customer Lifetime Value (CLV)"""
    # Simple CLV calculation: (Monthly Spend × Average Lifespan in months)
    # Lifespan approximated by tenure for retained customers
    clv_df = df.copy()
    
    # For retained customers, use current tenure as lifespan
    # For churned customers, use tenure at churn
    clv_df['CLV'] = clv_df['monthly_spend'] * (clv_df['tenure_days'] / 30)
    
    # Add CLV prediction for at-risk customers
    clv_df['Predicted_CLV'] = clv_df['CLV'] * (1 - clv_df.get('churn_probability', 0))
    
    return clv_df

def calculate_advanced_metrics(df):
    """Calculate advanced customer metrics"""
    metrics_df = df.copy()
    
    # Customer Health Score (0-100)
    # Based on multiple factors: tenure, spend, satisfaction, activity
    health_score = (
        (1 - metrics_df['last_purchase_days_ago'] / 365) * 25 +  # Recency (25%)
        (metrics_df['avg_nps'] / 10) * 25 +  # Satisfaction (25%)
        (metrics_df['monthly_spend'] / metrics_df['monthly_spend'].max()) * 25 +  # Spend (25%)
        (1 - metrics_df['num_tickets_30d'] / 10) * 25  # Low ticket count (25%)
    ) * 100
    
    metrics_df['Health_Score'] = np.clip(health_score, 0, 100)
    
    # Engagement Score
    engagement_score = (
        (metrics_df['avg_nps'] / 10) * 40 +  # Satisfaction (40%)
        (1 - metrics_df['num_tickets_30d'] / 10) * 30 +  # Low support needs (30%)
        (1 - metrics_df['last_purchase_days_ago'] / 365) * 30  # Recent activity (30%)
    ) * 100
    
    metrics_df['Engagement_Score'] = np.clip(engagement_score, 0, 100)
    
    # Risk Score (0-100, higher = more risky)
    risk_score = (
        (metrics_df['num_tickets_30d'] / 10) * 30 +  # High ticket count (30%)
        (metrics_df['last_purchase_days_ago'] / 365) * 25 +  # Inactivity (25%)
        (1 - metrics_df['avg_nps'] / 10) * 25 +  # Low satisfaction (25%)
        (1 - metrics_df['monthly_spend'] / metrics_df['monthly_spend'].max()) * 20  # Low spend (20%)
    ) * 100
    
    metrics_df['Risk_Score'] = np.clip(risk_score, 0, 100)
    
    return metrics_df

def generate_segment_insights(df):
    """Generate insights for each customer segment"""
    insights = {}
    
    for segment in df['Customer_Segment'].unique():
        segment_data = df[df['Customer_Segment'] == segment]
        
        insights[segment] = {
            'count': len(segment_data),
            'avg_spend': segment_data['monthly_spend'].mean(),
            'avg_tenure': segment_data['tenure_days'].mean(),
            'avg_nps': segment_data['avg_nps'].mean(),
            'churn_rate': segment_data['churn'].mean(),
            'total_revenue': segment_data['monthly_spend'].sum(),
            'recommendations': get_segment_recommendations(segment)
        }
    
    return insights

def get_segment_recommendations(segment):
    """Get recommendations for each customer segment"""
    recommendations = {
        'Champions': [
            'Upsell premium products/services',
            'Ask for referrals and testimonials',
            'Provide VIP customer experience',
            'Invite to beta testing programs'
        ],
        'Loyal Customers': [
            'Increase engagement with loyalty programs',
            'Offer personalized recommendations',
            'Provide exclusive early access',
            'Build stronger brand connection'
        ],
        'Potential Loyalists': [
            'Improve customer experience',
            'Offer incentives for repeat purchases',
            'Provide better onboarding',
            'Increase product education'
        ],
        'New Customers': [
            'Focus on onboarding and education',
            'Provide excellent first-time experience',
            'Set clear expectations',
            'Monitor early engagement closely'
        ],
        'At Risk': [
            'Immediate retention campaigns',
            'Personal outreach and support',
            'Offer special discounts or incentives',
            'Address specific pain points'
        ],
        'Cannot Lose Them': [
            'Executive-level attention',
            'Custom solutions and support',
            'Priority handling for all requests',
            'Regular check-ins and relationship building'
        ],
        'Need Attention': [
            'Re-engagement campaigns',
            'Product usage education',
            'Address support issues quickly',
            'Provide additional value'
        ]
    }
    
    return recommendations.get(segment, ['Monitor closely', 'Provide standard support'])

def create_advanced_visualizations(df):
    """Create advanced visualizations"""
    # RFM Analysis Heatmap
    rfm_pivot = df.groupby(['R_Score', 'F_Score']).size().unstack(fill_value=0)
    
    fig_rfm = px.imshow(
        rfm_pivot.values,
        labels=dict(x="Frequency Score", y="Recency Score", color="Number of Customers"),
        x=rfm_pivot.columns,
        y=rfm_pivot.index,
        title="RFM Analysis Heatmap",
        color_continuous_scale="RdYlBu_r"
    )
    
    # Customer Segment Distribution
    segment_counts = df['Customer_Segment'].value_counts()
    fig_segments = px.pie(
        values=segment_counts.values,
        names=segment_counts.index,
        title="Customer Segment Distribution"
    )
    
    # Health Score vs Risk Score Scatter
    fig_health = px.scatter(
        df,
        x='Risk_Score',
        y='Health_Score',
        color='Customer_Segment',
        size='monthly_spend',
        hover_data=['customer_id', 'tenure_days', 'avg_nps'],
        title="Customer Health vs Risk Analysis",
        labels={'Risk_Score': 'Risk Score', 'Health_Score': 'Health Score'}
    )
    
    return fig_rfm, fig_segments, fig_health

# Example usage in Streamlit app
def add_advanced_analytics_section(df):
    """Add advanced analytics section to the main app"""
    st.subheader("🔍 Advanced Customer Analytics")
    
    # Calculate advanced metrics
    df_with_metrics = calculate_advanced_metrics(df)
    df_with_rfm = calculate_rfm_analysis(df_with_metrics)
    df_with_clv = calculate_customer_lifetime_value(df_with_rfm)
    
    # Generate insights
    insights = generate_segment_insights(df_with_clv)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Health Score", f"{df_with_clv['Health_Score'].mean():.1f}")
    with col2:
        st.metric("Avg Risk Score", f"{df_with_clv['Risk_Score'].mean():.1f}")
    with col3:
        st.metric("Avg CLV", f"${df_with_clv['CLV'].mean():.0f}")
    with col4:
        st.metric("Total Segments", len(df_with_clv['Customer_Segment'].unique()))
    
    # Create visualizations
    fig_rfm, fig_segments, fig_health = create_advanced_visualizations(df_with_clv)
    
    # Display charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_rfm, use_container_width=True)
    with col2:
        st.plotly_chart(fig_segments, use_container_width=True)
    
    st.plotly_chart(fig_health, use_container_width=True)
    
    # Segment insights
    st.subheader("📊 Segment Insights & Recommendations")
    
    for segment, data in insights.items():
        with st.expander(f"🎯 {segment} ({data['count']} customers)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Metrics:**")
                st.write(f"• Average Spend: ${data['avg_spend']:.2f}")
                st.write(f"• Average Tenure: {data['avg_tenure']:.0f} days")
                st.write(f"• Average NPS: {data['avg_nps']:.1f}")
                st.write(f"• Churn Rate: {data['churn_rate']:.1%}")
                st.write(f"• Total Revenue: ${data['total_revenue']:.0f}/month")
            
            with col2:
                st.write(f"**Recommendations:**")
                for rec in data['recommendations']:
                    st.write(f"• {rec}")
    
    return df_with_clv

if __name__ == "__main__":
    # Example usage
    print("Advanced Analytics Module Loaded")
    print("Ready to enhance RetentionRx with:")
    print("• RFM Analysis")
    print("• Customer Segmentation")
    print("• Lifetime Value Calculation")
    print("• Advanced Metrics")
    print("• Segment Insights")
