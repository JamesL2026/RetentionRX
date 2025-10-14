#!/usr/bin/env python3
"""
Flexible Analytics Framework for RetentionRx
Handles different types of datasets beyond churn prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import lightgbm as lgb

def detect_dataset_type(df):
    """Automatically detect what type of analysis this dataset is suitable for"""
    
    # Check for common target columns
    target_columns = {
        'churn': ['churn', 'churned', 'is_churned', 'churn_flag', 'status'],
        'conversion': ['converted', 'conversion', 'purchase', 'sale', 'bought'],
        'satisfaction': ['satisfaction', 'nps', 'rating', 'score', 'review_score'],
        'revenue': ['revenue', 'spend', 'purchase_amount', 'value', 'total_spent'],
        'engagement': ['engagement', 'activity', 'usage', 'logins', 'sessions'],
        'retention': ['retained', 'active', 'subscription', 'renewal', 'lifetime']
    }
    
    detected_types = []
    
    for analysis_type, possible_cols in target_columns.items():
        for col in df.columns:
            if any(target_col in col.lower() for target_col in possible_cols):
                detected_types.append(analysis_type)
                break
    
    return detected_types

def create_analysis_config(analysis_type):
    """Create configuration for different types of analysis"""
    
    configs = {
        'churn': {
            'name': 'Customer Churn Analysis',
            'description': 'Predict which customers are likely to leave',
            'target_description': 'Binary: 1 = churned, 0 = retained',
            'business_impact': 'Reduce customer loss and increase retention',
            'recommended_actions': [
                'Identify at-risk customers early',
                'Create targeted retention campaigns',
                'Address root causes of churn',
                'Improve customer experience'
            ]
        },
        'conversion': {
            'name': 'Conversion Prediction',
            'description': 'Predict which prospects will convert to customers',
            'target_description': 'Binary: 1 = converted, 0 = did not convert',
            'business_impact': 'Increase conversion rates and optimize marketing spend',
            'recommended_actions': [
                'Focus sales efforts on high-probability prospects',
                'Optimize marketing campaigns',
                'Improve lead qualification process',
                'Personalize outreach strategies'
            ]
        },
        'satisfaction': {
            'name': 'Customer Satisfaction Analysis',
            'description': 'Predict and improve customer satisfaction scores',
            'target_description': 'Numerical: 1-10 satisfaction rating',
            'business_impact': 'Improve customer experience and loyalty',
            'recommended_actions': [
                'Identify satisfaction drivers',
                'Address low-satisfaction customers',
                'Replicate high-satisfaction experiences',
                'Monitor satisfaction trends'
            ]
        },
        'revenue': {
            'name': 'Revenue Prediction',
            'description': 'Predict customer revenue and spending patterns',
            'target_description': 'Numerical: Revenue or spending amount',
            'business_impact': 'Optimize revenue forecasting and pricing',
            'recommended_actions': [
                'Identify high-value customers',
                'Predict revenue trends',
                'Optimize pricing strategies',
                'Focus on revenue growth opportunities'
            ]
        },
        'engagement': {
            'name': 'Engagement Analysis',
            'description': 'Analyze and predict customer engagement levels',
            'target_description': 'Numerical: Engagement score or activity level',
            'business_impact': 'Increase customer engagement and product adoption',
            'recommended_actions': [
                'Identify engagement patterns',
                'Re-engage inactive customers',
                'Optimize product features',
                'Improve user onboarding'
            ]
        },
        'retention': {
            'name': 'Retention Analysis',
            'description': 'Predict customer retention and lifetime value',
            'target_description': 'Binary: 1 = retained, 0 = not retained',
            'business_impact': 'Improve customer lifetime value and reduce churn',
            'recommended_actions': [
                'Identify retention drivers',
                'Create retention strategies',
                'Monitor retention metrics',
                'Optimize customer lifecycle'
            ]
        }
    }
    
    return configs.get(analysis_type, {
        'name': 'Custom Analysis',
        'description': 'Analyze your specific dataset',
        'target_description': 'Custom target variable',
        'business_impact': 'Gain insights from your data',
        'recommended_actions': [
            'Explore data patterns',
            'Identify key insights',
            'Create actionable recommendations',
            'Monitor performance metrics'
        ]
    })

def flexible_model_training(df, target_column, analysis_type):
    """Train appropriate model based on analysis type"""
    
    # Prepare features (exclude target and ID columns)
    feature_columns = [col for col in df.columns if col not in [target_column, 'customer_id', 'id']]
    X = df[feature_columns]
    y = df[target_column]
    
    # Handle missing values
    X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
    y = y.fillna(y.median() if y.dtype in ['int64', 'float64'] else y.mode().iloc[0])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Choose model based on analysis type
    if analysis_type in ['churn', 'conversion', 'retention']:
        # Binary classification
        model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, feature_importance, {'accuracy': accuracy}, y_pred_proba
        
    elif analysis_type in ['satisfaction', 'revenue', 'engagement']:
        # Regression
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return model, feature_importance, {'rmse': rmse, 'mse': mse}, y_pred

def create_flexible_visualizations(df, target_column, analysis_type, predictions=None):
    """Create visualizations appropriate for the analysis type"""
    
    charts = []
    
    if analysis_type in ['churn', 'conversion', 'retention']:
        # Distribution of target variable
        fig_dist = px.histogram(
            df, 
            x=target_column, 
            title=f"Distribution of {target_column.replace('_', ' ').title()}",
            nbins=2
        )
        charts.append(('distribution', fig_dist))
        
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r"
            )
            charts.append(('correlation', fig_corr))
    
    elif analysis_type in ['satisfaction', 'revenue', 'engagement']:
        # Distribution of target variable
        fig_dist = px.histogram(
            df, 
            x=target_column, 
            title=f"Distribution of {target_column.replace('_', ' ').title()}",
            nbins=20
        )
        charts.append(('distribution', fig_dist))
        
        # Box plot by categorical variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_col = categorical_cols[0]
            fig_box = px.box(
                df,
                x=cat_col,
                y=target_column,
                title=f"{target_column.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}"
            )
            charts.append(('box_plot', fig_box))
    
    return charts

def generate_business_insights(df, target_column, analysis_type, feature_importance, metrics):
    """Generate business insights based on analysis type"""
    
    insights = []
    
    if analysis_type == 'churn':
        churn_rate = df[target_column].mean()
        insights.append(f"**Current churn rate:** {churn_rate:.1%}")
        
        if 'accuracy' in metrics:
            insights.append(f"**Model accuracy:** {metrics['accuracy']:.1%}")
        
        top_feature = feature_importance.iloc[0]['feature']
        insights.append(f"**Top churn predictor:** {top_feature.replace('_', ' ').title()}")
        
    elif analysis_type == 'conversion':
        conversion_rate = df[target_column].mean()
        insights.append(f"**Current conversion rate:** {conversion_rate:.1%}")
        
        if 'accuracy' in metrics:
            insights.append(f"**Model accuracy:** {metrics['accuracy']:.1%}")
        
        top_feature = feature_importance.iloc[0]['feature']
        insights.append(f"**Top conversion predictor:** {top_feature.replace('_', ' ').title()}")
        
    elif analysis_type in ['satisfaction', 'revenue', 'engagement']:
        avg_value = df[target_column].mean()
        insights.append(f"**Average {target_column.replace('_', ' ')}:** {avg_value:.2f}")
        
        if 'rmse' in metrics:
            insights.append(f"**Prediction error (RMSE):** {metrics['rmse']:.2f}")
        
        top_feature = feature_importance.iloc[0]['feature']
        insights.append(f"**Top predictor:** {top_feature.replace('_', ' ').title()}")
    
    return insights

def flexible_analytics_tab():
    """Flexible analytics tab for different dataset types"""
    st.subheader("🔍 Flexible Analytics - Beyond Churn")
    
    if st.session_state.current_df is not None:
        df = st.session_state.current_df
        
        # Detect dataset type
        detected_types = detect_dataset_type(df)
        
        st.write("**Detected Analysis Types:**")
        if detected_types:
            for analysis_type in detected_types:
                config = create_analysis_config(analysis_type)
                st.write(f"• **{config['name']}** - {config['description']}")
        else:
            st.info("No specific analysis type detected. Using general analysis.")
            detected_types = ['custom']
        
        # Let user select analysis type
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            options=detected_types if detected_types else ['custom'],
            format_func=lambda x: create_analysis_config(x)['name']
        )
        
        config = create_analysis_config(analysis_type)
        
        # Show analysis configuration
        with st.expander(f"📋 {config['name']} Configuration"):
            st.write(f"**Description:** {config['description']}")
            st.write(f"**Target:** {config['target_description']}")
            st.write(f"**Business Impact:** {config['business_impact']}")
            
            st.write("**Recommended Actions:**")
            for action in config['recommended_actions']:
                st.write(f"• {action}")
        
        # Select target column
        target_column = st.selectbox(
            "Select Target Column:",
            options=df.columns,
            help="Choose the column you want to predict or analyze"
        )
        
        # Show target column info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            if df[target_column].dtype in ['int64', 'float64']:
                st.metric("Average Value", f"{df[target_column].mean():.2f}")
            else:
                st.metric("Unique Values", df[target_column].nunique())
        with col3:
            st.metric("Missing Values", df[target_column].isnull().sum())
        
        # Train model
        if st.button(f"🚀 Train {config['name']} Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    model, feature_importance, metrics, predictions = flexible_model_training(
                        df, target_column, analysis_type
                    )
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Show metrics
                    st.subheader("📊 Model Performance")
                    for metric, value in metrics.items():
                        st.metric(metric.replace('_', ' ').title(), f"{value:.3f}")
                    
                    # Feature importance
                    st.subheader("🔍 Feature Importance")
                    fig_importance = px.bar(
                        feature_importance.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Most Important Features"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Business insights
                    st.subheader("💡 Business Insights")
                    insights = generate_business_insights(df, target_column, analysis_type, feature_importance, metrics)
                    for insight in insights:
                        st.write(insight)
                    
                    # Visualizations
                    st.subheader("📈 Data Visualizations")
                    charts = create_flexible_visualizations(df, target_column, analysis_type, predictions)
                    
                    for chart_name, fig in charts:
                        st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    st.info("Please check that your target column has appropriate data types and values.")
        
        # Show data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
    else:
        st.info("👆 Please load data in the Churn Prediction tab first")

# Example usage in main app
def add_flexible_analytics_to_app():
    """Add flexible analytics as a new tab"""
    # This would be called from the main app to add the flexible analytics tab
    pass
