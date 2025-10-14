# 🔮 RetentionRx - Customer Churn Prediction & Playbook Generator

**Find customers likely to churn, explain why, and generate actionable rescue plans**

RetentionRx is a lightweight tool that demonstrates how predictive analytics + explainability + controlled LLM can turn raw CRM/support data into operational actions that reduce churn. Built for quick pilots so CX teams can move from insight to action within days.

## 🎯 Key Features

- **Churn Prediction**: LightGBM model with AUC scoring
- **Feature Importance**: Understand what drives churn risk
- **Risk Segmentation**: Identify and prioritize high-risk customers
- **AI Playbook Generation**: LLM-powered rescue plans with fallback
- **Export Ready**: JSON output for integration with ServiceNow/Zendesk
- **Demo Friendly**: Works offline with sample data

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone or download the project**
```bash
git clone <repository-url>
cd retention-rx
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

## 📊 Data Requirements

Your CSV should have these columns:
- `customer_id`: Unique customer identifier
- `tenure_days`: Number of days since first purchase
- `monthly_spend`: Average monthly spending
- `num_tickets_30d`: Support tickets in last 30 days
- `avg_nps`: Average Net Promoter Score
- `last_purchase_days_ago`: Days since last purchase
- `churn`: Binary label (0=retained, 1=churned)

### Sample Data
The app includes sample data with 50 customers for immediate testing.

## 🔧 Configuration

### OpenAI API (Optional)
- Enter your OpenAI API key in the sidebar for AI-generated playbooks
- Without API key, the app uses a canned playbook (fully offline demo)

### Model Settings
- Uses LightGBM for churn prediction
- AUC score displayed for model performance
- Feature importance visualization included

## 📋 Demo Flow (5 minutes)

1. **Data Input**: Use sample data or upload your CSV
2. **Train Model**: Click "Train Churn Prediction Model"
3. **Review Performance**: Check AUC score and feature importance
4. **Set Threshold**: Adjust churn probability threshold
5. **Generate Playbook**: Click "Generate Playbook"
6. **Export**: Download JSON or copy for integration

## 🎯 Generated Playbook Structure

```json
{
  "priority_actions": [
    "Outbound 2-minute check-in call to top 50 customers...",
    "Auto-create high-priority ticket for customers with 3+ tickets...",
    "Offer targeted 20% promo or expedited shipping..."
  ],
  "agent_script": "Hi [Name], I'm [Agent]. I see you had issues recently...",
  "ops_fixes": [
    "Reduce support SLA for top-risk segment to 24 hours",
    "Flag accounts with 2+ tickets for supervisor review"
  ],
  "kpis_to_track": [
    "Churn rate for pilot group (30/60/90 days)",
    "Average resolution time for flagged tickets"
  ],
  "assumptions": [
    "Pilot targets top 100 high-risk customers",
    "Average revenue/customer = $100/month",
    "Expected conservative churn reduction = 10%"
  ]
}
```

## 📈 ROI Calculation

The app automatically calculates:
- Number of high-risk customers
- Monthly revenue at risk
- Expected savings from 10% churn reduction
- Quick ROI metrics for pilot justification

## 🔗 Integration Points

### Ready for Enterprise Connectors:
- **CRM**: Salesforce, Dynamics 365
- **Ticketing**: ServiceNow, Zendesk, Jira
- **Analytics**: Tableau, Power BI
- **Process Mining**: Celonis integration

### Export Formats:
- JSON playbooks for API integration
- CSV exports for data analysis
- Copy-paste ready for manual workflows

## 🏗️ Architecture

- **Frontend**: Streamlit single-page app
- **ML**: LightGBM + scikit-learn
- **AI**: OpenAI GPT-3.5-turbo (optional)
- **Data**: CSV input/output
- **Deployment**: Local, Docker-ready

## 🔒 Security & Privacy

- **Demo Mode**: Uses tokenized sample data (no PII)
- **API Keys**: Stored in session state only
- **Data Processing**: All local, no data sent externally (except OpenAI API calls)
- **Enterprise Ready**: Documented security considerations for production

## 📊 Success Metrics

### Product Demo
- ✅ Working demo generates playbook in <2 minutes
- ✅ Playbook output in valid JSON format
- ✅ Offline capability with sample data

### Business (Pilot Level)
- **Precision@top100**: Target 0.45 (tuneable to dataset)
- **Churn Reduction**: Target 5-15% (client-dependent)
- **ROI**: Clear, simple math for pilot justification

## 🛠️ Customization

### Adding New Features
- Modify `app.py` for UI changes
- Update `train_churn_model()` for different ML models
- Extend `generate_playbook_llm()` for custom prompts

### Model Tuning
- Adjust LightGBM parameters in `train_churn_model()`
- Add feature engineering in data preprocessing
- Implement model validation and retraining

## 🚀 Roadmap

### 0-3 months (Current MVP)
- ✅ Streamlit demo with sample CSV
- ✅ LLM playbook generation
- ✅ Canned playbook fallback

### 3-6 months
- CRM connectors (Salesforce, Zendesk)
- Basic MLOps (retrain pipelines)
- Admin UI for model management

### 6-12 months
- Process mining integration
- Push-to-ticketing workflows
- RBAC and enterprise deployment

## 🤝 Support

For questions or issues:
1. Check the demo flow above
2. Verify your CSV has required columns
3. Ensure Python dependencies are installed
4. Try the offline mode with sample data

## 📄 License

This project is for demonstration purposes. See PRD for enterprise deployment considerations.

---

**Built for CX teams who want to move from insight to action within days, not months.**
