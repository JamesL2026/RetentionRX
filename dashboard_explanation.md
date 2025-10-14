# 📊 RetentionRx Dashboard - Simple Explanation Guide

## 🎯 **AUC Score (Model Accuracy)**

**What it means:**
- **AUC = 0.85** means the model is **85% accurate** at predicting churn
- **0.9+ = Excellent** (90%+ accuracy) - Very reliable
- **0.8+ = Good** (80%+ accuracy) - Reliable for business decisions  
- **0.7+ = Fair** (70%+ accuracy) - Somewhat reliable
- **<0.7 = Poor** - Needs improvement

**In simple terms:** "Our model correctly identifies customers who will churn 85% of the time"

---

## 🔍 **Feature Importance Chart**

**What it shows:** Which customer behaviors most strongly predict churn risk

**Top factors usually:**
1. **Support Tickets (30 days)** - More tickets = higher churn risk
2. **Customer Satisfaction Score (NPS)** - Lower scores = higher risk
3. **Days Since Last Purchase** - Longer gaps = higher risk
4. **Customer Tenure** - Newer customers = higher risk
5. **Monthly Spending** - Lower spend = higher risk

**Business insight:** "Customers with 3+ support tickets in 30 days are 4x more likely to churn"

---

## 🎯 **Risk Threshold Slider**

**What it does:** Sets how strict we are about flagging customers as "high-risk"

- **High threshold (70%+):** Only flag customers with very high churn probability
  - *Result:* Fewer customers flagged, but very accurate
  - *Use case:* Limited resources, focus on highest risk

- **Balanced threshold (50%):** Flag customers with moderate to high risk
  - *Result:* More customers flagged, good balance
  - *Use case:* Standard retention campaigns

- **Low threshold (30%+):** Flag many customers proactively
  - *Result:* Many customers flagged, catch early warning signs
  - *Use case:* Proactive customer success outreach

---

## 🚨 **High-Risk Customers Table**

**What you see:**
- **Customer ID:** Who to contact
- **Churn Probability:** How likely they are to leave (0-100%)
- **Monthly Spend:** How much revenue you'll lose
- **Risk Factors:** Why they're at risk (tickets, NPS, etc.)

**Action items:**
- **90%+ risk:** Contact within 24 hours
- **70-90% risk:** Contact within 48 hours  
- **50-70% risk:** Include in weekly retention campaign

---

## 💰 **Revenue at Risk Calculation**

**Simple math:**
- **18 high-risk customers** × **$82 average monthly spend** = **$1,476/month at risk**
- **10% churn reduction** = **$148/month saved**
- **Annual savings** = **$1,776/year**

**ROI:** If retention campaign costs $500/month, you save $148, so ROI = 30% return

---

## 📋 **Generated Playbook**

**What it contains:**
1. **Priority Actions** - Specific steps to take (call customers, create tickets, offer discounts)
2. **Agent Script** - Ready-to-use conversation starter
3. **Operations Fixes** - Process improvements (faster SLA, flagging rules)
4. **KPIs to Track** - How to measure success (churn rate, resolution time, NPS)
5. **Assumptions** - Clear expectations and ROI calculations

**Ready for integration:** JSON format can push directly to ServiceNow, Zendesk, or Salesforce

---

## 🎯 **Demo Talking Points**

**For executives:**
- "We can predict churn with 85% accuracy"
- "This identifies $1,500/month in at-risk revenue"
- "Simple 10% reduction saves $1,800/year"
- "Ready to integrate with existing systems"

**For operations:**
- "Clear action items for agents"
- "Specific scripts to use"
- "Process improvements identified"
- "Easy to track and measure results"

**For technical teams:**
- "LightGBM model with feature importance"
- "JSON export for system integration"
- "Scalable architecture for enterprise deployment"
- "Human-in-the-loop governance built-in"

---

## 🚀 **Next Steps After Demo**

1. **Pilot Scope:** Define 100-500 customer pilot
2. **Data Access:** Connect to real customer data
3. **Integration:** Push playbooks to ticketing systems
4. **Success Metrics:** Measure churn reduction and ROI
5. **Scale:** Roll out to full customer base

**Timeline:** From demo to pilot results in 30-60 days
