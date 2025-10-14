# 📊 Data Preparation Guide for RetentionRx

## 🎯 **Flexible Column Mapping**

RetentionRx now automatically recognizes many common column name variations! Here are the supported formats:

### **Customer ID**
- `customer_id`, `id`, `customer`, `user_id`, `client_id`, `account_id`

### **Tenure (Days)**
- `tenure_days`, `tenure`, `days_since_signup`, `customer_age_days`, `account_age`

### **Monthly Spend**
- `monthly_spend`, `spend`, `revenue`, `monthly_revenue`, `avg_spend`, `spending`

### **Support Tickets**
- `num_tickets_30d`, `tickets`, `support_tickets`, `tickets_30d`, `support_calls`

### **Customer Satisfaction**
- `avg_nps`, `nps`, `satisfaction`, `customer_satisfaction`, `satisfaction_score`

### **Last Purchase**
- `last_purchase_days_ago`, `days_since_purchase`, `last_order`, `days_since_order`

### **Churn Label**
- `churn`, `churned`, `is_churned`, `churn_flag`, `retained`, `status`

---

## 🔧 **Data Transformation Examples**

### **Example 1: Salesforce Export**
```csv
Account_ID,Account_Age_Days,Monthly_Revenue,Support_Calls_30D,Customer_Satisfaction,Days_Since_Last_Order,Churn_Flag
ACC001,540,120.0,1,8,30,0
ACC002,45,60.0,5,3,90,1
```

**✅ This will work automatically!** No changes needed.

### **Example 2: Zendesk Export**
```csv
user_id,tenure,spend,tickets,nps,last_order,churned
U001,540,120.0,1,8,30,0
U002,45,60.0,5,3,90,1
```

**✅ This will work automatically!** No changes needed.

### **Example 3: Custom CRM**
```csv
client_id,account_age,revenue,support_tickets,satisfaction,days_since_order,status
C001,540,120.0,1,8,30,retained
C002,45,60.0,5,3,90,churned
```

**⚠️ Needs minor adjustment:** Change `status` values from "retained"/"churned" to 0/1

---

## 📋 **Data Requirements**

### **Required Columns (any of these names work):**
1. **Customer ID** - Unique identifier
2. **Tenure** - Days since first purchase/signup
3. **Monthly Spend** - Average monthly revenue/spending
4. **Support Tickets** - Number of support requests in last 30 days
5. **Customer Satisfaction** - NPS score or satisfaction rating (1-10)
6. **Last Purchase** - Days since most recent purchase
7. **Churn Label** - Binary indicator (0=retained, 1=churned)

### **Data Types:**
- **Customer ID**: String or numeric
- **Tenure**: Numeric (days)
- **Monthly Spend**: Numeric (currency)
- **Support Tickets**: Integer (count)
- **Customer Satisfaction**: Numeric (1-10 scale)
- **Last Purchase**: Numeric (days)
- **Churn Label**: Binary (0 or 1)

---

## 🚀 **Quick Data Prep Steps**

### **Step 1: Export from Your System**
- Export customer data with the last 6-12 months
- Include all customers (churned and retained)
- Ensure churn label is accurate

### **Step 2: Check Column Names**
- Upload to RetentionRx
- If columns aren't recognized, the app will show suggestions
- Rename columns to match suggested variations

### **Step 3: Handle Missing Data**
- **Missing values**: App handles automatically (fills with median/mode)
- **Invalid values**: Remove or correct before upload
- **Outliers**: App is robust to outliers, but extreme values may affect results

### **Step 4: Data Quality Check**
- **Minimum 50 customers** for reliable results
- **At least 10% churn rate** for meaningful predictions
- **Recent data** (within last 6 months) for accuracy

---

## 🔍 **Common Data Issues & Solutions**

### **Issue: "Missing required columns"**
**Solution:** Check the column mapping suggestions in the app and rename your columns

### **Issue: "Invalid data types"**
**Solution:** Ensure numeric columns contain only numbers, churn column has 0/1 values

### **Issue: "No churn data"**
**Solution:** Create churn labels based on:
- No activity for 90+ days
- Explicit cancellation
- Account closure

### **Issue: "Low model accuracy"**
**Solution:** 
- Check data quality
- Ensure sufficient sample size (100+ customers)
- Verify churn labels are accurate

---

## 📊 **Sample Data Formats**

### **Format 1: Standard (sample_data.csv)**
```csv
customer_id,tenure_days,monthly_spend,num_tickets_30d,avg_nps,last_purchase_days_ago,churn
```

### **Format 2: Alternative (sample_data_alternative.csv)**
```csv
id,account_age,revenue,tickets,satisfaction,days_since_order,churned
```

### **Format 3: Enterprise CRM**
```csv
Account_ID,Account_Age_Days,Monthly_Revenue,Support_Calls_30D,Customer_Satisfaction,Days_Since_Last_Order,Churn_Flag
```

**All formats work automatically!** 🎉

---

## 💡 **Pro Tips**

1. **Start with sample data** to test the app
2. **Use descriptive column names** for easier mapping
3. **Include recent data** for better predictions
4. **Export 6-12 months** of historical data
5. **Ensure balanced churn data** (not all 0s or 1s)

**Need help?** The app shows detailed error messages and suggestions for fixing data issues!
