# 🚀 RetentionRx Deployment Guide

## 📋 **Deployment Options**

### **Option 1: Streamlit Cloud (Recommended - Free & Easy)**
### **Option 2: Firebase Hosting (Static Site)**
### **Option 3: Heroku (Full App)**
### **Option 4: Railway/Render (Modern Alternatives)**

---

## 🌟 **Option 1: Streamlit Cloud (FREE & EASIEST)**

### **Step 1: Upload to GitHub**
```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit files
git commit -m "Initial commit: RetentionRx Customer Analytics Platform"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/retention-rx.git

# Push to GitHub
git push -u origin main
```

### **Step 2: Deploy on Streamlit Cloud**
1. **Go to:** https://share.streamlit.io/
2. **Click:** "New app"
3. **Connect GitHub:** Authorize Streamlit to access your repos
4. **Select Repository:** Choose `retention-rx`
5. **Select Branch:** `main`
6. **Main file path:** `app.py`
7. **Click:** "Deploy!"

**✅ Done! Your app will be live at:** `https://YOUR_USERNAME-retention-rx-app-xxxxx.streamlit.app/`

---

## 🔥 **Option 2: Firebase Hosting (Static)**

### **Step 1: Install Firebase CLI**
```bash
npm install -g firebase-tools
```

### **Step 2: Initialize Firebase**
```bash
firebase login
firebase init hosting
```

### **Step 3: Configure Firebase**
```json
// firebase.json
{
  "hosting": {
    "public": "public",
    "ignore": [
      "firebase.json",
      "**/.*",
      "**/node_modules/**"
    ],
    "rewrites": [
      {
        "source": "**",
        "destination": "/index.html"
      }
    ]
  }
}
```

### **Step 4: Build Static Site**
```bash
# Install streamlit-to-static
pip install streamlit-to-static

# Convert to static HTML
streamlit-to-static app.py --output-dir public
```

### **Step 5: Deploy**
```bash
firebase deploy
```

---

## 🚀 **Option 3: Heroku (Full App)**

### **Step 1: Install Heroku CLI**
Download from: https://devcenter.heroku.com/articles/heroku-cli

### **Step 2: Create Heroku App**
```bash
heroku login
heroku create retention-rx-app
```

### **Step 3: Set Environment Variables**
```bash
heroku config:set STREAMLIT_TELEMETRY_ENABLED=false
heroku config:set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### **Step 4: Deploy**
```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### **Step 5: Open App**
```bash
heroku open
```

---

## ⚡ **Option 4: Railway (Modern Alternative)**

### **Step 1: Connect GitHub**
1. **Go to:** https://railway.app/
2. **Sign up** with GitHub
3. **Click:** "New Project"
4. **Select:** "Deploy from GitHub repo"

### **Step 2: Configure Railway**
```bash
# railway.json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true",
    "healthcheckPath": "/",
    "healthcheckTimeout": 100
  }
}
```

### **Step 3: Environment Variables**
```
STREAMLIT_TELEMETRY_ENABLED=false
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
PORT=8080
```

---

## 🎯 **Recommended: Streamlit Cloud (FREE)**

### **Why Streamlit Cloud?**
- ✅ **100% FREE** - No credit card required
- ✅ **Zero Configuration** - Just connect GitHub and deploy
- ✅ **Auto-Deploy** - Updates automatically when you push to GitHub
- ✅ **Custom Domain** - Add your own domain
- ✅ **Built for Streamlit** - Optimized for Streamlit apps

### **Quick Start (5 minutes):**
1. **Push to GitHub** (see Step 1 above)
2. **Go to:** https://share.streamlit.io/
3. **Deploy from GitHub**
4. **Share your live app!**

---

## 🔧 **GitHub Setup Commands**

```bash
# Navigate to your project directory
cd "C:\Retention RX"

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: RetentionRx Customer Analytics Platform"

# Create repository on GitHub (go to github.com and create new repo)

# Add remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/retention-rx.git

# Push to GitHub
git push -u origin main
```

---

## 📱 **After Deployment**

### **Share Your App:**
- **Streamlit Cloud:** `https://YOUR_USERNAME-retention-rx-app-xxxxx.streamlit.app/`
- **Heroku:** `https://retention-rx-app.herokuapp.com/`
- **Railway:** `https://retention-rx-app.railway.app/`

### **Update Your App:**
```bash
# Make changes to your code
git add .
git commit -m "Update app with new features"
git push origin main
# App automatically updates!
```

### **Environment Variables (if needed):**
- `OPENAI_API_KEY` - For AI playbook generation
- `STREAMLIT_TELEMETRY_ENABLED=false` - Disable telemetry
- `STREAMLIT_BROWSER_GATHER_USAGE_STATS=false` - Disable usage stats

---

## 🎉 **You're Ready to Deploy!**

**Recommended Path:**
1. **Push to GitHub** (5 minutes)
2. **Deploy on Streamlit Cloud** (2 minutes)
3. **Share your live app** (instant!)

**Your RetentionRx platform will be live and accessible to anyone with the URL!** 🚀
