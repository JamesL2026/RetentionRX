#!/usr/bin/env python3
"""
GitHub Deployment Helper for RetentionRx
This script helps you push your code to GitHub for automatic deployment
"""

import subprocess
import os
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main deployment process"""
    print("🚀 RetentionRx GitHub Deployment Helper")
    print("=" * 50)
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("📁 Initializing Git repository...")
        if not run_command("git init", "Initialize Git repository"):
            sys.exit(1)
    
    # Add all files
    if not run_command("git add .", "Add all files to Git"):
        sys.exit(1)
    
    # Check if there are changes to commit
    result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if not result.stdout.strip():
        print("ℹ️ No changes to commit. Repository is up to date.")
        return
    
    # Commit changes
    commit_message = input("📝 Enter commit message (or press Enter for default): ").strip()
    if not commit_message:
        commit_message = "Update RetentionRx - Enhanced analytics and deployment ready"
    
    if not run_command(f'git commit -m "{commit_message}"', "Commit changes"):
        sys.exit(1)
    
    # Check if remote exists
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    if not result.stdout.strip():
        print("\n🌐 GitHub Repository Setup Required:")
        print("1. Go to https://github.com/new")
        print("2. Create a new repository named 'retention-rx'")
        print("3. Copy the repository URL")
        print("4. Run this command:")
        print("   git remote add origin https://github.com/YOUR_USERNAME/retention-rx.git")
        print("   git branch -M main")
        print("   git push -u origin main")
        print("\nThen go to https://share.streamlit.io to deploy!")
        return
    
    # Push to GitHub
    print("\n🚀 Pushing to GitHub...")
    if not run_command("git push origin main", "Push to GitHub"):
        print("⚠️ Push failed. You may need to set up the remote repository first.")
        print("Run these commands manually:")
        print("git remote add origin https://github.com/YOUR_USERNAME/retention-rx.git")
        print("git branch -M main")
        print("git push -u origin main")
        return
    
    print("\n🎉 Successfully pushed to GitHub!")
    print("\n📋 Next Steps:")
    print("1. Go to https://share.streamlit.io")
    print("2. Click 'New app'")
    print("3. Connect your GitHub repository")
    print("4. Select branch: main")
    print("5. Main file path: app.py")
    print("6. Click 'Deploy!'")
    print("\nYour app will be live at: https://retention-rx.streamlit.app")

if __name__ == "__main__":
    main()
