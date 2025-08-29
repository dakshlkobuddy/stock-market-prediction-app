#!/usr/bin/env python3
"""
Deployment Helper Script for Stock Market Prediction App
=======================================================

This script helps prepare and deploy the stock market prediction app.
It checks for common issues and provides guidance for deployment.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_requirements():
    """Check if all required files exist"""
    required_files = [
        'streamlit_app.py',
        'requirements.txt',
        'real_time_stock_predictor.py',
        'config.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def check_git_status():
    """Check if Git is initialized and repository is ready"""
    try:
        # Check if Git is initialized
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Git repository not initialized")
            return False
        
        # Check if there are uncommitted changes
        result = subprocess.run(['git', 'diff', '--name-only'], capture_output=True, text=True)
        if result.stdout.strip():
            print("⚠️  You have uncommitted changes. Consider committing them before deployment.")
            return False
        
        print("✅ Git repository is ready")
        return True
        
    except FileNotFoundError:
        print("❌ Git is not installed. Please install Git first.")
        return False

def check_dependencies():
    """Check if all dependencies are properly specified"""
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        # Check for essential dependencies
        essential_deps = ['streamlit', 'pandas', 'numpy', 'yfinance']
        missing_deps = []
        
        for dep in essential_deps:
            if not any(dep in req for req in requirements):
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"❌ Missing essential dependencies: {', '.join(missing_deps)}")
            return False
        
        print("✅ Dependencies look good")
        return True
        
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def create_deployment_checklist():
    """Create a deployment checklist"""
    checklist = """
📋 DEPLOYMENT CHECKLIST
======================

Before deploying, make sure you have:

1. ✅ GitHub Account
   - Create account at github.com if you don't have one

2. ✅ Streamlit Cloud Account
   - Sign up at share.streamlit.io
   - Connect your GitHub account

3. ✅ Code Repository
   - Initialize Git: git init
   - Add files: git add .
   - Commit: git commit -m "Initial commit"
   - Create GitHub repository
   - Push: git push origin main

4. ✅ Deploy on Streamlit Cloud
   - Go to share.streamlit.io
   - Click "New app"
   - Select your repository
   - Set main file: streamlit_app.py
   - Click "Deploy!"

5. ✅ Test Your App
   - Visit your deployed URL
   - Test all features
   - Check on different devices

6. ✅ Share Your App
   - Share the URL with others
   - Add to your portfolio
   - Monitor usage and performance

🚀 READY TO DEPLOY!
"""
    return checklist

def main():
    """Main deployment helper function"""
    print("🚀 Stock Market Prediction App - Deployment Helper")
    print("=" * 50)
    
    # Check prerequisites
    print("\n🔍 Checking prerequisites...")
    
    if not check_requirements():
        print("\n❌ Please fix the missing files before deployment.")
        return
    
    if not check_dependencies():
        print("\n❌ Please fix the dependency issues before deployment.")
        return
    
    if not check_git_status():
        print("\n❌ Please initialize Git and commit your changes.")
        return
    
    print("\n✅ All checks passed!")
    
    # Show deployment checklist
    print(create_deployment_checklist())
    
    # Provide next steps
    print("\n🎯 NEXT STEPS:")
    print("1. Run: git init")
    print("2. Run: git add .")
    print("3. Run: git commit -m 'Initial commit for deployment'")
    print("4. Create a GitHub repository")
    print("5. Run: git remote add origin YOUR_REPO_URL")
    print("6. Run: git push -u origin main")
    print("7. Deploy on Streamlit Cloud")
    
    print("\n📚 For detailed instructions, see: deployment_guide.md")
    print("🌐 Your app will be available at: https://your-app-name.streamlit.app")

if __name__ == "__main__":
    main()
