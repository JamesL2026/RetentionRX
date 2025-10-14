#!/usr/bin/env python3
"""
RetentionRx Installation Script
Quick setup for the churn prediction and playbook generation tool
"""

import subprocess
import sys
import os

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

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", f"{version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor} is compatible")
    return True

def install_dependencies():
    """Install required packages"""
    return run_command(
        "pip install -r requirements.txt",
        "Installing Python dependencies"
    )

def verify_files():
    """Verify all required files exist"""
    required_files = [
        "app.py",
        "sample_data.csv",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def main():
    """Main installation process"""
    print("🔮 RetentionRx Installation Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Verify files
    if not verify_files():
        print("❌ Please ensure you're in the RetentionRx directory")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Installation failed. Please check error messages above.")
        sys.exit(1)
    
    print("\n🎉 Installation completed successfully!")
    print("\n🚀 To start the application:")
    print("   streamlit run app.py")
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main()
