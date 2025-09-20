#!/usr/bin/env python3
"""
Setup script for Neural Machine Translation project
This script helps set up the project environment and dependencies
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    directories = ["models", "data", "logs", "templates"]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def run_tests():
    """Run the test suite"""
    print("🧪 Running tests...")
    try:
        result = subprocess.run([sys.executable, "test_nmt.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print(f"❌ Tests failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Neural Machine Translation Setup")
    print("=" * 50)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Installing dependencies", install_dependencies),
        ("Running tests", run_tests)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            return False
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Run 'python 0102.py' to train the model")
    print("2. Run 'python 0102.py web' to start the web interface")
    print("3. Open http://localhost:5000 in your browser")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
