#!/usr/bin/env python3
"""
Simple validation script for Neural Machine Translation project
This script validates the project structure and basic functionality without heavy dependencies
"""

import os
import json
import sys

def check_project_structure():
    """Check if all required files exist"""
    print("🔍 Checking project structure...")
    
    required_files = [
        "0102.py",
        "requirements.txt", 
        "README.md",
        "setup.py",
        "test_nmt.py",
        "templates/index.html",
        ".gitignore"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_python_syntax():
    """Check Python syntax of main files"""
    print("\n🔍 Checking Python syntax...")
    
    python_files = ["0102.py", "setup.py", "test_nmt.py"]
    
    for file in python_files:
        try:
            with open(file, 'r') as f:
                content = f.read()
            compile(content, file, 'exec')
            print(f"✅ {file} - Syntax OK")
        except SyntaxError as e:
            print(f"❌ {file} - Syntax Error: {e}")
            return False
        except Exception as e:
            print(f"⚠️  {file} - Warning: {e}")
    
    return True

def check_requirements():
    """Check if requirements.txt is properly formatted"""
    print("\n🔍 Checking requirements.txt...")
    
    try:
        with open("requirements.txt", 'r') as f:
            lines = f.readlines()
        
        valid_packages = 0
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                if '>=' in line or '==' in line or line.isalpha():
                    valid_packages += 1
                    print(f"✅ Package: {line}")
                else:
                    print(f"⚠️  Package: {line} - Check format")
        
        print(f"✅ Found {valid_packages} valid packages")
        return valid_packages > 0
        
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def check_html_template():
    """Check if HTML template is valid"""
    print("\n🔍 Checking HTML template...")
    
    try:
        with open("templates/index.html", 'r') as f:
            content = f.read()
        
        # Basic HTML validation
        if "<!DOCTYPE html>" in content:
            print("✅ DOCTYPE declaration found")
        else:
            print("⚠️  No DOCTYPE declaration")
        
        if "<html" in content and "</html>" in content:
            print("✅ HTML structure looks good")
        else:
            print("❌ Invalid HTML structure")
            return False
        
        if "Neural Machine Translation" in content:
            print("✅ Title found")
        else:
            print("⚠️  Title not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading HTML template: {e}")
        return False

def check_readme():
    """Check if README has essential sections"""
    print("\n🔍 Checking README.md...")
    
    try:
        with open("README.md", 'r') as f:
            content = f.read()
        
        essential_sections = [
            "# Neural Machine Translation",
            "## Features",
            "## Installation", 
            "## Usage",
            "## Project Structure"
        ]
        
        found_sections = 0
        for section in essential_sections:
            if section in content:
                print(f"✅ Found: {section}")
                found_sections += 1
            else:
                print(f"⚠️  Missing: {section}")
        
        return found_sections >= 3
        
    except Exception as e:
        print(f"❌ Error reading README.md: {e}")
        return False

def main():
    """Run all validation checks"""
    print("🚀 Neural Machine Translation - Project Validation")
    print("=" * 60)
    
    checks = [
        ("Project Structure", check_project_structure),
        ("Python Syntax", check_python_syntax),
        ("Requirements", check_requirements),
        ("HTML Template", check_html_template),
        ("README Documentation", check_readme)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}")
        print("-" * 40)
        try:
            if check_func():
                passed += 1
                print(f"✅ {check_name} - PASSED")
            else:
                print(f"❌ {check_name} - FAILED")
        except Exception as e:
            print(f"❌ {check_name} - ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Validation Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All validations passed! Project structure is correct.")
        print("\n📝 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training: python3 0102.py")
        print("3. Start web interface: python3 0102.py web")
    else:
        print("⚠️  Some validations failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
