#!/usr/bin/env python3
"""
Fix Prediction Files Utility
============================

This script helps fix corrupted prediction JSON files by:
1. Validating JSON format
2. Backing up corrupted files
3. Cleaning up malformed data
4. Recreating valid prediction files
"""

import os
import json
import glob
from datetime import datetime
import shutil

def validate_json_file(filepath: str) -> bool:
    """Validate if a JSON file is properly formatted"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return False
            json.loads(content)
            return True
    except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
        return False

def backup_file(filepath: str) -> str:
    """Create a backup of the file"""
    backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(filepath, backup_path)
        print(f"✅ Backed up: {filepath} -> {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Failed to backup {filepath}: {str(e)}")
        return None

def fix_prediction_file(filepath: str) -> bool:
    """Attempt to fix a corrupted prediction file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print(f"⚠️  Empty file: {filepath}")
            return False
        
        # Try to parse the JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in {filepath}: {str(e)}")
            
            # Try to fix common JSON issues
            fixed_content = fix_common_json_issues(content)
            if fixed_content:
                try:
                    data = json.loads(fixed_content)
                    print(f"🔧 Fixed JSON issues in {filepath}")
                except json.JSONDecodeError:
                    print(f"❌ Could not fix JSON in {filepath}")
                    return False
            else:
                return False
        
        # Validate the data structure
        if not isinstance(data, list):
            print(f"❌ Invalid data structure in {filepath}: expected list, got {type(data)}")
            return False
        
        # Validate each prediction entry
        valid_predictions = []
        for i, pred in enumerate(data):
            if validate_prediction_entry(pred):
                valid_predictions.append(pred)
            else:
                print(f"⚠️  Invalid prediction entry {i} in {filepath}")
        
        # Save the cleaned data
        if valid_predictions:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(valid_predictions, f, indent=2)
            print(f"✅ Fixed {filepath}: {len(valid_predictions)} valid predictions")
            return True
        else:
            print(f"❌ No valid predictions found in {filepath}")
            return False
            
    except Exception as e:
        print(f"❌ Error processing {filepath}: {str(e)}")
        return False

def fix_common_json_issues(content: str) -> str:
    """Fix common JSON formatting issues"""
    # Remove trailing commas
    content = content.replace(',]', ']').replace(',}', '}')
    
    # Fix missing quotes around keys
    import re
    content = re.sub(r'(\w+):', r'"\1":', content)
    
    # Fix single quotes to double quotes
    content = content.replace("'", '"')
    
    return content

def validate_prediction_entry(pred: dict) -> bool:
    """Validate a single prediction entry"""
    required_fields = ['timestamp', 'symbol', 'predictions', 'current_price']
    
    # Check required fields
    for field in required_fields:
        if field not in pred:
            return False
    
    # Check timestamp format
    try:
        datetime.strptime(pred['timestamp'], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    
    # Check predictions structure
    if not isinstance(pred['predictions'], dict):
        return False
    
    # Check current_price is numeric
    if not isinstance(pred['current_price'], (int, float)) and pred['current_price'] is not None:
        return False
    
    return True

def scan_and_fix_predictions_directory():
    """Scan the predictions directory and fix all corrupted files"""
    predictions_dir = "predictions"
    
    if not os.path.exists(predictions_dir):
        print(f"❌ Predictions directory '{predictions_dir}' not found")
        return
    
    print(f"🔍 Scanning predictions directory: {predictions_dir}")
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(predictions_dir, "*.json"))
    
    if not json_files:
        print("ℹ️  No JSON files found in predictions directory")
        return
    
    print(f"📁 Found {len(json_files)} JSON files")
    
    corrupted_files = []
    fixed_files = []
    skipped_files = []
    
    for filepath in json_files:
        filename = os.path.basename(filepath)
        print(f"\n📄 Processing: {filename}")
        
        if validate_json_file(filepath):
            print(f"✅ Valid file: {filename}")
            skipped_files.append(filename)
        else:
            print(f"❌ Corrupted file: {filename}")
            corrupted_files.append(filepath)
            
            # Create backup
            backup_path = backup_file(filepath)
            
            # Try to fix the file
            if fix_prediction_file(filepath):
                fixed_files.append(filename)
            else:
                print(f"❌ Could not fix {filename}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Total files: {len(json_files)}")
    print(f"   Valid files: {len(skipped_files)}")
    print(f"   Corrupted files: {len(corrupted_files)}")
    print(f"   Fixed files: {len(fixed_files)}")
    
    if fixed_files:
        print(f"\n✅ Successfully fixed files:")
        for filename in fixed_files:
            print(f"   - {filename}")
    
    if corrupted_files and len(fixed_files) < len(corrupted_files):
        print(f"\n⚠️  Some files could not be fixed. Check the backup files for manual review.")

def main():
    """Main function"""
    print("🔧 Prediction Files Fix Utility")
    print("=" * 40)
    
    # Scan and fix prediction files
    scan_and_fix_predictions_directory()
    
    print(f"\n🎉 Fix process completed!")
    print(f"\n💡 If you still have issues, you can:")
    print(f"   1. Delete the predictions directory to start fresh")
    print(f"   2. Run the prediction system again to generate new data")
    print(f"   3. Check the backup files for manual data recovery")

if __name__ == "__main__":
    main()
