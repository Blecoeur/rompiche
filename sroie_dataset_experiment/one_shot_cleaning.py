#!/usr/bin/env python3
"""
One-shot script to clean the SROIE dataset from train.jsonl to train_ultimate_cleaned.jsonl.
This script handles all the data standardization in a single pass.
"""

import json
import re
from datetime import datetime
from collections import defaultdict

def clean_date(date_str):
    """Convert various date formats to ISO format (YYYY-MM-DD)"""
    if not date_str or not date_str.strip():
        return None
    
    date_str = date_str.strip().upper()
    
    # Remove parentheses
    if date_str.startswith('(') and date_str.endswith(')'):
        date_str = date_str[1:-1]
    
    # Try different date formats
    date_formats = [
        ('%d/%m/%Y', r'\d{2}/\d{2}/\d{4}'),  # DD/MM/YYYY
        ('%d-%m-%Y', r'\d{2}-\d{2}-\d{4}'),  # DD-MM-YYYY
        ('%d-%m-%y', r'\d{2}-\d{2}-\d{2}'),  # DD-MM-YY
        ('%d %b %Y', r'\d{2}\s[A-Z]{3}\s\d{4}'),  # DD MON YYYY
        ('%d/%m/%Y', r'\d{1,2}/\d{1,2}/\d{4}'),  # D/M/YYYY
        ('%Y-%m-%d', r'\d{4}-\d{2}-\d{2}'),  # YYYY-MM-DD
        ('%d-%b-%Y', r'\d{2}-\d{3}-\d{4}'),  # DD-MON-YYYY
        ('%Y%m%d', r'\d{8}'),  # YYYYMMDD
        ('%d/%m/%y', r'\d{2}/\d{2}/\d{2}'),  # DD/MM/YY
        ('%m/%d/%Y', r'\d{2}/\d{2}/\d{4}'),  # MM/DD/YYYY (US format)
    ]
    
    for date_format, pattern in date_formats:
        if re.match(pattern, date_str):
            try:
                if '%y' in date_format:
                    if '-' in date_str:
                        day, month, year = re.findall(r'(\d{2})-(\d{2})-(\d{2})', date_str)[0]
                    else:
                        day, month, year = re.findall(r'(\d{2})/(\d{2})/(\d{2})', date_str)[0]
                    year = f"20{year}" if int(year) < 50 else f"19{year}"
                    date_str = f"{day}/{month}/{year}"
                    date_format = '%d/%m/%Y'
                
                dt = datetime.strptime(date_str, date_format)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
    
    # Manual parsing for edge cases
    try:
        if '-' in date_str and len(date_str.split('-')) == 3:
            parts = date_str.split('-')
            if len(parts[1]) == 3:
                day, month, year = parts[0], parts[1], parts[2]
                month_num = datetime.strptime(month, '%b').month
                dt = datetime(int(year), month_num, int(day))
                return dt.strftime('%Y-%m-%d')
    except:
        pass
    
    try:
        if '/' in date_str and len(date_str.split('/')) == 3:
            parts = date_str.split('/')
            if len(parts[1]) == 3:
                day, month, year = parts[0], parts[1], parts[2]
                month_num = datetime.strptime(month, '%b').month
                dt = datetime(int(year), month_num, int(day))
                return dt.strftime('%Y-%m-%d')
    except:
        pass
    
    try:
        if '/' in date_str and len(date_str.split('/')) == 3:
            parts = date_str.split('/')
            if len(parts[0]) == 4 and len(parts[1]) == 2 and len(parts[2]) == 2:
                year, month, day = parts[0], parts[1], parts[2]
                dt = datetime(int(year), int(month), int(day))
                return dt.strftime('%Y-%m-%d')
    except:
        pass
    
    try:
        if ' ' in date_str:
            parts = date_str.split(' ')
            if len(parts) == 3 and len(parts[1]) == 3 and len(parts[2]) == 2:
                day, month, year = parts[0], parts[1], f"20{parts[2]}" if int(parts[2]) < 50 else f"19{parts[2]}"
                month_num = datetime.strptime(month, '%b').month
                dt = datetime(int(year), month_num, int(day))
                return dt.strftime('%Y-%m-%d')
    except:
        pass
    
    try:
        if ',' in date_str:
            parts = date_str.split(',')
            if len(parts) == 2:
                date_part = parts[0].strip()
                year = parts[1].strip()
                date_parts = date_part.split(' ')
                if len(date_parts) == 2:
                    month, day = date_parts[0], date_parts[1]
                    month_num = datetime.strptime(month, '%b').month
                    dt = datetime(int(year), month_num, int(day))
                    return dt.strftime('%Y-%m-%d')
    except:
        pass
    
    try:
        if len(date_str) == 8 and date_str.isdigit():
            year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
            dt = datetime(int(year), int(month), int(day))
            return dt.strftime('%Y-%m-%d')
    except:
        pass
    
    return date_str

def clean_total(total_str):
    """Convert various total formats to numeric float"""
    if not total_str or not total_str.strip():
        return None
    
    total_str = total_str.strip()
    total_str = total_str.replace('$', '').replace('RM', '').replace(',', '').strip()
    
    try:
        return float(total_str)
    except ValueError:
        return None

def clean_text_full_caps(text):
    """Normalize free text fields to full uppercase."""
    if text is None:
        return None
    return str(text).upper()

def clean_dataset(input_file, output_file):
    """Clean the dataset and save to output file"""
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    stats = defaultdict(int)
    changes_made = []
    
    for item in data:
        if 'results' in item:
            results = item['results']
            original_results = dict(results)  # Keep copy for change tracking
            
            # Clean date
            if 'date' in results:
                original_date = results['date']
                cleaned_date = clean_date(original_date)
                if cleaned_date and cleaned_date != original_date:
                    results['date'] = cleaned_date
                    stats['dates_cleaned'] += 1
                    changes_made.append({
                        'id': item.get('id', 'unknown'),
                        'field': 'date',
                        'original': original_date,
                        'cleaned': cleaned_date
                    })
                elif cleaned_date is None:
                    stats['dates_removed'] += 1
                    del results['date']
                    changes_made.append({
                        'id': item.get('id', 'unknown'),
                        'field': 'date',
                        'original': original_date,
                        'cleaned': None
                    })

            # Normalize company to full caps
            if 'company' in results:
                original_company = results['company']
                cleaned_company = clean_text_full_caps(original_company)
                if cleaned_company != original_company:
                    results['company'] = cleaned_company
                    stats['companies_cleaned'] += 1
                    changes_made.append({
                        'id': item.get('id', 'unknown'),
                        'field': 'company',
                        'original': original_company,
                        'cleaned': cleaned_company
                    })

            # Normalize address to full caps
            if 'address' in results:
                original_address = results['address']
                cleaned_address = clean_text_full_caps(original_address)
                if cleaned_address != original_address:
                    results['address'] = cleaned_address
                    stats['addresses_cleaned'] += 1
                    changes_made.append({
                        'id': item.get('id', 'unknown'),
                        'field': 'address',
                        'original': original_address,
                        'cleaned': cleaned_address
                    })
            
            # Clean total
            if 'total' in results:
                original_total = results['total']
                cleaned_total = clean_total(original_total)
                if cleaned_total is not None:
                    results['total'] = cleaned_total
                    stats['totals_cleaned'] += 1
                    changes_made.append({
                        'id': item.get('id', 'unknown'),
                        'field': 'total',
                        'original': original_total,
                        'cleaned': cleaned_total
                    })
                else:
                    stats['totals_removed'] += 1
                    del results['total']
                    changes_made.append({
                        'id': item.get('id', 'unknown'),
                        'field': 'total',
                        'original': original_total,
                        'cleaned': None
                    })
    
    # Save cleaned data
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    # Save change log
    with open('data_cleaning_changes.json', 'w') as f:
        json.dump(changes_made, f, indent=2)
    
    return stats, changes_made

if __name__ == '__main__':
    input_file = 'train.jsonl'
    output_file = 'train_cleaned.jsonl'
    
    print(f"🚀 Starting one-shot cleaning from {input_file}...")
    stats, changes = clean_dataset(input_file, output_file)
    
    print(f"\n✅ Cleaning complete! Saved to {output_file}")
    print(f"\n📊 Statistics:")
    print(f"  Dates cleaned: {stats['dates_cleaned']}")
    print(f"  Dates removed: {stats['dates_removed']}")
    print(f"  Companies cleaned: {stats['companies_cleaned']}")
    print(f"  Addresses cleaned: {stats['addresses_cleaned']}")
    print(f"  Totals cleaned: {stats['totals_cleaned']}")
    print(f"  Totals removed: {stats['totals_removed']}")
    
    print(f"\n📝 Change log saved to data_cleaning_changes.json")
    print(f"   Total changes made: {len(changes)}")
    
    # Show some examples of changes
    print(f"\n📋 Sample changes made:")
    for i, change in enumerate(changes[:10]):
        print(f"   {i+1}. ID: {change['id']}, Field: {change['field']}")
        print(f"      Original: {change['original']}")
        print(f"      Cleaned: {change['cleaned']}")
    
    if len(changes) > 10:
        print(f"   ... and {len(changes) - 10} more changes")
    
    print(f"\n🎉 Dataset successfully standardized!")
