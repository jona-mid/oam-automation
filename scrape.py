import requests
import pandas as pd
import json
from tqdm import tqdm
import time

def fetch_openaerial_data(base_url="https://api.openaerialmap.org/meta", max_pages=None, retry_count=3, 
                          delay_between_requests=0.5):
    """Fetch all data from OpenAerialMap API with pagination"""
    
    # Get first page to determine total pages
    response = requests.get(base_url)
    if not response.ok:
        raise Exception(f"Failed to fetch initial data: {response.status_code}")
    
    data = response.json()
    
    total_records = data['meta']['found']
    limit_per_page = data['meta']['limit']
    total_pages = (total_records + limit_per_page - 1) // limit_per_page
    
    print(f"Found {total_records} total records across {total_pages} pages")
    
    if max_pages:
        total_pages = min(total_pages, max_pages)
        print(f"Limiting to {max_pages} pages")
    
    # Initialize with first page results
    all_results = data['results']
    
    # Fetch remaining pages
    for page in tqdm(range(2, total_pages + 1), desc="Fetching pages"):
        page_url = f"{base_url}?page={page}"
        
        # Add retry mechanism for robust fetching
        for attempt in range(retry_count):
            try:
                response = requests.get(page_url)
                if response.ok:
                    page_data = response.json()
                    all_results.extend(page_data['results'])
                    break
                else:
                    print(f"Page {page} request failed with status {response.status_code}")
                    if attempt < retry_count - 1:
                        time.sleep(1)  # Wait longer between retries
            except Exception as e:
                print(f"Error fetching page {page}, attempt {attempt+1}: {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(1)
        
        # Small delay to avoid overloading the API
        time.sleep(delay_between_requests)
        
    print(f"Successfully fetched {len(all_results)} records")
    return all_results

def process_data(results):
    """Process and flatten nested data structures"""
    processed_data = []
    
    for item in tqdm(results, desc="Processing records"):
        flat_item = item.copy()
        
        # Extract information from footprint
        if 'footprint' in flat_item and flat_item['footprint']:
            flat_item['footprint_coords'] = flat_item['footprint'].replace("POLYGON((", "").replace("))", "")
        
        # Extract bbox as separate columns
        if 'bbox' in flat_item and flat_item['bbox'] and len(flat_item['bbox']) == 4:
            flat_item['bbox_min_lon'] = flat_item['bbox'][0]
            flat_item['bbox_min_lat'] = flat_item['bbox'][1]
            flat_item['bbox_max_lon'] = flat_item['bbox'][2]
            flat_item['bbox_max_lat'] = flat_item['bbox'][3]
        
        # Flatten properties
        if 'properties' in flat_item and flat_item['properties']:
            for key, value in flat_item['properties'].items():
                flat_item[f'property_{key}'] = value
        
        # Remove nested structures to avoid duplication
        for key in ['geojson', 'properties']:
            if key in flat_item:
                del flat_item[key]
        
        processed_data.append(flat_item)
    
    return processed_data

def main():
    # Set max_pages to None to fetch all pages
    results = fetch_openaerial_data(max_pages=None)  # This will fetch ALL pages
    
    processed_data = process_data(results)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(processed_data)
    
    # Print dataset info
    print("\nDataset Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print("\nPlatform distribution:")
    if 'platform' in df.columns:
        print(df['platform'].value_counts())
    
    # Save to CSV
    df.to_csv("openaerial_data.csv", index=False)
    print(f"\nSaved dataset to openaerial_data.csv")

if __name__ == "__main__":
    main()
