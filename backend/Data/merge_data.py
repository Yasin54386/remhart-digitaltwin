import pandas as pd
import glob
import os

def process_electrical_logs():
    # 1. Define paths based on your VS Code screenshot
    # Your folder is named 'Data' (Capital D)
    input_pattern = "Data/Dataset 3 Phase/*.xlsx" 
    
    # This goes one level up from 'backend' to the root 'REMHART-DIGITALTWIN'
    output_file = "../master_data.xlsx" 

    # 2. Find all Excel files
    all_files = glob.glob(input_pattern)
    
    if not all_files:
        print(f"No files found in: {os.path.abspath(input_pattern)}")
        print("Check if your .xlsx files are actually inside the 'Data' folder.")
        return

    # 3. Read and Combine
    print(f"Found {len(all_files)} files. Merging...")
    df_list = [pd.read_excel(f) for f in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # 4. Clean and Sort
    # Convert Timestamp to datetime for sorting
    merged_df['Timestamp'] = pd.to_datetime(merged_df['Timestamp'])
    
    # Sort and remove duplicates
    merged_df = merged_df.sort_values(by='Timestamp')
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates()
    
    print(f"Removed {initial_count - len(merged_df)} duplicate rows.")

    # 5. Save to the external file
    merged_df.to_excel(output_file, index=False)
    print(f"Success! Master file created at: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    process_electrical_logs()