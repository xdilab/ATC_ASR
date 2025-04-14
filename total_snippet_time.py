import csv
from datetime import datetime, timedelta

# Update this path to your actual CSV file location
csv_file_path = r"C:\Users\tim3l\OneDrive\Desktop\Local_Wav2Vec\wav2vec_predictions\wav2vec2-large-960h-lv60-self-en-atc-uwb-atcc - KGSO_ATIS.csv"

total_duration = timedelta()

with open(csv_file_path, mode='r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Adjust these keys if your CSV header is different
        start_str = row.get('Start')
        end_str   = row.get('End')
        
        if not start_str or not end_str:
            # Skip rows that don't have proper Start/End
            continue
        
        # Convert hh:mm:ss to a datetime object
        start_time = datetime.strptime(start_str, '%H:%M:%S')
        end_time   = datetime.strptime(end_str,   '%H:%M:%S')
        
        # Compute the difference (timedelta) and accumulate
        duration = end_time - start_time
        total_duration += duration

# Convert total_duration (timedelta) to an hh:mm:ss string
hours, remainder = divmod(total_duration.total_seconds(), 3600)
minutes, seconds = divmod(remainder, 60)
formatted_total_duration = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

print("Total time:", formatted_total_duration)
