import os
import csv

def create_csv_for_snippets(snippets_dir):
    # Ensure the directory exists
    if not os.path.isdir(snippets_dir):
        print(f"The directory {snippets_dir} does not exist.")
        return

    # Iterate through each subfolder in Snippets
    for folder_name in os.listdir(snippets_dir):
        folder_path = os.path.join(snippets_dir, folder_name)

        # Only proceed if it's a directory
        if os.path.isdir(folder_path):
            csv_rows = []  # Will hold rows of data for this folder

            # Collect all .wav files in the folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(".wav"):
                    # Example of filename: KGSO1-App-Dep-Jun-01-2024-0200Z-0_13_06-0_13_12.wav

                    # Remove extension
                    base_name, _ = os.path.splitext(filename)

                    # Split on "Z-". The part after Z- is: 0_13_06-0_13_12
                    parts = base_name.split("Z-")
                    if len(parts) < 2:
                        # If the naming pattern is unexpected, skip or handle differently
                        continue

                    time_part = parts[-1]  # "0_13_06-0_13_12"

                    # Split that part on '-': [ "0_13_06", "0_13_12" ]
                    try:
                        start_str, end_str = time_part.split('-')
                    except ValueError:
                        # If there's not exactly one '-', the format isn't as expected
                        continue

                    # Convert underscores to colons, e.g. "0_13_06" -> "0:13:06"
                    start_str = start_str.replace('_', ':')
                    end_str   = end_str.replace('_', ':')

                    # Append row: [Filename, Start, End, Transcription]
                    # Transcription is left blank.
                    csv_rows.append([str(parts[0])+"Z.wav", start_str, end_str, ""])

            # Write the CSV for this folder
            csv_file_path = os.path.join(folder_path, f"{folder_name}.csv")
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(["Filename", "Start", "End", "Transcription"])
                # Write data rows
                writer.writerows(csv_rows)

            print(f"CSV created for folder '{folder_name}' at: {csv_file_path}")

if __name__ == "__main__":
    # Change this path to your actual Snippets directory
    snippets_directory = r"C:\Users\tim3l\OneDrive\Desktop\Sample Collector\Snippets"
    create_csv_for_snippets(snippets_directory)
