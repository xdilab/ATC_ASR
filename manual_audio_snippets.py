import os
import random
import pandas as pd
import numpy as np
from pydub import AudioSegment
import noisereduce as nr

def build_audio_lookup_dict(audio_root):
    """
    Walk through 'audio_root' one time and build a dictionary {filename: full_path}.
    This lets us look up audio files quickly instead of repeatedly calling os.walk.
    """
    lookup = {}
    for root, dirs, files in os.walk(audio_root):
        for f in files:
            if f.lower().endswith(".wav"):
                lookup[f] = os.path.join(root, f)
    return lookup

def extract_times_from_datasets(directory):
    """
    Reads all CSV files in `directory` and returns a list of dictionaries.
    Each dict contains {'filename': ..., 'start_time': ..., 'end_time': ...}.
    """
    results = []
    for filename in os.listdir(directory):
        if not filename.lower().endswith('.csv'):
            continue

        filepath = os.path.join(directory, filename)
        try:
            data = pd.read_csv(filepath)
        except Exception as e:
            print(f"Could not read {filename}: {e}")
            continue

        for _, row in data.iterrows():
            audio_file = row.get('Filename')
            start_time = row.get('Start')  # 'hh:mm:ss'
            end_time   = row.get('End')    # 'hh:mm:ss'
            if audio_file and start_time and end_time:
                results.append({
                    'filename': audio_file.strip(),
                    'start_time': str(start_time).strip(),
                    'end_time': str(end_time).strip()
                })
    return results

def convert_to_seconds(time_str):
    """
    Converts a 'hh:mm:ss' string to total seconds.
    Raises ValueError if the format is not correct.
    """
    parts = time_str.strip().split(':')
    if len(parts) != 3:
        raise ValueError(f"Time string '{time_str}' does not match 'hh:mm:ss' format.")
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s

def find_audio_file(audio_lookup, filename):
    """
    Quickly returns the full path for 'filename' by looking it up in 'audio_lookup'.
    If not present, returns None.
    """
    return audio_lookup.get(filename)

def denoise_segment(audio_segment, prop_decrease=0.5):
    """
    Perform noise reduction on a pydub AudioSegment using noisereduce.
    'prop_decrease' indicates how much the noise should be reduced (0 to 1).
    
    Higher prop_decrease = more reduction (but also more potential artifacts).
    """
    # Convert AudioSegment to numpy array
    samples = np.array(audio_segment.get_array_of_samples())

    # If stereo, samples interleave left/right channels. 
    # For noisereduce, we often process each channel separately.
    channels = audio_segment.channels
    if channels == 1:
        # Mono
        reduced_samples = nr.reduce_noise(
            y=samples.astype(float),
            sr=audio_segment.frame_rate,
            prop_decrease=prop_decrease
        )
    else:
        # Stereo (or more channels)
        channel_length = len(samples) // channels
        split_channels = [
            samples[i * channel_length : (i + 1) * channel_length].astype(float)
            for i in range(channels)
        ]
        reduced_channels = [
            nr.reduce_noise(y=ch_data, sr=audio_segment.frame_rate, prop_decrease=prop_decrease)
            for ch_data in split_channels
        ]
        # Interleave the channels back
        reduced_samples = np.zeros(len(samples), dtype=np.float32)
        for i in range(channels):
            reduced_samples[i::channels] = reduced_channels[i]

    # Convert reduced samples back to int16
    reduced_samples_int16 = reduced_samples.astype(np.int16, copy=False)
    denoised_segment = AudioSegment(
        reduced_samples_int16.tobytes(),
        frame_rate=audio_segment.frame_rate,
        sample_width=2,  # int16
        channels=channels
    )
    return denoised_segment

def cleanup_audio_segment(audio_segment, 
                          apply_high_pass=True, 
                          apply_low_pass=True, 
                          apply_normalization=True, 
                          high_pass_cutoff=300, 
                          low_pass_cutoff=8000):
    """
    Optionally apply:
      - High-pass filter (remove rumble below 'high_pass_cutoff' Hz)
      - Low-pass filter (remove high freq noise above 'low_pass_cutoff' Hz)
      - Normalization (bring average level close to 0 dB)

    Adjust cutoff frequencies to taste.
    """
    # High-pass filter
    if apply_high_pass:
        audio_segment = audio_segment.high_pass_filter(high_pass_cutoff)

    # Low-pass filter
    if apply_low_pass:
        audio_segment = audio_segment.low_pass_filter(low_pass_cutoff)

    # Normalization
    if apply_normalization:
        # This applies gain such that the loudest peak is near 0dBFS
        # If you want a bit of headroom, you could do something like -audio_segment.dBFS - 1.0
        change_in_dBFS = -audio_segment.dBFS
        audio_segment = audio_segment.apply_gain(change_in_dBFS)

    return audio_segment

def create_snippet(audio_fullpath, start_sec, end_sec, output_path, audio_cache):
    """
    Extracts a snippet from `audio_fullpath` between `start_sec` and `end_sec` (in seconds),
    applies optional cleanup + denoises it, and saves to `output_path`.
    
    Uses 'audio_cache' to avoid reloading the same WAV file multiple times.
    """
    # If this file hasn't been loaded yet, load it once and store in cache
    if audio_fullpath not in audio_cache:
        audio_cache[audio_fullpath] = AudioSegment.from_file(audio_fullpath, format="wav")
    audio = audio_cache[audio_fullpath]
    
    # Pydub slices in milliseconds
    snippet = audio[start_sec * 1000 : end_sec * 1000]

    # ----- [New] Optional Pre-filtering / Cleanup -----
    snippet_cleaned = cleanup_audio_segment(
        snippet,
        apply_high_pass=True,
        apply_low_pass=True,
        apply_normalization=True,
        high_pass_cutoff=300,   # remove sub-300 Hz rumble
        low_pass_cutoff=8000    # remove frequencies above 8 kHz
    )

    # ----- Noise Reduction -----
    snippet_denoised = denoise_segment(snippet_cleaned, prop_decrease=0.5)

    # Export the denoised snippet
    snippet_denoised.export(output_path, format="wav")

def main():
    # 1. Path to the folder that contains subfolders of WAV files
    audio_path = r"C:\Users\tim3l\OneDrive\Desktop\Datasets"
    
    # 2. Path to the folder containing your CSV files with start/end times
    times_csv_folder = r"C:\Users\tim3l\OneDrive\Desktop\Sample Collector\timesets"

    # 4. Extract the list of {filename, start_time, end_time}
    times = extract_times_from_datasets(times_csv_folder)
    total_samples = len(times)
    if total_samples == 0:
        print("No valid snippet rows found.")
        return

    # Build a single file lookup dict so we only walk 'audio_path' once
    audio_lookup = build_audio_lookup_dict(audio_path)

    # Initialize the audio cache for storing loaded AudioSegments
    audio_cache = {}

    # Example random selection logic with a max time limit
    max_folders = 4
    used_indices = set()
    for folder_num in range(max_folders):
        # 3. Where to place the snippets
        snippets_folder = "Snippets\\ATIS" + str(folder_num + 1)
        os.makedirs(snippets_folder, exist_ok=True)
        
        max_time = 2400  # e.g., 40 minutes

        while max_time > 0 and len(used_indices) < total_samples:
            idx = random.randint(0, total_samples - 1)
            if idx in used_indices:
                continue
            used_indices.add(idx)

            sample = times[idx]
            # Convert times to seconds
            if sample['start_time'] == 'nan' or sample['end_time'] == 'nan':
                print(f"Skipping invalid time: {sample['start_time']} - {sample['end_time']}")
                continue
            start_sec = convert_to_seconds(sample['start_time'])
            end_sec   = convert_to_seconds(sample['end_time'])
            duration  = end_sec - start_sec
            if duration <= 0:
                print(f"Skipping invalid duration: start={start_sec}, end={end_sec}")
                continue

            max_time -= duration

            # Find the WAV file in our lookup dict
            audio_file = sample['filename']
            audio_fullpath = find_audio_file(audio_lookup, audio_file)
            if not audio_fullpath:
                print(f"Audio file not found in lookup: {audio_file}")
                continue

            # Construct output snippet name, e.g., "myfile-00_12_53-00_13_10.wav"
            snippet_filename = (
                f"{os.path.splitext(audio_file)[0]}-"
                f"{sample['start_time'].replace(':','_')}-"
                f"{sample['end_time'].replace(':','_')}.wav"
            )
            snippet_output_path = os.path.join(snippets_folder, snippet_filename)

            # Create and denoise the snippet
            create_snippet(audio_fullpath, start_sec, end_sec, snippet_output_path, audio_cache)

            print(f"Snippet created: {snippet_output_path} (duration {duration} seconds)")

if __name__ == "__main__":
    main()
