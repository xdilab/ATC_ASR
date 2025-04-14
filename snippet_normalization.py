import os
import re
from datetime import datetime

snippets_dir = r"C:\Users\tim3l\OneDrive\Desktop\Local_Wav2Vec\Snippets"
pattern = re.compile(r'(.+Z)-(\d+[_\-]\d+[_\-]\d+)-(\d+[_\-]\d+[_\-]\d+)\.wav$')

renamed = 0
skipped = 0

def normalize_time_part(time_str):
    try:
        # Convert to datetime using flexible separators
        parts = [int(p) for p in re.split(r'[_\-]', time_str)]
        if len(parts) == 3:
            h, m, s = parts
            return f"{h:02}_{m:02}_{s:02}"
    except Exception as e:
        pass
    return None

print("🔄 Renaming snippet files to padded HH_MM_SS format...\n")

for fname in os.listdir(snippets_dir):
    if not fname.lower().endswith(".wav"):
        continue

    match = pattern.match(fname)
    if not match:
        print(f"⚠️ Skipping (no match): {fname}")
        skipped += 1
        continue

    base, start, end = match.groups()
    new_start = normalize_time_part(start)
    new_end = normalize_time_part(end)

    if new_start and new_end:
        new_name = f"{base}-{new_start}-{new_end}.wav"
        src = os.path.join(snippets_dir, fname)
        dst = os.path.join(snippets_dir, new_name)

        if src != dst:
            os.rename(src, dst)
            print(f"✅ Renamed: {fname} → {new_name}")
            renamed += 1
    else:
        print(f"⚠️ Skipping (bad time format): {fname}")
        skipped += 1

print(f"\n🎉 Done! Renamed {renamed} files. Skipped {skipped}.")
