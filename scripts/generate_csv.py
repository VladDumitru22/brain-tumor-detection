import os
import glob
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
yes_dir = os.path.join(data_dir, 'yes', '*.jpg')
no_dir = os.path.join(data_dir, 'no', '*.jpg')
output_csv = os.path.join(data_dir, 'annotations.csv')

file_paths = []
labels = []

for file in glob.iglob(no_dir):
    file_paths.append(file)
    labels.append(0)

for file in glob.iglob(yes_dir):
    file_paths.append(file)
    labels.append(1)


df = pd.DataFrame({'filename': file_paths, 'label': labels})

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv(output_csv, index=False)

print(f"annotations.csv created with {len(df)} samples at {output_csv}")
