import os
import kaggle

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
yes_dir = os.path.join(data_dir, 'yes')
no_dir = os.path.join(data_dir, 'no')
data_url = 'https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection'
dataset = 'navoneel/brain-mri-images-for-brain-tumor-detection'

def fetch_dataset(dataset_name=dataset, output_dir=data_dir):
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            dataset=dataset_name,
            path=output_dir,
            unzip=True
        )
    except Exception as e:
        print(f"Error: {e}")
        raise

def main():
    if os.path.exists(yes_dir) and os.path.exists(no_dir):
        print("Dataset already available. Skipping download.")
    elif os.path.exists(data_dir) and not (os.path.exists(yes_dir) and os.path.exists(no_dir)):
        print("Dataset download incomplete. Re-downloading...")
        fetch_dataset()
    else:
        os.makedirs(data_dir, exist_ok=True)
        fetch_dataset()
        print("Dataset successfully downloaded.")

if __name__ == "__main__":
    main()
