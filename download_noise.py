import requests
from bs4 import BeautifulSoup
from pathlib import Path

BUCKET_URL = "https://whispyr-noise-files.s3.amazonaws.com/"
LOCAL_NOISE_DIR = Path(__file__).parent / "noise"


def list_s3_files(bucket_url):
    files = []

    def fetch_keys(prefix=""):
        url = f"{bucket_url}?prefix={prefix}&delimiter=/"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "xml")

        # Fetch files
        for content in soup.find_all("Contents"):
            key = content.find("Key").text
            if not key.endswith("/"):
                files.append(key)

        # Fetch subdirectories recursively
        for common_prefix in soup.find_all("CommonPrefixes"):
            sub_prefix = common_prefix.find("Prefix").text
            fetch_keys(sub_prefix)

    fetch_keys()
    return files


def download_files(file_keys, bucket_url, local_dir):
    for key in file_keys:
        local_file = local_dir / key
        local_file.parent.mkdir(parents=True, exist_ok=True)

        url = bucket_url + key
        if local_file.exists():
            print(f"{key} already exists, skipping.")
            continue

        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(local_file, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Saved {key}")
        else:
            print(f"Failed to download {key}: {response.status_code}")


if __name__ == "__main__":
    print("Listing all files in the bucket...")
    files = list_s3_files(BUCKET_URL)
    print(f"Found {len(files)} files.")

    download_files(files, BUCKET_URL, LOCAL_NOISE_DIR)

    print("Download completed.")
