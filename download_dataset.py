# download_dataset.py (Ubuntu/Linux optimized)
import os
import zipfile
import opendatasets as od

DATASET_URL = "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
TARGET_DIR = "datasets/chest_xray"
KAGGLE_PATH = os.path.expanduser("./kaggle.json")

def check_kaggle_key():
    if not os.path.exists(KAGGLE_PATH):
        raise FileNotFoundError(
            f"❌ Kaggle API key not found!\n"
            f"Please place kaggle.json here:\n  {KAGGLE_PATH}\n"
        )
    else:
        os.system(f"chmod 600 {KAGGLE_PATH}")
        print(f"🔑 Kaggle API located & permissions set → {KAGGLE_PATH}")

def download_dataset():
    print("📥 Downloading dataset from Kaggle...\n")
    od.download(DATASET_URL)  # 다운로드 시작

    # 다운로드 후 zip 파일 자동 탐색 → unzip
    for f in os.listdir():
        if f.endswith(".zip"):
            print(f"📦 Extracting {f} ...")
            with zipfile.ZipFile(f, "r") as zip_ref:
                zip_ref.extractall(".")
            print("✔ Unzip complete\n")

    # 폴더명 정리 (압축 없이 바로 내려오는 경우 포함)
    if os.path.exists("chest-xray-pneumonia"):
        os.makedirs("datasets", exist_ok=True)
        os.rename("chest-xray-pneumonia", TARGET_DIR)
        print(f"📂 Dataset moved → {TARGET_DIR}")

    print("\n🔥 Dataset setup complete!")
    print(f"📍 Ready for training → {TARGET_DIR}\n")


if __name__ == "__main__":
    check_kaggle_key()
    download_dataset()
