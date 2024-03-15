
from pathlib import Path
import csv
from unicodedata import category
import argparse

ID = "id"
CONTEXT = "review"
PRED_ID = "id-#aspect"
PRED_LABEL = "sentiment"
NUM_OPTIONS = 4

MAIN_CATEGORIES = {
    "Location": (1, 3),
    "Service": (4, 4),
    "Price": (8, 3),
    "Ambience": (11, 4),
    "Food": (15, 4),
}

CATEGORY2INT = {
    "Location#Transportation": 1,
    "Location#Downtown": 2,
    "Location#Easy_to_find": 3,
    "Service#Queue": 4,
    "Service#Hospitality": 5,
    "Service#Parking": 6,
    "Service#Timely": 7,
    "Price#Level": 8,
    "Price#Cost_effective": 9,
    "Price#Discount": 10,
    "Ambience#Decoration": 11,
    "Ambience#Noise": 12,
    "Ambience#Space": 13,
    "Ambience#Sanitary": 14,
    "Food#Portion": 15,
    "Food#Taste": 16,
    "Food#Appearance": 17,
    "Food#Recommend": 18,
}

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    return parser.parse_args()

def fill_zero(data_dir):

    file_src: Path = data_dir / "test.csv"
    fp_src = file_src.open()
    reader = csv.DictReader(fp_src, delimiter=',', quotechar='\"')
        
    # Filename setup
    dir_dest: Path = data_dir
    dir_dest.mkdir(exist_ok=True)
    file_dest: Path = dir_dest / "test_fillzero.csv"
    fp_dest = file_dest.open("w")

    subcategories = list(CATEGORY2INT.keys())
    fieldnames = [ID, CONTEXT] + subcategories
    writer = csv.DictWriter(fp_dest, fieldnames, delimiter=',', quotechar='\"')
    writer.writeheader()

    # Read rows from source and write to destination
    for row in reader:
        row.update({col:"0" for _ in range(len(subcategories)) for col in subcategories})
        writer.writerow(row)

    # Close files
    fp_dest.close()
    fp_src.close()

if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir)

    fill_zero(data_dir) 