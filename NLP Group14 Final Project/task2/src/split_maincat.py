
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

CATEGORY2CHINESE = {
    "Location#Transportation": "地點#交通",
    "Location#Downtown": "地點#市區",
    "Location#Easy_to_find": "地點#容易尋找",
    "Service#Queue": "服務#排隊",
    "Service#Hospitality": "服務#接待",
    "Service#Parking": "服務#停車",
    "Service#Timely": "服務#及時",
    "Price#Level": "價格#水準",
    "Price#Cost_effective": "價格#划算",
    "Price#Discount": "價格#折扣",
    "Ambience#Decoration": "環境#裝飾",
    "Ambience#Noise": "環境#噪音",
    "Ambience#Space": "環境#空間",
    "Ambience#Sanitary": "環境#衛生",
    "Food#Portion": "食物#份量",
    "Food#Taste": "食物#味道",
    "Food#Appearance": "食物#外觀",
    "Food#Recommend": "食物#推薦",
}

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("split", "merge"), default="split")
    # parser.add_argument("--merge", "-m", action="store_true")
    parser.add_argument("--output_dir", "-O", type=str)
    parser.add_argument("--input_file", "-I", type=str)

    return parser.parse_args()

def merge_result(output_dir: Path):

    print("Start processing the prediction results")

    # Open file
    file_src: Path = output_dir / "test_predictions.csv"
    fp_dest = file_src.open("w")
    fieldnames = [PRED_ID, PRED_LABEL]
    writer = csv.DictWriter(fp_dest, fieldnames, delimiter=',', quotechar='\"')
    writer.writeheader()

    all_rows = list()

    for main_category, (category_start, category_size) in MAIN_CATEGORIES.items():
        subcategories = list(CATEGORY2INT.keys())[category_start-1:category_start+category_size-1]
        print(main_category)
        
        # Filename setup
        dir_src = output_dir / main_category
        file_src: Path = dir_src / "test_predictions.csv"
        print(str(file_src))
        fp_src = file_src.open()
        
        reader = csv.DictReader(fp_src, delimiter=',', quotechar='\"')

        # Read rows from source and write to destination
        i = 0
        for row in reader:
            i += 1
            row_id, row_aspect = row[PRED_ID].split("-")
            row[PRED_ID] = "{}-{}".format(row_id, category_start + int(row_aspect) - 1)
            all_rows.append(row)
        print(i)
        # Close source file
        fp_src.close()

    # print(all_rows[0])
    # all_rows = sorted(all_rows, key=lambda x: x[PRED_ID].split("-")[-1])
    # print(all_rows)

    writer.writerows(all_rows)

    # Close source file
    fp_dest.close()
    pass

# Fill zeros in the test dataset
def fill_zeros(data_dir: Path):

    # Testing set
    split_name = "test"
    print("Start processing the {} split".format(split_name))

    # Open file
    file_src: Path = data_dir / "{}.csv".format(split_name)
    fp_src = file_src.open()
    reader = csv.DictReader(fp_src, delimiter=',', quotechar='\"')

    for main_category, (category_start, category_size) in MAIN_CATEGORIES.items():
        subcategories = list(CATEGORY2INT.keys())[category_start-1:category_start+category_size-1]
        fieldnames = [ID, CONTEXT] + subcategories
        
        # Filename setup
        dir_dest = data_dir / main_category
        dir_dest.mkdir(exist_ok=True)
        file_dest: Path = dir_dest / "{}.csv".format(split_name)
        fp_dest = file_dest.open("w")
        
        writer = csv.DictWriter(fp_dest, fieldnames, delimiter=',', quotechar='\"')
        writer.writeheader()

        # Read rows from source and write to destination
        for row in reader:
            row.update({col:"0" for _ in range(category_size) for col in subcategories})
            writer.writerow(row)

        # Close destination file
        fp_dest.close()

        # Move pointer to the second line
        fp_src.seek(0)
        next(fp_src)

    # Close source file
    fp_src.close()

# split training and validation sets by category
def split_dataset(data_dir: Path, split_name: str):

    ## TODO: Split datasets by main categories

    # Testing set
    print("Start processing the {} split".format(split_name))

    # Open file
    file_src: Path = data_dir / "{}.csv".format(split_name)
    fp_src = file_src.open()
    reader = csv.DictReader(fp_src, delimiter=',', quotechar='\"')

    for main_category, (category_start, category_size) in MAIN_CATEGORIES.items():
        print(f"Main category {main_category}")

        subcategories = list(CATEGORY2INT.keys())[category_start-1:category_start+category_size-1]
        fieldnames = [ID, CONTEXT] + subcategories
        
        # Filename setup
        dir_dest = data_dir / main_category
        dir_dest.mkdir(exist_ok=True)
        file_dest: Path = dir_dest / "{}.csv".format(split_name)
        fp_dest = file_dest.open("w")
        
        writer = csv.DictWriter(fp_dest, fieldnames, delimiter=',', quotechar='\"')
        writer.writeheader()

        # Read rows from source and write to destination
        i = 0
        count = 0
        for row in reader:
            row = {col:row[col] for _ in fieldnames for col in fieldnames}

            unique_values = set(list(row.values())[2:])
            # print(list(row.values())[2:])
            count += list(row.values()).count("1") + list(row.values()).count("0") + list(row.values()).count("-1")
            
            if len(unique_values) == 1 and len(unique_values.intersection(["-2"])) :
                # print(unique_values)
                continue
            i += 1

            writer.writerow(row)
            
        print(f"{i} rows. {count} values not -2")

        # Close destination file
        fp_dest.close()

        # Move pointer to the second line
        fp_src.seek(0)
        next(fp_src)

    # Close source file
    fp_src.close()

if __name__ == "__main__":
    args = parse_args()
    if args.action == "split":
        data_dir = Path("data")
        split_dataset(data_dir, "train")
        split_dataset(data_dir, "dev")
        fill_zeros(data_dir)
    else:
        output_dir = Path(args.output_dir)
        merge_result(output_dir)