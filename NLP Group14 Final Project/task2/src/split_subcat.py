
from pathlib import Path
import csv
import argparse

ID = "id"
CONTEXT = "review"
PRED_ID = "id-#aspect"
PRED_LABEL = "sentiment"
NUM_OPTIONS = 4

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

    for category, category_start in CATEGORY2INT.items():
        print(category)
        category_idx = CATEGORY2INT[category]
        
        # Filename setup
        dir_src = output_dir / category
        file_src: Path = dir_src / "test_predictions.csv"
        print(str(file_src))
        fp_src = file_src.open()
        
        reader = csv.DictReader(fp_src, delimiter=',', quotechar='\"')

        # Read rows from source and write to destination
        i = 0
        for row in reader:
            i += 1
            all_rows.append(row)
        print(i)

        # Close source file
        fp_src.close()

    writer.writerows(all_rows)

    # Close source file
    fp_dest.close()
    pass

# Split data by category
def split_data_by_category(data_dir: Path):

    selected_cols = (
        ID, 
        CONTEXT,
        "category"
        "label",
    )

    # Training and Validation set
    for split_name in ["train", "dev"]:
        print("Start processing the {} split".format(split_name))

        # Temporarily save all data as csv
        data = dict()

        # Open file
        file: Path = data_dir / "{}.csv".format(split_name)
        fp = file.open()
        reader = csv.reader(fp, delimiter=',', quotechar='\"')

        # Read column title
        # column_titles = str2list(fp.readline())
        column_titles = next(reader)
        data["categories"] = CATEGORY2INT.keys()
        print("Column titles found:\n" + "\n".join(column_titles))
        for col in column_titles:
            data[col] = list() # Initialize

        # Read rest of the rows
        for line in reader:
            # print(line)
            columns = line

            for idx, col in enumerate(column_titles):
                data[col].append(columns[idx])

        # Close file
        fp.close()

        # Show number of rows
        data["n_rows"] = len(data[ID])
        print("{} rows found.".format(data["n_rows"]))

        
        full_file = data_dir / "full_{}.csv".format(split_name)
        with full_file.open("w") as fp:
            full_writer = csv.writer(fp, delimiter=',', quotechar='\"')
            full_writer.writerow(selected_cols)
            # Loop over each category
            for category in data["categories"]:

                ## Create folders for each directory
                category_dir: Path = data_dir / category
                try:
                    category_dir.mkdir()
                    print("Creating directory: {}".format(str(category_dir)))
                except FileExistsError:
                    print("Directory exists.")
                    
                ## TODO: Write context and labels to file
                category_file: Path = category_dir / "{}.csv".format(split_name)
                print("Now writing \"{}\" category data of the \"{}\" split to {}".format(category, split_name, str(category_file)))
                with category_file.open("w") as fp:
                    writer = csv.writer(fp, delimiter=',', quotechar='\"')
                    # fp.write(",".join(selected_cols) + "\n")
                    writer.writerow(selected_cols)

                    selected_data = (
                        [f"{i}-{CATEGORY2INT[category]}" for i in data[ID]],
                        data[CONTEXT],
                        [CATEGORY2CHINESE[category] for _ in range(data["n_rows"])],
                        data[category],
                    )

                    i = 0
                    for row in zip(*selected_data):
                        # fp.write(",".join(row) + "\n")
                        if row[3] == "-2":
                            continue
                        i += 1
                        writer.writerow(row)
                        full_writer.writerow(row)
                    print(i)
    
    # Testing set
    split_name = "test"
    print("Start processing the {} split".format(split_name))

    # Temporarily save all data as csv
    data = dict()

    # Open file
    file: Path = data_dir / "{}.csv".format(split_name)
    fp = file.open()
    reader = csv.reader(fp, delimiter=',', quotechar='\"')

    # Read column title
    # column_titles = str2list(fp.readline())
    column_titles = next(reader)
    print("Column titles found:\n" + "\n".join(column_titles))
    data["categories"] = CATEGORY2INT.keys()
    for col in column_titles:
        data[col] = list() # Initialize

    # Read rest of the rows
    for line in reader:
        # print(line)
        columns = line

        for idx, col in enumerate(column_titles):
            data[col].append(columns[idx])

    # Close file
    fp.close()

    # Show number of rows
    data["n_rows"] = len(data[ID])
    print("{} rows found.".format(data["n_rows"]))

    full_file = data_dir / "full_{}.csv".format(split_name)
    with full_file.open("w") as fp:
        full_writer = csv.writer(fp, delimiter=',', quotechar='\"')
        full_writer.writerow(selected_cols)
        # Loop over each category
        for category in data["categories"]:

            ## Create folders for each directory
            category_dir: Path = data_dir / category
            try:
                category_dir.mkdir()
                print("Creating directory: {}".format(str(category_dir)))
            except FileExistsError:
                print("Directory exists.")
                
            ## TODO: Write context and labels to file
            category_file: Path = category_dir / "{}.csv".format(split_name)
            print("Now writing \"{}\" category data of the \"{}\" split to {}".format(category, split_name, str(category_file)))
            with category_file.open("w") as fp:
                writer = csv.writer(fp, delimiter=',', quotechar='\"')
                # fp.write(",".join(selected_cols) + "\n")
                writer.writerow(selected_cols)

                selected_data = (
                    [f"{i}-{CATEGORY2INT[category]}" for i in data[ID]],
                    data[CONTEXT],
                    [CATEGORY2CHINESE[category] for _ in range(data["n_rows"])],
                    ["0" for _ in range(data["n_rows"])],
                )
                for row in zip(*selected_data):
                    # fp.write(",".join(row) + "\n")
                    writer.writerow(row)
                    full_writer.writerow(row)

if __name__ == "__main__":
    args = parse_args()
    if args.action == "split":
        data_dir = Path("data")
        split_data_by_category(data_dir)
    else:
        output_dir = Path(args.output_dir)
        merge_result(output_dir)