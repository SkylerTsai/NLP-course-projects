
from pathlib import Path
import csv

ID = "id"
CONTEXT = "review"
NUM_OPTIONS = 4

# Split data by category
def split_data_by_category(data_dir: Path):
    
    def str2list(s: str):
        return s.strip().split(",")

    category2int = {
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

    category2chinese = {
        "Location#Transportation": "地點#交通",
        "Location#Downtown": "地點#市區",
        "Location#Easy_to_find": "地點#容易尋找",
        "Service#Queue": "服務#排隊",
        "Service#Hospitality": "服務#接待",
        "Service#Parking": "服務#停車",
        "Service#Timely": "服務#及時",
        "Price#Level": "價格#水準",
        "Price#Cost_effective": "這家餐廳的價格划算嗎？",
        "Price#Discount": "這家餐廳的價格有折扣嗎？",
        "Ambience#Decoration": "這家餐廳的環境漂亮嗎？",
        "Ambience#Noise": "這家餐廳的環境安靜嗎？",
        "Ambience#Space": "這家餐廳的環境空間大嗎？",
        "Ambience#Sanitary": "這家餐廳的環境衛生嗎？",
        "Food#Portion": "這家餐廳的食物份量如何？",
        "Food#Taste": "這家餐廳的食物味道如何？",
        "Food#Appearance": "這家餐廳的食物外觀如何？",
        "Food#Recommend": "這家餐廳的食物推薦嗎？",
    }

    category2choices = {
        "Location#Transportation": ("沒提到", "不好", "普通", "好"),
        "Location#Downtown": ("沒提到", "不好", "普通", "好"),
        "Location#Easy_to_find": ("沒提到", "不好", "普通", "好"),
        "Service#Queue": ("沒提到", "不好", "普通", "好"),
        "Service#Hospitality": ("沒提到", "不好", "普通", "好"),
        "Service#Parking": ("沒提到", "不好", "普通", "好"),
        "Service#Timely": ("沒提到", "不好", "普通", "好"),
        "Price#Level": ("沒提到", "不好", "普通", "好"),
        "Price#Cost_effective": ("沒提到", "不好", "普通", "好"),
        "Price#Discount": ("沒提到", "不好", "普通", "好"),
        "Ambience#Decoration": ("沒提到", "不好", "普通", "好"),
        "Ambience#Noise": ("沒提到", "不好", "普通", "好"),
        "Ambience#Space": ("沒提到", "不好", "普通", "好"),
        "Ambience#Sanitary": ("沒提到", "不好", "普通", "好"),
        "Food#Portion": ("沒提到", "不好", "普通", "好"),
        "Food#Taste": ("沒提到", "不好", "普通", "好"),
        "Food#Appearance": ("沒提到", "不好", "普通", "好"),
        "Food#Recommend": ("沒提到", "不好", "普通", "好"),
    }

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
        data["categories"] = category2int.keys()
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

        selected_cols = (
            ID, 
            CONTEXT, 
            "category",
            "label",
            "option0",
            "option1",
            "option2",
            "option3",
        )
        
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
                        [f"{i}-{category2int[category]}" for i in data[ID]],
                        data[CONTEXT],
                        [category2chinese[category] for _ in range(data["n_rows"])],
                        [str(int(i)+2) for i in data[category]],
                        ["沒提到" for _ in range(data["n_rows"])],
                        ["不好" for _ in range(data["n_rows"])],
                        ["普通" for _ in range(data["n_rows"])],
                        ["好" for _ in range(data["n_rows"])],
                    )
                    for row in zip(*selected_data):
                        # fp.write(",".join(row) + "\n")
                        if row[3] == "0":
                            continue
                        writer.writerow(row)
                        full_writer.writerow(row)
    
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
    data["categories"] = category2int.keys()
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
                    [f"{i}-{category2int[category]}" for i in data[ID]],
                    data[CONTEXT],
                    [category2chinese[category] for _ in range(data["n_rows"])],
                    ["0" for _ in range(data["n_rows"])],
                    ["沒提到" for _ in range(data["n_rows"])],
                    ["不好" for _ in range(data["n_rows"])],
                    ["普通" for _ in range(data["n_rows"])],
                    ["好" for _ in range(data["n_rows"])],
                )
                for row in zip(*selected_data):
                    # fp.write(",".join(row) + "\n")
                    writer.writerow(row)
                    full_writer.writerow(row)


if __name__ == "__main__":
    # data_dir = Path(input("Enter data location: "))
    data_dir = Path("data")
    split_data_by_category(data_dir)