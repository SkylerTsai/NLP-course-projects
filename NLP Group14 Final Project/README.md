# NLP Final Project Group 14

## Task 1
```shell
# Preprocessing
cd ./task1/data
python preprocess.py
cd ..
# Training and Testing
python train.py \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--num_train_epochs 2 \
--train_file ./data/_train.csv \
--validation_file ./data/_dev.csv \
--test_file ./data/_test.csv \
--output_dir ${path:-./output/}

```

## Task 2

Please enter the folder `task2` and run the `preprocess.sh` first.
- Preprocessing
```bash
./preprocess.sh [train_file] [dev_file] [test_file]
```
> `train_file` is the path to the training data.  
> `dev_file` is the path to the validation data.  
> `test_file` is the path to the testing data.  

### Method 1: One Model For all Categories
- Training
```bash
./train_allcat.sh [model_name] [model_type]
```
> *(Optional)* `model_name` is the name of the pretrained model (e.g. bert-base-chinese)  
> *(Optional)* `model_type` is the type of the pretrained model (e.g. bert)

- Testing
```bash
./test_allcat.sh [model_name] [model_type]
```
> *(Optional)* `model_name` is the name of the pretrained model (e.g. bert-base-chinese)  
> *(Optional)* `model_type` is the type of the pretrained model (e.g. bert)

### Method 2: Train models for each group of categories (5 models)
- Training
```bash
./train_groups.sh [model_name] [model_type]
```
> *(Optional)* `model_name` is the name of the pretrained model (e.g. bert-base-chinese)  
> *(Optional)* `model_type` is the type of the pretrained model (e.g. bert)

- Testing
```bash
./test_groups.sh [model_name] [model_type]
```
> *(Optional)* `model_name` is the name of the pretrained model (e.g. bert-base-chinese)  
> *(Optional)* `model_type` is the type of the pretrained model (e.g. bert)

### Method 3: Train models for each category (18 models)
- Training
```bash
./train_subcat.sh [model_name] [model_type]
```
> *(Optional)* `model_name` is the name of the pretrained model (e.g. bert-base-chinese)  
> *(Optional)* `model_type` is the type of the pretrained model (e.g. bert)

- Testing
```bash
./test_subcat.sh [model_name] [model_type]
```
> *(Optional)* `model_name` is the name of the pretrained model (e.g. bert-base-chinese)  
> *(Optional)* `model_type` is the type of the pretrained model (e.g. bert)