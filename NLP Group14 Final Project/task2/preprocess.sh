export TRAIN_DATA=${1:-"data/train.csv"}
export DEV_DATA=${2:-"data/dev.csv"}
export TEST_DATA=${3:-"data/test.csv"}

mkdir "data"
cp "${TRAIN_DATA}" ./data/train.csv
cp "${DEV_DATA}" ./data/dev.csv
cp "${TEST_DATA}" ./data/test.csv

python src/allcat_fillzero.py
python src/split_maincat.py split
python src/split_subcat.py split