
export MODEL_NAME=${1:-"bert-base-chinese"}
export MODEL_TYPE=${2:-"bert"}

for category in "Food" "Ambience" "Location" "Price" "Service"
do
  export CATEGORY_NAME="$category_name"
  sh ./test_maincat_single.sh
  printf " ----------------------------------------------------\n"
done

python src/split_absa.py -m