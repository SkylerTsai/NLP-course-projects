
export MODEL_NAME=${1:-"bert-base-chinese"}
export MODEL_TYPE=${2:-"bert"}

for category in "Food" "Ambience" "Location" "Price" "Service"
do
  export CATEGORY_NAME="$category"
  sh ./train_maincat_single.sh
  printf " ----------------------------------------------------\n"
done

python src/split_maincat.py merge --output "output/maincat/${MODEL_TYPE}"