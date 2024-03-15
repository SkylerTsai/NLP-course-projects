
export MODEL_NAME=${1:-"bert-base-chinese"}
export MODEL_TYPE=${2:-"bert"}

for category_name in  "Location#Transportation" "Location#Downtown" "Location#Easy_to_find" "Service#Queue" "Service#Hospitality" "Service#Parking" "Service#Timely" "Price#Level" "Price#Cost_effective" "Price#Discount" "Ambience#Decoration" "Ambience#Noise" "Ambience#Space" "Ambience#Sanitary" "Food#Portion" "Food#Taste" "Food#Appearance" "Food#Recommend"
do
  export CATEGORY_NAME="$category_name"
  sh ./train_subcat_single.sh
  printf " ----------------------------------------------------\n"
done

python src/split_subcat.py merge -O "output/subcat/${MODEL_TYPE}/"