
printf "Subcategory Name: ${CATEGORY_NAME:=Location#Transportation}\n"
printf "Model Name: ${MODEL_NAME:=bert-base-chinese}\n"
printf "Model Type: ${MODEL_TYPE:=bert}\n"
printf "Output Dir: ${OUTPUT_DIR:=./output/subcat/${MODEL_TYPE}/${CATEGORY_NAME}/}\n"
printf "Logging Dir: ${LOGGING_DIR:=./logs/subcat/${MODEL_TYPE}/${CATEGORY_NAME}/}\n"

python src/run_subcat_no_trainer.py \
  --model_name_or_path "$MODEL_NAME" \
  --train_file "data/$CATEGORY_NAME/train.csv" \
  --validation_file "data/$CATEGORY_NAME/dev.csv" \
  --test_file "data/$CATEGORY_NAME/test.csv" \
  --max_length 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --with_tracking \
  --checkpointing_steps epoch \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}" \
  # --debug \
#   --max_train_steps 2500 \
#   --debug \
#   --resume_from_checkpoint ./tmp/classification/ \
#   --model_name_or_path bert-base-chinese \
  # --train_file "data/Food#Appearance/train.csv" \
  # --validation_file "data/Food#Appearance/dev.csv" \