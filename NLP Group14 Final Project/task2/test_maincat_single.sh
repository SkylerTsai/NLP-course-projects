
printf "Main Category Name: ${CATEGORY_NAME:=Location}\n"
printf "Model Name: ${MODEL_NAME:=bert-base-chinese}\n"
printf "Model Type: ${MODEL_TYPE:=bert}\n"
printf "Output Dir: ${OUTPUT_DIR:=./output/maincat/${MODEL_TYPE}/${CATEGORY_NAME}}\n"
printf "Logging Dir: ${LOGGING_DIR:=./logs/maincat/${MODEL_TYPE}/${CATEGORY_NAME}}\n"

python src/run_maincat_no_trainer.py \
  --model_name_or_path "${MODEL_NAME}" \
  --train_file "data/${CATEGORY_NAME}/train.csv" \
  --validation_file "data/${CATEGORY_NAME}/dev.csv" \
  --test_file "data/${CATEGORY_NAME}/test.csv" \
  --max_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --with_tracking \
  --checkpointing_steps epoch \
  --resume_from_checkpoint "${OUTPUT_DIR}/epoch_2" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}"