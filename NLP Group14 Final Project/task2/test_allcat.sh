
export MODEL_NAME=${1:-"bert-base-chinese"}
export MODEL_TYPE=${2:-"bert"}

printf "All categories\n"
printf "Model Name: ${MODEL_NAME}\n"
printf "Model Type: ${MODEL_TYPE}\n"
printf "Output Dir: ${OUTPUT_DIR:=./output/allcat/${MODEL_TYPE}}\n"
printf "Logging Dir: ${LOGGING_DIR:=./logs/allcat/${MODEL_TYPE}}\n"

python src/run_maincat_no_trainer.py \
  --model_name_or_path "${MODEL_NAME}" \
  --train_file "data/train.csv" \
  --validation_file "data/dev.csv" \
  --test_file "data/test_fillzero.csv" \
  --max_length 512 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --with_tracking \
  --checkpointing_steps epoch \
  --resume_from_checkpoint "${OUTPUT_DIR}/epoch_2" \
  --output_dir "${OUTPUT_DIR}" \
  --logging_dir "${LOGGING_DIR}" \