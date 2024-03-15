
import json
import csv
import logging
import math
import os
import random
import time
import datetime

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

class TrainerSubCat():

    def __init__(self, args, logger):
        self.start_time = time.time()
        self.args = args
        self.logger = logger

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
        self.accelerator = Accelerator(log_with="all", logging_dir=args.logging_dir) if args.with_tracking else Accelerator()
        self.device = self.accelerator.device # Use the device given by the `accelerator` object.ration for debugging.

        # Make one log on every process with the configuration
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        self.accelerator.wait_for_everyone()


    def prepare(self):
        self.load_datasets()
        self.load_model_and_tokenizer()
        self.preprocess_datasets()
        self.setup_dataloaders()
        self.setup_optimizer()
        self.setup_scheduler()
        self.prepare_with_accelerator()
        self.init_tracking()
        self.load_metric()

    def load_datasets(self):
        # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
        # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).

        # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
        # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
        # label if at least two columns are provided.

        # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
        # single column. You can easily tweak this behavior (see below)

        # Loading the dataset from local csv or json file.
        data_files = {}
        if self.args.train_file is not None:
            data_files["train"] = self.args.train_file
        if self.args.validation_file is not None:
            data_files["validation"] = self.args.validation_file
        if self.args.test_file is not None:
            data_files["test"] = self.args.test_file
        extension = (self.args.train_file if self.args.train_file is not None else self.args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
        # See more about loading any type of standard or custom dataset at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Trim a number of training examples
        if self.args.debug:
            for split in raw_datasets.keys():
                num_samples = min(200, len(raw_datasets[split]))
                raw_datasets[split] = raw_datasets[split].select(range(num_samples))

        # Get the ids of the data
        if self.args.train_file is not None:
            self.train_ids = raw_datasets["train"]["id"]
        if self.args.validation_file is not None:
            self.eval_ids = raw_datasets["validation"]["id"]
        if self.args.test_file is not None:
            self.test_ids = raw_datasets["test"]["id"]

        # Labels
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        self.raw_datasets = raw_datasets
        self.label_list = raw_datasets["train"].unique("label")
        self.label_list.sort()  # Let's sort it for determinism
        self.num_labels = len(self.label_list)

    def load_model_and_tokenizer(self):
        # Load pretrained model and tokenizer
        #
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        self.config = AutoConfig.from_pretrained(self.args.model_name_or_path, num_labels=self.num_labels, finetuning_task=self.args.task_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=not self.args.use_slow_tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
        # model = RobertaForSequenceClassification.from_pretrained(
            self.args.model_name_or_path,
            from_tf=bool(".ckpt" in self.args.model_name_or_path),
            config=self.config,
            ignore_mismatched_sizes=self.args.ignore_mismatched_sizes,
        )
        self.model.to(self.device)

    def preprocess_datasets(self):
        # Preprocessing the datasets
        
        non_label_column_names = [name for name in self.raw_datasets["train"].column_names if name != "label"]
        sentence1_key, sentence2_key = "review", "category"

        label_to_id = {v: i for i, v in enumerate(self.label_list)}

        if label_to_id is not None:
            self.model.config.label2id = label_to_id
            self.model.config.id2label = {id: label for label, id in self.config.label2id.items()}

        padding = "max_length" if self.args.pad_to_max_length else False

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = self.tokenizer(*texts, padding=padding, max_length=self.args.max_length, truncation=True)

            if "label" in examples:
                if label_to_id is not None:
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l] for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = examples["label"]
            return result

        with self.accelerator.main_process_first():
            processed_datasets = self.raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=self.raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )

        self.train_dataset = processed_datasets["train"]
        self.eval_dataset = processed_datasets["validation_matched" if self.args.task_name == "mnli" else "validation"]
        if self.args.test_file is not None:
            self.test_dataset = processed_datasets["test"]

        # Log a few random samples from the training set:
        LOG_SAMPLES = False
        if LOG_SAMPLES:
            for index in random.sample(range(len(self.train_dataset)), 3):
                self.logger.info(f"Sample {index} of the training set: {self.train_dataset[index]}.")
        del LOG_SAMPLES # Make sure this variable is not used elsewhere

    def setup_dataloaders(self):
        # DataLoaders creation:
        if self.args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=(8 if self.accelerator.use_fp16 else None))

        self.train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, collate_fn=data_collator, batch_size=self.args.per_device_train_batch_size
        )
        self.eval_dataloader = DataLoader(self.eval_dataset, collate_fn=data_collator, batch_size=self.args.per_device_eval_batch_size)
        if self.args.test_file is not None:
            self.test_dataloader = DataLoader(self.test_dataset, collate_fn=data_collator, batch_size=self.args.per_device_eval_batch_size)

    def setup_optimizer(self):
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)

    def setup_scheduler(self):
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        else:
            self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.args.max_train_steps,
        )

    def prepare_with_accelerator(self):
        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler
        )
        if self.args.test_file is not None:
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch

        # Figure out how many steps we should save the Accelerator states
        if hasattr(self.args.checkpointing_steps, "isdigit"):
            self.checkpointing_steps = self.args.checkpointing_steps
            if self.args.checkpointing_steps.isdigit():
                self.checkpointing_steps = int(self.args.checkpointing_steps)
        else:
            self.checkpointing_steps = None

    def init_tracking(self):
        # We need to initialize the trackers we use, and also store our configuration
        if self.args.with_tracking:
            experiment_config = vars(self.args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            self.accelerator.init_trackers("glue_no_trainer", experiment_config)

    # Get the metric function
    def load_metric(self):
        self.metric = load_metric("accuracy")

    # Training part in an epoch
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for step, batch in enumerate(self.train_dataloader):
            # We need to skip steps until we reach the resumed step
            if self.args.resume_from_checkpoint and epoch == self.starting_epoch:
                if self.resume_step is not None and step < self.resume_step:
                    self.completed_steps += 1
                    continue
                
            outputs = self.model(**batch)
            loss = outputs.loss
            # We keep track of the loss at each epoch
            if self.args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss)
            if step % self.args.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.progress_bar.update(1)
                self.completed_steps += 1

            if isinstance(self.checkpointing_steps, int):
                if self.completed_steps % self.checkpointing_steps == 0:
                    output_dir = f"step_{self.completed_steps }"
                    if self.args.output_dir is not None:
                        output_dir = os.path.join(self.args.output_dir, output_dir)
                    self.accelerator.save_state(output_dir)

            if self.completed_steps >= self.args.max_train_steps:
                break
        
        return total_loss # Returns 0 if args.with_tracking is False

    # Validation part in an epoch
    def eval_epoch(self, epoch):
        self.model.eval()
        samples_seen = 0

        for step, batch in enumerate(self.eval_dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = self.accelerator.gather((predictions, batch["labels"]))

            # If we are in a multiprocess environment, the last batch has duplicates
            if self.accelerator.num_processes > 1:
                if step == len(self.eval_dataloader) - 1:
                    predictions = predictions[: len(self.eval_dataloader.dataset) - samples_seen]
                    references = references[: len(self.eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            self.metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = self.metric.compute()
        return eval_metric

    def train(self):
        # Train!
        total_batch_size = self.args.per_device_train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")

        # Only show the progress bar once on each machine.
        self.progress_bar = tqdm(range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        self.completed_steps = 0
        self.starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint is not None or self.args.resume_from_checkpoint != "":
                self.accelerator.print(f"Resumed from checkpoint: {self.args.resume_from_checkpoint}")
                self.accelerator.load_state(self.args.resume_from_checkpoint)
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            self.training_difference = os.path.splitext(path)[0]

            if "epoch" in self.training_difference:
                self.starting_epoch = int(self.training_difference.replace("epoch_", "")) + 1
                self.resume_step = None
            else:
                self.resume_step = int(self.training_difference.replace("step_", ""))
                self.starting_epoch = self.resume_step // len(self.train_dataloader)
                self.resume_step -= self.starting_epoch * len(self.train_dataloader)

        # Traning loop
        for epoch in range(self.starting_epoch, self.args.num_train_epochs):
            
            # Train
            total_loss = self.train_epoch(epoch)

            # Eval
            eval_metric = self.eval_epoch(epoch)

            # Logging
            self.logger.info(f"epoch {epoch}: {eval_metric}")
            if self.args.with_tracking:
                self.accelerator.log(
                    {
                        "accuracy" if self.args.task_name is not None else "glue": eval_metric,
                        "train_loss": total_loss,
                        "epoch": epoch,
                        "step": self.completed_steps,
                    },
                )

            if self.args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if self.args.output_dir is not None:
                    output_dir = os.path.join(self.args.output_dir, output_dir)
                self.accelerator.save_state(output_dir)
        
        self.eval_metric = eval_metric

    def pred_step(self, dataset_split):
        if dataset_split == "train":
            dataloader = self.train_dataloader
            ids = self.train_ids
        elif dataset_split == "validation":
            dataloader = self.eval_dataloader
            ids = self.eval_ids
        elif dataset_split == "test":
            dataloader = self.test_dataloader
            ids = self.test_ids
        else:
            raise Exception("Dataset split not recognized: {}".format(dataset_split))
        id2label = self.model.config.id2label
        
        self.model.eval()

        samples_seen = 0
        print(self.device)
        full_predictions: torch.Tensor = torch.Tensor().to(self.device)
        for step, batch in enumerate(dataloader):
            predictions: torch.Tensor
            references: torch.Tensor

            with torch.no_grad():
                outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions , references = self.accelerator.gather((predictions, batch["labels"])) # torch.Tensor, torch.Tensor

            # If we are in a multiprocess environment, the last batch has duplicates
            if self.accelerator.num_processes > 1:
                if step == len(dataloader) - 1:
                    predictions = predictions[: len(dataloader.dataset) - samples_seen]
                    references = references[: len(dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            full_predictions = torch.cat((full_predictions, predictions))

        full_predictions = full_predictions.int().cpu().apply_(lambda x: id2label[x])
        final_predictions: list = list(zip(ids, full_predictions.detach().tolist()))
        return final_predictions

    def predict(self):
        LOG_SAMPLES = False

        # Prediction on training dataset
        # self.logger.info(f"Generating perdictions for training data...")
        # self.train_predictions: list = self.pred_step("train")
        # if LOG_SAMPLES:
        #     self.logger.info(self.train_predictions[:10])

        # # Prediction on validation dataset
        # self.logger.info(f"Generating perdictions for validation data...")
        # self.eval_predictions: list = self.pred_step("validation")
        # if LOG_SAMPLES:
        #     self.logger.info(self.eval_predictions[:10])

        # Prediction on test dataset
        self.logger.info(f"Generating perdictions for testing data...")
        self.test_predictions: list = self.pred_step("test")
        if LOG_SAMPLES:
            self.logger.info(self.test_predictions[:10])

    def save_model(self):
        if self.args.output_dir is not None:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                self.args.output_dir, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
            )
            if self.accelerator.is_main_process:
                self.tokenizer.save_pretrained(self.args.output_dir)

    def save_results(self):
        if self.args.output_dir is not None:
            try:
                with open(os.path.join(self.args.output_dir, "all_results.json"), "w") as f:
                    dump_result = {
                        "elapsed_time": str(datetime.timedelta(seconds=time.time() - self.start_time)),
                        "eval_accuracy": self.eval_metric["accuracy"], 
                    }
                    json.dump(dump_result, f)
            except UnboundLocalError:
                self.logger.info(f"No evaluation has been executed during training. It might be that you have skipped all training epochs.")

            # if self.args.train_file is not None:
            #     with open(os.path.join(self.args.output_dir, "train_predictions.csv"), "w") as f:
            #         writer = csv.writer(f)
            #         writer.writerow(("id-#aspect", "sentiment"))
            #         writer.writerows(self.train_predictions)

            # if self.args.validation_file is not None:
            #     with open(os.path.join(self.args.output_dir, "eval_predictions.csv"), "w") as f:
            #         writer = csv.writer(f)
            #         writer.writerow(("id-#aspect", "sentiment"))
            #         writer.writerows(self.eval_predictions)

            if self.args.test_file is not None:
                with open(os.path.join(self.args.output_dir, "test_predictions.csv"), "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(("id-#aspect", "sentiment"))
                    writer.writerows(self.test_predictions)