# ADL Homework 1

By Pei-Hsun, Tsai, B07902037
---
## Training

1. You can use command `bash preprocess.sh` before the following step, though the following scripts will do the preprocess if you didn't run this command.
2. Use command `bash train_intent.sh /path/to/data/dir /path/to/save/model/dir` to train the data for intent classification task.
3. Use command `bash train_slot.sh /path/to/data/dir /path/to/save/model/dir` to train the data for slot tagging task.

P.S. the model will be saved to the `/path/to/save/model/dir` with the name `eval_accuracy.pt`

---
## Testing

1. Use command `bash download.sh` to download  the **best_intent.pt** , **best_slot** and **Cache** files.
2. Use command `bash intent_cls.sh /path/to/test/data /path/to/prediction/data` to generate prediction form the test data and model **best_intent.pt** . 
3. Use command `bash slot_tag.sh /path/to/test/data /path/to/prediction/data` to generate prediction form the test data and model **best_slot.pt** . 

P.S. If you want to test the model you trained by above step, neglect first step and change the model name to best_intent.pt, best_slot.pt respect to the task. than run the command on the second / third step to get prediction

--- 

## Files

- report: The report
- utils.py: **Vocab** and **pad_to_len** functions
- dataset.py: **SeqClsDataset** and **TagClsDataset** datasets
- model.py: **SeqClassifier** and **TagClassifier** models
- Other files are mentioned above