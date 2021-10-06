from transformers import AlbertConfig

from ClozeBert import ClozeBertModelForTransformers
from ClozeBert_utils import *

bert_path = 'albert-large-v2'
data_path = 'ELE\\'
dataset_path = 'datasets'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = AlbertConfig.from_pretrained(bert_path)
MODEL = ClozeBertModelForTransformers(config).to(DEVICE)
train_dataset, eval_dataset = get_dataset(dataset_path, 'Dataset', True)
MODEL_FILE_NAME = 'cloze_bert_transformers'


class ClozeAlbertTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        output, option_opts, loss, opt_acc = model(**inputs)
        return (option_opts, output) if return_outputs else loss


TRAIN_ARGS = transformers.TrainingArguments(
    output_dir=MODEL_FILE_NAME,
    do_train=True,
    do_eval=True,
    fp16=True,
    evaluation_strategy='steps',
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    eval_steps=500,
    dataloader_pin_memory=False,
    label_names='labels'
)
TRAINER = ClozeAlbertTrainer(model=MODEL, train_dataset=train_dataset, eval_dataset=eval_dataset,
                             args=TRAIN_ARGS)
TRAINER.train()
TRAINER.save_model(MODEL_FILE_NAME + '_final')
