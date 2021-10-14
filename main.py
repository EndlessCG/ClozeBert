from transformers import BertConfig, AlbertConfig

from ClozeBert_utils import *
from ClozeBert import ClozeBertModelForTransformers


# plot_train_len(read_data_json("train")[0])

# train_data, dev_data, test_data = read_data_json_for_whole_passage('train'), \
#                                   read_data_json_for_whole_passage('dev'), \
#                                   read_data_json_for_whole_passage('test')
# train_loaders, dev_loaders, test_loaders = pack_loaders(train_data, dev_data, test_data)

train_loaders, dev_loaders, test_loaders = get_saved_loaders(loader_path)

config = AlbertConfig.from_pretrained(bert_path)
# summary_model(MODEL_FILE_NAME)
# model = ClozeBertModel.from_pretrained('bert-base-uncased').cuda()
model = ClozeBertModelForTransformers(config).to(DEVICE)


plt.figure(figsize=(8, 8), dpi=80)
plt.figure(1)
LR = 9e-5
loss_ls, acc_ls = train(model, train_loaders, dev_loaders, LR)
print(loss_ls)
print(acc_ls)
spl1 = plt.subplot(121)
spl1.plot(loss_ls, color='b')
spl2 = plt.subplot(122)
spl2.plot(acc_ls, color='g')
plt.show()
