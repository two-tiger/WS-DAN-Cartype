from datasets import get_trainval_datasets
import config
from torch.utils.data import DataLoader


def split_list_with_batchsize(image_list, batchsize):
    reshape_list = []
    for i in range(0, len(image_list), batchsize):
        reshape_list.append(image_list[i:i+batchsize])
    return reshape_list

train_data, test_data = get_trainval_datasets(config.tag, config.image_size)
all_image_list = test_data.get_all_image_path()
res = split_list_with_batchsize(all_image_list, config.batch_size)
print(res)
test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

# for i, (X, y) in enumerate(test_loader):
#     print(len(X))
#     #print(y)
#     print(len(res[i]))