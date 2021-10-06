import math

import torch
import ClozeBert_utils


class ClozeIterDataset(torch.utils.data.IterableDataset):
    def __init__(self, dir_name):
        self.input_ids, self.input_types, \
        self.input_masks, self.options, \
        self.answers, self.all_masked_indices, \
        self.all_complete_ids = ClozeBert_utils.read_data_json(dir_name)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        dataset_len = len(self.input_ids)
        if worker_info is None:
            iter_start = 0
            iter_end = dataset_len
        else:
            per_worker = int(math.ceil(dataset_len / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, dataset_len)
        return {'input_ids': iter(self.input_ids),
                'token_type_ids': iter(self.input_types),
                'masks': iter(self.input_masks),
                'mask_ids': iter(self.all_masked_indices),
                'labels': iter(self.all_complete_ids)}


class ClozeDataset(torch.utils.data.Dataset):
    def __init__(self, dir_name):
        # super.__init__(pin_memory=False)
        self.ids, self.types, \
        self.masks, self.options, \
        self.answers, self.mask_ids, \
        self.complete_ids = ClozeBert_utils.read_data_json(dir_name)

        self.dataset_len = len(self.input_ids)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return {'input_ids': self.ids[idx],
                'masks': self.input_masks[idx],
                'token_type_ids': self.input_types[idx],
                'mask_ids': self.mask_ids,
                'options': self.options,
                'answers': self.answers}