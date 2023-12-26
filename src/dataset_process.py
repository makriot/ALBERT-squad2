import numpy as np
import torch
from tqdm.auto import tqdm
from random import shuffle
from random import seed
from typing import Dict, List


def create_list_repres(ds_train, ds_val):
    seed(0)

    dataset_train = list(ds_train)
    dataset_validation = list(ds_val)
    shuffle(dataset_train)
    shuffle(dataset_validation)


def unpack_answers(ds):
    for answer in ds:
        if len(answer["answers"]["text"]) == 0:
            answer["answers"]["text"] = ""
            answer["answers"]["answer_start"] = -1
        else:
            answer["answers"]["text"] = answer["answers"]["text"][0]
            answer["answers"]["answer_start"] = answer["answers"]["answer_start"][0]


def add_end_index(ds):
    for item in ds:
        start_idx = item["answers"]["answer_start"]
        end_idx = start_idx + len(item["answers"]["text"])
        item["answers"]["answer_end"] = end_idx


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, ds, max_length=512):
        # clean <= 512 here
        self.ds = [item for item in tqdm(ds, total=len(ds))
                            if (tokenizer(item["context"], return_tensors='pt')["input_ids"].shape[-1] <= max_length)]

    def __getitem__(self, idx):
        return self.ds[idx]

    def __len__(self):
        return len(self.ds)


def collate_fn(tokenizer, batch: List[Dict]):
    encoded_context = tokenizer([i['context'] for i in batch], return_tensors='pt', padding=True)
    encoded_query = tokenizer([i['question'] for i in batch], return_tensors='pt', padding=True)
    answer = []
    for i, item in enumerate(batch):
        if item["answers"]["answer_end"] == -1:
            answer.append((0, 0))
        else:
            answer.append(
                (
                    encoded_context[i].char_to_token(item["answers"]["answer_start"]),
                    encoded_context[i].char_to_token(item["answers"]["answer_end"] - 1) # because index of end is shifted to the next character out of token
                )
            )
    items = {
        'id': [i['id'] for i in batch],
        'context': encoded_context,
        'question': encoded_query,
        'answers': torch.tensor(answer)
    }
    return items