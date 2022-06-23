import json
import pickle
import re
from enum import Enum
from typing import Any, List, Tuple
from matplotlib.style import context

import numpy as np
import torch
from sklearn.preprocessing import KBinsDiscretizer
from torch.utils.data import Dataset
from transformers import TensorType, BatchEncoding
from transformers.file_utils import PaddingStrategy

from config import Config


class DataItem:
    def __init__(self, answers_predicted: list[str] = None, answers_true: list[str] = None, context: str = "",
                 question: str = "", answer_categories: list[str] = None):
        self.answer_categories: list[str] = answer_categories if answer_categories else []
        self.answers_predicted: list[str] = answers_predicted if answers_predicted else []
        self.answers_true: list[str] = answers_true if answers_true else []
        self.context: str = context
        self.question: str = question

    def get_position_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        token_ids: torch.Tensor = Config.tokenizer().encode(self.context, return_tensors=TensorType.PYTORCH)
        answer_ids: torch.Tensor = Config.tokenizer().encode(self.answers_true[0], return_tensors=TensorType.PYTORCH)
        start_idx, end_idx = find_sub_list(answer_ids[0], token_ids[0])
        if not start_idx:
            for answer in self.answers_true:
                if start_idx:
                        break
                start_idx_in_string: int = self.context.find(answer)
                while start_idx_in_string >= 0:
                    last_char_idx: int = start_idx_in_string + len(answer)
                    is_answer_last: bool = last_char_idx == len(self.context)
                    for i in range(2):  # number of chars
                        if start_idx:
                            break
                        for j in range(2): # forward/backward
                            if start_idx:
                                break
                            if is_answer_last and not j:
                                continue
                            new_start_idx: int = (start_idx_in_string - i - 1) if j else start_idx_in_string
                            new_end_idx: int = last_char_idx if j else (last_char_idx + i + 1)
                            new_answer: str = self.context[new_start_idx:new_end_idx]
                            answer_ids = Config.tokenizer().encode(new_answer, return_tensors=TensorType.PYTORCH)
                            start_idx, end_idx = find_sub_list(answer_ids[0], token_ids[0])
                    start_idx_in_string = self.context.find(answer, start_idx_in_string + 1)
        if not start_idx:
            print(f"Answer not found: {self.answers_true[0]} || Question: {self.question}")
        start: torch.Tensor = torch.tensor([start_idx], dtype=torch.long)
        end: torch.Tensor = torch.tensor([end_idx], dtype=torch.long)
        return start, end


class GermanQuADdataset(Dataset):
    def __init__(self, path: str):
        self.data_items: list[DataItem] = get_data_items(path)

    def __getitem__(self, idx: int) -> BatchEncoding:
        data_item: DataItem = self.data_items[idx]
        input_text: str = "\t".join([data_item.question, data_item.context])
        # "How long is the answer to the following question?" + "\n" +
        if not Config.tokenizer().pad_token:
            Config.tokenizer().add_special_tokens({'pad_token': '[PAD]'})
        encodings: BatchEncoding = Config.tokenizer()(
            input_text, padding=PaddingStrategy.MAX_LENGTH, max_length=Config.max_length,
            return_tensors=TensorType.PYTORCH, truncation=True)  # data_item.context,
        encodings.data = {k: v.squeeze().to(Config.device) for k, v in encodings.data.items()}
        # input_ids, attention_mask = (encodings["input_ids"].to(Config.device).squeeze(),
        #                              encodings["attention_mask"].to(Config.device).squeeze())

        # answer_tokens: torch.Tensor = torch.tensor(Config.tokenizer().encode(data_item.answers_true[0])[1:-1]).to(
        #     Config.device)
        # answer_token_count: int = len(answer_tokens)
        # encodings.data["labels"] = answer_token_count  # alp_labels

        # if Config.is_alp:
        #     answer_indices: tuple[int, int] = find_sub_list(answer_tokens, encodings.data["input_ids"])
        #     encodings.data["start_positions"] = [answer_indices[0]]
        #     encodings.data["end_positions"] = [answer_indices[1]]
        #     alp_labels: torch.Tensor = torch.tensor([0] * Config.num_labels, device=Config.device).float()
        #     alp_labels[answer_token_count - 1] = 1
        #     encodings.data["labels"] = alp_labels

        # label_bin: np.ndarray = Config.kbd.transform(np.array(answer_token_count - 1).reshape(-1, 1))
        # labels[int(label_bin[0])] = 1
        start_position, end_position = data_item.get_position_indices()
        encodings.data["start_positions"] = start_position
        encodings.data["end_positions"] = end_position
        encodings.data = {k: v.to(Config.device) for k,v in encodings.data.items()}
        encodings.data["question_indices"] = torch.tensor([idx])
        return encodings

    def __len__(self) -> int:
        return len(self.data_items)


class QuestionType(Enum):
    aboutwhat = r"(worum|worüber) "
    amount = r"(.* )?(wie ?viel.*)|inwiefern"
    entity = r"(.* )?(we[mnrs](sen)?|was|nenne|woraus|welch[a-z]+)[ .?!]"
    goal = r"(wozu|wofür) "
    how = r"(.* )?(wie|wodurch|womit|woran|worauf|worin) "
    spatial = r"(.* )?wo(her|hin|nach)? "
    temporal = r"(.* )?wann "
    # placeholder, never matches anything
    unknown = r"(?!x)x"
    whereof = r"(.* )?wovon "
    why = r"(warum|weshalb|wieso|wovor) "
    yesNo = r"(können|ist|dürfen|haben|hat|trinkt|schlagen) "


def find_sub_list(sl: torch.Tensor, l: torch.Tensor) -> Tuple[int, int]:
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if len(l[ind:ind + sll]) == sll:
            result: Any = l[ind:ind + sll] == sl
            equality_check: torch.Tensor = result
            if torch.all(equality_check):
                return ind, ind + sll - 1
    return 0, 0


def get_data_items(dataset_path: str) -> list[DataItem]:
    dataset: dict = json.load(open(dataset_path))
    paragraphs: list[dict] = [paragraph for item in dataset["data"] for paragraph in item["paragraphs"]]
    dis: list[DataItem] = []
    for paragraph in paragraphs:
        context: str = paragraph["context"]
        qas: list[dict] = paragraph["qas"]
        for i in range(len(qas)):
            question: str = qas[i]["question"]
            dis.append(DataItem(question=question, context=context, answers_true=[], answers_predicted=[]))
            for answer in qas[i]["answers"]:
                answer_text: str = answer["text"]
                dis[-1].answers_true.append(answer_text.strip())
                dis[-1].answer_categories.append(answer["answer_category"])
    return dis


def get_question_type(question: str) -> QuestionType:
    for qt in QuestionType:
        if re.match(qt.value, question):
            return qt
    return QuestionType.unknown


def init_sklearn(dataset: GermanQuADdataset):
    labels: list[int] = [len(Config.tokenizer().encode(x.answers_true[0])[1:-1]) - 1 for x in dataset.data_items]
    Config.kbd = KBinsDiscretizer(n_bins=Config.num_labels, encode='ordinal', strategy='quantile')
    Config.kbd.fit(np.array(labels).reshape(-1, 1))
    Config.cv.fit([x.question + "\n" + x.context for x in dataset.data_items])
    with open(Config.classifier_path, "rb") as f:
        Config.classifier = pickle.load(f)
