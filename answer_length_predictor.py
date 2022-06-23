import pickle
import random
import re
from collections import Counter
from typing import List

import numpy as np
import torch
from datasets import load_metric
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import LinearSVC
from torch.utils.data import Dataset, Subset
from tqdm import tqdm
from transformers import TrainingArguments, IntervalStrategy, Trainer, BatchEncoding, AutoModelForQuestionAnswering, \
    AutoModelForSequenceClassification
from transformers.integrations import TensorBoardCallback
import seaborn as sns
from transformers.modeling_outputs import SequenceClassifierOutput, QuestionAnsweringModelOutput

from config import Config
from data import GermanQuADdataset, DataItem, get_data_items, init_sklearn, QuestionType, get_question_type


class MultilabelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        class_frequencies: dict[int, int] = {
            4: 1029, 3: 972, 2: 913, 1: 874, 5: 872, 6: 588, 7: 513, 8: 448, 9: 372, 10: 337, 12: 270, 11: 257, 14: 248, 13: 244, 15: 206, 16: 179, 17: 171, 18: 167, 19: 140, 21: 135, 20: 133, 22: 125, 24: 119, 25: 115, 23: 103, 28: 94, 26: 90, 27: 85, 30: 84, 29: 82, 31: 67, 36: 65, 35: 65, 34: 63, 33: 60, 38: 55, 32: 51, 40: 51, 43: 45, 48: 45, 44: 45, 39: 45, 37: 44, 42: 41, 45: 41, 41: 40, 50: 34, 51: 32, 52: 30, 47: 29, 55: 27, 54: 25, 46: 22, 57: 22, 49: 19, 67: 19, 59: 18, 61: 18, 53: 17, 58: 15, 56: 15, 69: 15, 63: 15, 66: 15, 79: 13, 64: 13, 68: 12, 74: 11, 70: 11, 60: 10, 80: 10, 62: 10, 65: 10, 86: 10, 72: 10, 71: 9, 75: 8, 95: 7, 82: 7, 81: 7, 90: 7, 99: 7, 77: 7, 84: 7, 83: 6, 93: 6, 104: 6, 109: 6, 94: 6, 76: 6, 105: 5, 92: 5, 111: 4, 73: 4, 100: 4, 97: 4, 98: 4, 112: 4, 88: 4, 89: 4, 131: 3, 85: 3, 78: 3, 127: 3, 120: 3, 117: 3, 107: 3, 125: 2, 122: 2, 134: 2, 96: 2, 129: 2, 110: 2, 103: 2, 150: 2, 91: 2, 101: 2, 126: 2, 115: 2, 108: 2, 102: 1, 174: 1, 216: 1, 113: 1, 157: 1, 118: 1, 139: 1, 212: 1, 144: 1, 142: 1, 188: 1, 160: 1, 136: 1, 166: 1, 161: 1, 267: 1, 146: 1, 114: 1, 140: 1, 132: 1, 133: 1, 143: 1, 119: 1, 121: 1, 137: 1, 124: 1, 87: 1, 106: 1, 151: 1}
        weights: torch.Tensor = torch.zeros(Config.num_labels).to(Config.device)
        max_frequency: int = max(class_frequencies.values())
        for key in class_frequencies:
            weights[key] = class_frequencies[key] / max_frequency
        self.loss_fct = torch.nn.BCEWithLogitsLoss()
        # torch.nn.MSELoss()  # torch.nn.CrossEntropyLoss(label_smoothing=0.7, weight=weights)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs: dict, return_outputs=False):
        if Config.is_alp:
            alp_outputs: SequenceClassifierOutput
            qa_outputs: QuestionAnsweringModelOutput
            alp_outputs, qa_outputs = model(**inputs)
            # logits = alp_outputs if isinstance(alp_outputs, torch.Tensor) else alp_outputs.logits
            logits = alp_outputs.logits
            labels = inputs["labels"]
            alp_loss = self.loss_fct(logits.view(-1, Config.num_labels), labels.float().view(-1, Config.num_labels))
            # loss += qa_outputs.loss
            loss = qa_outputs.loss + alp_loss  # alp_outputs.loss
            return (loss, dict(alp=alp_outputs.logits, qa_start=qa_outputs.start_logits,
                               qa_end=qa_outputs.end_logits)) if return_outputs else loss
        else:
            outputs = model(**inputs)
            if isinstance(outputs[0], QuestionAnsweringModelOutput):
                return (outputs[0].loss, outputs) if return_outputs else outputs[0].loss
            logits = outputs[0] if isinstance(outputs[0], torch.Tensor) else outputs[0].logits
            loss = self.loss_fct(logits.view(-1), labels.float())
            return (loss, outputs) if return_outputs else loss


class AnswerLengthPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.answer_length_predictor = AutoModelForSequenceClassification.from_pretrained(
            Config.model_name, problem_type="multi_label_classification", num_labels=Config.num_labels)
        self.linear = torch.nn.Linear(768, Config.num_labels)
        self.qa = AutoModelForQuestionAnswering.from_pretrained(Config.model_name)
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor,
                start_positions: torch.Tensor, end_positions: torch.Tensor, labels: torch.Tensor):
        qa_outputs = self.qa(input_ids, attention_mask, token_type_ids, start_positions=start_positions,
                             end_positions=end_positions)
        # embeddings_of_last_layer = outputs[0]  # (batch_size, seq_len, emb_dim)
        # cls_embeddings = embeddings_of_last_layer[:, 0, :]
        # logits_raw = self.linear(cls_embeddings)
        # logits_scaled = self.softmax(logits_raw)
        # return logits_scaled
        alp_outputs = self.answer_length_predictor(input_ids, attention_mask, token_type_ids)  # , labels=labels
        return alp_outputs, qa_outputs


def compute_metrics(eval_preds) -> dict:
    logits, labels = eval_preds  # logits: alp_logits, qa_start_logits, qa_end_logits
    if Config.is_alp:
        logits = logits[0]
    metric = load_metric("squad_v2")  # load_metric("accuracy")
    predictions = np.argmax(logits, axis=-1)
    # restrict the number of labels to the actual number of predictions
    # true_labels = np.argmax(labels, axis=-1)[:len(predictions)]
    ret_val: dict = metric.compute(predictions=predictions, references=labels)
    # ret_val["rmse"] = mean_squared_error(true_labels, predictions, squared=False)
    return ret_val
    # return dict(rmse=mean_squared_error(labels, logits, squared=False))


def evaluate():
    test_dataset: Dataset = GermanQuADdataset(Config.germanquad_test_path)
    init_sklearn(GermanQuADdataset(Config.germanquad_train_path))
    # train_args: TrainingArguments = TrainingArguments(
    #     output_dir=".", dataloader_pin_memory=False, per_device_eval_batch_size=64, seed=42)
    # trainer: Trainer = MultilabelTrainer(model=Config.model(), args=train_args)
    # trainer.evaluate(eval_dataset=test_dataset)
    Config.model().eval()
    results: list[int] = []
    for i in tqdm(range(len(test_dataset))):
        inputs: dict = test_dataset[i]
        for k, v in inputs.items():
            inputs[k] = v.unsqueeze(0)
        labels = inputs.pop("labels")
        outputs = Config.model()(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits)
        true = torch.argmax(labels)
        # print(f"Pred: {int(pred)}, true: {int(true)}")
        results.append(pred == true)
    print("Accuracy: ", sum(results) / len(results))


def get_answer_length_by_question_type(data_items: List[DataItem]):
    exceptions: set[str] = {"Wir groß ist Namibias Straßennetz? ",
                            "In Zuge von wurde Abdülhamid II. osmanischer Sultan?",
                            "in Regionen im Westen gelangten die buddhistischen Missionare von König Ashoka?"}
    replacements: dict[str, str] = {"\n": "", "Wanne": "Wann", "Mit vielen": "Mit wie vielen"}
    question_type_dict: dict[QuestionType, list[int]] = dict()
    counter: Counter = Counter()
    for qt in QuestionType:
        question_type_dict[qt] = []
    for di in tqdm(data_items):
        if di.question in exceptions:
            continue
        question_normalized: str = di.question
        for key in replacements:
            question_normalized = question_normalized.replace(key, replacements[key])
        question_lower: str = question_normalized.lower()
        qt: QuestionType = get_question_type(question_lower)
        if qt == QuestionType.unknown:
            print(di.question)
            break
        question_type_dict[qt] += [len(x) for x in di.answers_true]
        for answer in di.answers_true:
            token_count: int = len(Config.tokenizer()(answer).encodings[0].ids[1:-1])
            counter.update({token_count: 1})
    answer_length_dict: dict[QuestionType, float] = dict()
    for key in question_type_dict:
        answer_length_dict[key] = sum(question_type_dict[key]) / max(len(question_type_dict[key]), 1)
    keys: list[QuestionType] = [x for x in question_type_dict]
    plt.boxplot([question_type_dict[x] for x in keys])
    x_labels: list[str] = [x.name for x in keys]
    plt.xticks(range(1, len(x_labels) + 1), x_labels, rotation=30)
    plt.grid(True)
    plt.savefig("answer_length_by_question_type.png", dpi=600)
    plt.show()


def get_max_answer_token_count(data_items: List[DataItem]) -> int:
    answer_token_counts: list[int] = []
    for data_item in tqdm(data_items):
        answers_encoded: BatchEncoding = Config.tokenizer()(data_item.answers_true)
        for input_id_list in answers_encoded.data["input_ids"]:
            answer_token_counts.append(len(input_id_list))
    g = sns.displot(answer_token_counts)
    plt.show()
    return max(answer_token_counts)


def train() -> None:
    eval_steps: int = 128  # 256 8 16
    batch_size: int = 1  # 4
    eval_batch_size: int = 16  # 64
    accumulation_steps: int = 1  # 16 1
    train_dataset: GermanQuADdataset = GermanQuADdataset(Config.germanquad_train_path)
    # init_sklearn(train_dataset)
    test_dataset: Dataset = GermanQuADdataset(Config.germanquad_test_path)
    all_indices: list[int] = list(range(len(test_dataset)))
    # reduce size of test dataset to increase evaluation speed
    test_dataset = Subset(test_dataset, random.sample(all_indices, int(len(all_indices) / 10)))
    model = AnswerLengthPredictor()
    # model = Config.model()

    # lrst = get_cosine_with_hard_restarts_schedule_with_warmup(
    #     optimizer=AdamW(model.parameters()), num_warmup_steps=8, num_training_steps=180, num_cycles=18)
    # optimizer = AdamW(model.parameters())
    train_args: TrainingArguments = TrainingArguments(
        output_dir=".", dataloader_pin_memory=False, logging_steps=eval_steps, logging_strategy=IntervalStrategy.STEPS,
        evaluation_strategy=IntervalStrategy.STEPS, eval_steps=eval_steps, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size, num_train_epochs=1, save_strategy=IntervalStrategy.STEPS,
        save_steps=eval_steps, gradient_accumulation_steps=accumulation_steps, seed=42,
        save_total_limit=3, learning_rate=1e-1, weight_decay=0.01)  #
    # 4e-3 OptimizerNames.ADAFACTOR optim=OptimizerNames.ADAMW_HF, lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
    trainer: Trainer = MultilabelTrainer(
        model=model, args=train_args, train_dataset=train_dataset, callbacks=[TensorBoardCallback],
        eval_dataset=test_dataset, compute_metrics=compute_metrics)  # , optimizers=(optimizer, lrst)
    trainer.train()


def train_pytorch():
    pass


def train_scikit():
    data_items: list[DataItem] = get_data_items(Config.germanquad_train_path)
    # data_items: list[DataItem] = get_data_items(Config.germanquad_test_path)
    Config.classifier = OneVsRestClassifier(LinearSVC(random_state=21, loss="hinge"))
    x_values: np.ndarray = Config.cv.fit_transform([x.question + "\n" + x.context for x in data_items])
    y_values: list[int] = [len(Config.tokenizer().encode(x.answers_true[0])[1:-1]) - 1 for x in tqdm(data_items)]
    kbd: KBinsDiscretizer = KBinsDiscretizer(n_bins=Config.num_labels, encode='ordinal', strategy='quantile')
    y_values = np.array(y_values).reshape(-1, 1)
    y_values = kbd.fit_transform(y_values)
    X_train, X_test, y_train, y_test = train_test_split(x_values, y_values, test_size=0.1, random_state=42)
    Config.classifier.fit(X_train, y_train)
    print("Train: ", Config.classifier.score(X_train, y_train))
    print("Test: ", Config.classifier.score(X_test, y_test))
    with open(Config.classifier_path, "wb+") as f:
        pickle.dump(Config.classifier, f)
    a = 0


# train()
# evaluate()
# get_max_answer_token_count(get_data_items(Config.germanquad_train_path))
# get_answer_length_by_question_type(get_data_items(Config.germanquad_train_path))
# train_scikit()
# train_pytorch()
