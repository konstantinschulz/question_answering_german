import json
import os.path
import string
import numpy
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BatchEncoding, TensorType
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from config import Config
from data import DataItem, get_data_items, QuestionType, get_question_type
from enums import AnswerLengthComparison

BATCH_SIZE: int = 2
LEARNING_RATE: float = 1e-4  # 5e-5
train_dataset_path: str = os.path.join(Config.dataset_dir, "train.txt")


class QAdataset(Dataset):
    def __init__(self, path: str):
        with open(path) as f:
            self.lines: list[str] = f.readlines()

    def __getitem__(self, idx: int) -> BatchEncoding:
        line_parts: list[str] = self.lines[idx].split("?")
        question: BatchEncoding = Config.tokenizer().encode_plus(
            f"{line_parts[0]}?", truncation=True, padding="max_length", max_length=512,
            return_tensors=TensorType.PYTORCH)
        # answer: BatchEncoding = tokenizer.encode_plus(line_parts[1][1:-1], truncation=True, padding="max_length",
        #                                               max_length=512, return_tensors=TensorType.PYTORCH)
        # return dict(question=question, answer=answer)
        return question

    def __len__(self) -> int:
        return len(self.lines)


def compare_predicted_to_true_answer(pred: str, real_answers: list[str], do_print: bool = False) -> \
        AnswerLengthComparison:
    result: str = ""
    alc: AnswerLengthComparison
    while True:
        if not pred:
            alc = AnswerLengthComparison.NONE
            break
        else:
            result = next((x for x in real_answers if x == pred), "")
            if result:
                alc = AnswerLengthComparison.EQUAL
                break
            else:
                result = next((x for x in real_answers if pred in x), "")
                if result:
                    alc = AnswerLengthComparison.TOO_SHORT
                    break
                else:
                    result = next((x for x in real_answers if x in pred), "")
                    if result:
                        alc = AnswerLengthComparison.TOO_LONG
                        break
                    else:
                        alc = AnswerLengthComparison.NONE
                        break
    if do_print:
        print(alc, f"pred: {pred} || real: {result}")
    return alc


def analyze_answer_length(y_true: list[list[list[str]]], y_pred: list[list[list[str]]], do_print: bool = False):
    alc: dict[AnswerLengthComparison, int] = dict()
    alc[AnswerLengthComparison.TOO_SHORT] = 0
    alc[AnswerLengthComparison.TOO_LONG] = 0
    alc[AnswerLengthComparison.EQUAL] = 0
    alc[AnswerLengthComparison.NONE] = 0
    mse: float = mean_squared_error([len(y) for x in y_true for y in x], [len(y) for x in y_pred for y in x])
    print("MSE: ", mse)
    for i in range(len(y_true)):
        real_answers: list[str] = [Config.tokenizer().convert_tokens_to_string(x) for x in y_true[i]]
        pred: str = Config.tokenizer().convert_tokens_to_string(y_pred[i][0])
        alc_value: AnswerLengthComparison = compare_predicted_to_true_answer(pred, real_answers, do_print)
        alc[alc_value] += 1
        # for j in range(len(y_true[i])):
        #     pred: str = Config.tokenizer.convert_tokens_to_string(y_pred[i][j])
        #     real: str = Config.tokenizer.convert_tokens_to_string(y_true[i][j])
        #     msg: str = ""
        #     if not pred:
        #         none += 1
        #         msg = f"None: {pred}"
        #     elif pred == real:
        #         equal += 1
        #     elif pred in real:
        #         too_short += 1
        #         msg = f"Short: {pred} || {real}"
        #     elif real in pred:
        #         too_long += 1
        #         msg = f"Long: {real} || {pred}"
        #     else:
        #         none += 1
        #         msg = f"None: {pred}"
        #     if do_print:
        #         print(msg)
    print(
        f"Long: {alc[AnswerLengthComparison.TOO_LONG]}, Short: {alc[AnswerLengthComparison.TOO_SHORT]}, Equal: {alc[AnswerLengthComparison.EQUAL]}, None: {alc[AnswerLengthComparison.NONE]}")


def analyze_answer_overlap(y_true: list[list[list[str]]], y_pred: list[list[list[str]]], data_items: list[DataItem]):
    overlaps: list[float] = []
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            matches: int = sum([1 for x in y_true[i][j] if x in y_pred[i][j]])
            overlaps.append(matches / len(y_true[i][j]))
    print("lexical overlap: ", sum(overlaps) / len(overlaps))
    results: list[int] = []
    for data_item in tqdm(data_items):
        question_result: int = 0
        for i in range(len(data_item.answers_true)):
            positions_true: tuple[int, int] = get_position(data_item.context, data_item.answers_true[i])
            positions_pred: tuple[int, int] = get_position(data_item.context, data_item.answers_predicted[i])
            set_true: set[int] = set(range(positions_true[0], positions_true[1]))
            range_pred: range = range(positions_pred[0], positions_pred[1])
            question_result += 1 if len(set_true.intersection(range_pred)) > 0 else 0
        results.append(1 if question_result > 0 else 0)
    print("positional overlap: ", sum(results) / len(results))


def compute_answers_for_data_items(data_items: list[DataItem]) -> list[list[str]]:
    inputs: list[tuple[str, str]] = [(x.question, x.context) for x in data_items]
    encodings = Config.tokenizer().batch_encode_plus(inputs, padding="max_length", max_length=512,
                                                     return_tensors=TensorType.PYTORCH, truncation=True)
    input_ids, attention_mask = encodings["input_ids"].to(Config.device), encodings["attention_mask"].to(Config.device)
    qamo: QuestionAnsweringModelOutput = Config.model()(input_ids, attention_mask=attention_mask)
    scores_start, scores_end = tuple(qamo.values())
    results: list[list[str]] = []
    # move to CPU so we can use numpy
    input_ids = input_ids.cpu()
    scores_start = scores_start.detach().cpu()
    scores_end = scores_end.detach().cpu()
    for i in range(len(input_ids)):
        # start_index, end_index = (torch.argmax(scores_start[i]), torch.argmax(scores_end[i]))
        start_index, end_index = get_best_indices(scores_start[i], scores_end[i], data_items[i], input_ids[i])
        results.append(get_tokens_by_index(input_ids[i], start_index, end_index))
    return results


def evaluate():
    y_pred: list[list[list[str]]] = []
    y_true: list[list[list[str]]] = []
    batch: list[DataItem] = []
    data_items: list[DataItem] = get_data_items(Config.germanquad_test_path)
    # init_sklearn(GermanQuADdataset(Config.germanquad_train_path))
    # data_items = random.sample(data_items, 100)
    Config.model().eval()
    total: int = len(data_items)
    for i in tqdm(range(total)):
        batch.append(data_items[i])
        y_true.append([Config.tokenizer().tokenize(x) for x in data_items[i].answers_true])
        if len(batch) >= BATCH_SIZE or i == total - 1:
            answers_predicted: list[list[str]] = compute_answers_for_data_items(batch)
            for k in range(len(answers_predicted)):
                answer_string: str = Config.tokenizer().convert_tokens_to_string(answers_predicted[k])
                batch[k].answers_predicted += [answer_string] * len(batch[k].answers_true)
                new_answers: list[list[str]] = []
                for j in range(len(y_true[len(y_pred)])):
                    new_answers.append(answers_predicted[k])
                y_pred.append(new_answers)
            idx: int = i + 1
            data_items[idx - len(batch):idx] = batch
            batch = []
    analyze_answer_length(y_true, y_pred, do_print=False)
    # analyze_answer_overlap(y_true, y_pred, data_items)


def get_answer_by_index(input_ids_subset: torch.Tensor, start_idx: int, end_idx: int) -> str:
    old_tokens: list[str] = get_tokens_by_index(input_ids_subset, start_idx, end_idx)
    return Config.tokenizer().convert_tokens_to_string(old_tokens)


def get_average_answer_length() -> None:
    data_items: list[DataItem] = get_data_items(Config.germanquad_test_path)
    question_lengths: list[int] = []
    answer_lengths: list[int] = []
    for data_item in data_items:
        for answer in data_item.answers_true:
            question_lengths.append(len(data_item.question.split()))
            answer_lengths.append(len(answer.split()))
    from sklearn.linear_model import LinearRegression
    x: np.ndarray = np.array(question_lengths).reshape(-1, 1)
    y: np.ndarray = np.array(answer_lengths).reshape(-1, 1)
    lr: LinearRegression = LinearRegression().fit(x, y)
    print(lr.coef_, lr.intercept_)


def get_best_indices(start: torch.Tensor, end: torch.Tensor, data_item: DataItem, input_ids_subset: torch.Tensor) -> \
        tuple[int, int]:
    true_ranges: list[tuple[int, int]] = get_true_answer_ranges(data_item, input_ids_subset)
    # check if the correct answer was removed from the context because of truncation
    if not true_ranges:
        return 0, 0
    start_tuples_src: list[tuple[int, torch.Tensor]] = sorted(enumerate(start), key=lambda x: x[1], reverse=True)
    end_tuples_src: list[tuple[int, torch.Tensor]] = sorted(enumerate(end), key=lambda x: x[1], reverse=True)
    top_n: int = 5
    start_tuples: list[tuple[int, torch.Tensor]] = start_tuples_src[:top_n]
    end_tuples: list[tuple[int, torch.Tensor]] = end_tuples_src[:top_n]
    # x_value: np.ndarray = Config.cv.transform([data_item.question + "\n" + data_item.context])
    # label: int = int(Config.classifier.predict(x_value))
    # answer_edges: tuple[int, int] = (Config.kbd.bin_edges_[0][label], Config.kbd.bin_edges_[0][label + 1])
    predictions: list[tuple[torch.Tensor, int, int]] = get_start_end_predictions(start_tuples, end_tuples, data_item)
    # answer_edges
    predictions.sort(key=lambda x: x[0], reverse=True)
    # if predictions[0][1:] not in true_ranges:
    #     new_predictions: list[tuple[torch.Tensor, int, int]] = []
    #     true_ranges_lengths: list[int] = [(x[1] - x[0]) for x in true_ranges]
    #     for i in range(len(predictions)):
    #         pred_length: int = predictions[i][2] - predictions[i][1]
    #         closest_true_length: int = min(true_ranges_lengths, key=lambda x: abs(x - pred_length))
    #         min_diff: float = abs(pred_length - closest_true_length)
    #         new_score: int = predictions[i][0] - min_diff
    #         new_predictions.append((new_score,) + predictions[i][1:])
    #     new_predictions.sort(key=lambda x: x[0], reverse=True)
    #     predictions = new_predictions
    #     # if new_predictions[0][1:] != predictions[0][1:]:
    #     #     old: str = get_answer_by_index(input_ids_subset, predictions[0][1], predictions[0][2])
    #     #     new: str = get_answer_by_index(input_ids_subset, new_predictions[0][1], new_predictions[0][2])
    #     #     alc_old: AnswerLengthComparison = compare_predicted_to_true_answer(old, data_item.answers_true)
    #     #     alc_new: AnswerLengthComparison = compare_predicted_to_true_answer(new, data_item.answers_true)
    #     #     print(f"{alc_old} >>> {alc_new}")
    return (predictions[0][1], predictions[0][2]) if predictions else (0, 0)


def get_position(context: str, answer: str) -> tuple[int, int]:
    start: int = context.find(answer)
    if start < 0:
        new_answer: str = ''.join(ch for ch in answer if ch not in set(string.punctuation))
        start = context.find(new_answer)
    return start, start + len(answer)


def get_short_answer_samples() -> None:
    short_answer_dict: dict[str, str] = dict()
    data_items: list[DataItem] = get_data_items(Config.germanquad_test_path)
    for data_item in data_items:
        if all(len(answer.split(" ")) < 3 for answer in data_item.answers_true):
            short_answer_dict[data_item.question] = data_item.answers_true[0]
    json.dump(short_answer_dict, open(os.path.join(Config.dataset_dir, "GermanQuAD_short.json"), "w+"))


def get_start_end_predictions(
        start_tuples: list[tuple[int, torch.Tensor]], end_tuples: list[tuple[int, torch.Tensor]],
        data_item: DataItem = None, answer_edges: tuple[int, int] = None) -> list[tuple[torch.Tensor, int, int]]:
    predictions: list[tuple[torch.Tensor, int, int]] = []
    tensor_list: list[list[list[torch.Tensor]]] = []
    for start_idx, start_logit in start_tuples:
        tensor_list.append([[start_logit, end_logit] for idx, end_logit in end_tuples])
    array: numpy.ndarray = numpy.array(tensor_list)
    pairwise_sum: numpy.ndarray = array.sum(axis=-1)
    long_qts: set[QuestionType] = {QuestionType.aboutwhat, QuestionType.goal, QuestionType.why, QuestionType.yesNo}
    short_qts: set[QuestionType] = {QuestionType.amount, QuestionType.temporal}
    question_lower: str = data_item.question.lower()
    qt: QuestionType = get_question_type(question_lower)
    is_long: bool = qt in long_qts
    is_short: bool = qt in short_qts
    for j in range(len(start_tuples)):
        start_idx: int = start_tuples[j][0]
        for i in range(len(end_tuples)):
            end_idx: int = end_tuples[i][0]
            if end_idx < start_idx:
                continue
            score: torch.Tensor = pairwise_sum[j][i]
            # if answer_length < answer_edges[0] or answer_length > answer_edges[1]:
            #     score *= 0.9

            # average true answer length throughout the corpus
            # all_avg: float = 11.88295593635251
            # short_avg: float = 4.5
            # long_avg: float = 22.1
            answer_length: int = end_idx - start_idx
            # if is_short or is_long:
            #     avg: float = short_avg if is_short else long_avg
            #     score /= (avg - answer_length) ** 2
            if qt in [QuestionType.amount, QuestionType.temporal, QuestionType.spatial]:
                avg: float = 4.5 if qt in [QuestionType.amount, QuestionType.temporal] else 9.9
                squared_error: float = (avg - answer_length) ** 2
                sigmoid = torch.sigmoid(torch.tensor(squared_error))
                score /= float(sigmoid)
            # if (is_short and answer_length > avg) or (is_long and answer_length < avg):
            #     score *= 0.95
            predictions.append((score, start_idx, end_idx))
    return predictions


def get_tokens_by_index(input_ids_subset: torch.Tensor, start_index: int, end_index: int) -> list[str]:
    tokens = input_ids_subset[start_index: end_index + 1]
    return Config.tokenizer().convert_ids_to_tokens(tokens, skip_special_tokens=True)


def get_true_answer_ranges(data_item: DataItem, input_ids: torch.Tensor) -> list[tuple[int, int]]:
    true_ranges: list[tuple[int, int]] = []
    for answer_true in data_item.answers_true:
        # ignore CLS and SEP tokens from encoder
        tokens: torch.Tensor = Config.tokenizer().encode(answer_true, return_tensors=TensorType.PYTORCH).squeeze()[1:-1]
        intersection_indices: torch.Tensor = (input_ids.unsqueeze(1) == tokens).nonzero()
        start_indices: list[int] = [int(x) for x in torch.where(intersection_indices[:, 1:].squeeze() == 0)[0]]
        for start_idx in start_indices:
            start_idx = int(intersection_indices[start_idx][0])
            end_idx: int = start_idx + len(tokens) - 1
            if torch.equal(input_ids[start_idx:end_idx + 1], tokens):
                true_ranges.append((start_idx, end_idx))
                break
    return true_ranges


def make_dataset():
    data_items: list[DataItem] = get_data_items(Config.germanquad_train_path)
    lines: list[str] = []
    for data_item in data_items:
        for answer in data_item.answers_true:
            if not answer:
                print(data_item.question)
            lines.append(f"{data_item.question} {answer}\n")
    with open(train_dataset_path, "w+") as f:
        f.writelines(lines)


def train_baseline(dataset: QAdataset):
    data_loader: DataLoader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    Config.model().train()
    optimizer = torch.optim.Adam(params=Config.model().parameters(), lr=LEARNING_RATE)
    for idx_train, batch in enumerate(data_loader, 0):
        outputs = Config.model()(**batch)  # (batch_size, dim, num_labels)
        a = 0


question: str = "Welcher Name wird auch verwendet, um den Amazonas-Regenwald auf Englisch zu beschreiben?"
question_context: str = 'Der Amazonas-Regenwald, auf Englisch auch als Amazonien oder Amazonas-Dschungel bekannt, ist ein feuchter Laubwald, der den größten Teil des Amazonas-Beckens Südamerikas bedeckt. Dieses Becken umfasst 7.000.000 Quadratkilometer (2.700.000 Quadratmeilen), von denen 5.500.000 Quadratkilometer (2.100.000 Quadratmeilen) vom Regenwald bedeckt sind. Diese Region umfasst Gebiete von neun Nationen. Der größte Teil des Waldes befindet sich in Brasilien mit 60% des Regenwaldes, gefolgt von Peru mit 13%, Kolumbien mit 10% und geringen Mengen in Venezuela, Ecuador, Bolivien, Guyana, Suriname und Französisch-Guayana. Staaten oder Abteilungen in vier Nationen enthalten "Amazonas" in ihren Namen. Der Amazonas repräsentiert mehr als die Hälfte der verbleibenden Regenwälder des Planeten und umfasst den größten und artenreichsten tropischen Regenwald der Welt mit geschätzten 390 Milliarden Einzelbäumen, die in 16.000 Arten unterteilt sind.'
# print(get_answer(question, question_context))
# get_short_answer_samples()
evaluate()
# get_average_answer_length()
# make_dataset()
# dataset: QAdataset = QAdataset(train_dataset_path)
# train_baseline(dataset)
