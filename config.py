import os

import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, ElectraForSequenceClassification, \
    ElectraTokenizerFast, AutoModelForQuestionAnswering, AutoModelForCausalLM, PreTrainedTokenizerFast


class Config:
    models_dir: str = os.path.abspath("models")
    classifier: OneVsRestClassifier
    classifier_path: str = os.path.join(models_dir, "linear_svc_classifier.pickle")
    cv: CountVectorizer = CountVectorizer()
    dataset_dir: str = os.path.abspath("GermanQuAD")
    # device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device: torch.device = torch.device("cpu")
    germanquad_test_path: str = os.path.join(dataset_dir, "GermanQuAD_test.json")
    germanquad_train_path: str = os.path.join(dataset_dir, "GermanQuAD_train.json")
    is_alp: bool = False
    kbd: KBinsDiscretizer
    max_length: int = 1024  # 512 2048
    model_name: str = "malteos/gpt2-wechsel-german-ds-meg"
    # deepset/gelectra-large-germanquad facebook/xglm-564M deepset/gelectra-large deepset/gelectra-base
    model_path: str = "checkpoint-32"  # os.path.join(models_dir, "model.pt")
    num_labels: int = max_length  # 1 2 270
    _model = None
    # model: ElectraForQuestionAnswering = ElectraForQuestionAnswering.from_pretrained("deepset/gelectra-base").to(device)
    # model: BertForMaskedLM = AutoModelForMaskedLM.from_pretrained("bert-base-german-cased")
    _tokenizer = None
    # tokenizer: ElectraTokenizer = ElectraTokenizer.from_pretrained("deepset/gelectra-base")
    # tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained("bert-base-german-cased")
    use_checkpoint: bool = False

    @staticmethod
    def model() -> ElectraForSequenceClassification:
        if not Config._model:
            if Config.use_checkpoint:
                Config._model = AutoModelForSequenceClassification.from_pretrained(
                    Config.model_path, problem_type="multi_label_classification", num_labels=Config.num_labels).to(
                    Config.device)
            else:
                Config._model = AutoModelForQuestionAnswering.from_pretrained(
                    Config.model_name, problem_type="multi_label_classification",
                    num_labels=Config.num_labels).to(Config.device)  # regression
                # Config._model = AutoModelForCausalLM.from_pretrained(Config.model_name).to(Config.device)
                # Config._model = AutoModelForQuestionAnswering.from_pretrained(Config.model_name).to(
                #     Config.device)
        return Config._model

    @staticmethod
    def tokenizer() -> PreTrainedTokenizerFast:
        if not Config._tokenizer:
            Config._tokenizer = AutoTokenizer.from_pretrained(
                Config.model_name)  # deepset/gelectra-large-germanquad
        return Config._tokenizer


print("CUDA: ", torch.cuda.is_available())
