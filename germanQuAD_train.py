import logging
import os.path
from typing import Union

from datasets import load_dataset, Dataset, DatasetDict

from answer_length_predictor import MultilabelTrainer, compute_metrics
import torch
from torch.utils.data import DataLoader
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import EvalPrediction, GPT2Model, GPT2PreTrainedModel, GPTJForQuestionAnswering, TrainingArguments, \
    IntervalStrategy, Trainer, DefaultDataCollator
from transformers.integrations import TensorBoardCallback
from config import Config
from data import GermanQuADdataset, get_position_indices
from transformers_qa.trainer_qa import QuestionAnsweringTrainer
from transformers_qa.utils_qa import postprocess_qa_predictions

BATCH_SIZE: int = 2
LEARNING_RATE: float = 1e-4  # 5e-5


class GPT2ForQuestionAnswering(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"h\.\d+\.attn\.bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.qa_outputs = torch.nn.Linear(config.hidden_size, config.num_labels)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            id=None,  # question indices
            example_id=None,  # question indices
            offset_mapping=None  # char start & end indices for every token
    ):
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def post_processing_function(examples, features, predictions, hf_data: Dataset = None, stage="eval"):
    output_dir: str = os.path.abspath("output")
    os.makedirs(output_dir, exist_ok=True)
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=False,
        n_best_size=1,
        max_answer_length=270,
        null_score_diff_threshold=0,
        output_dir=output_dir,
        log_level=logging.WARNING,
        prefix=stage,
        hf_data=hf_data
    )
    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in
                             predictions.items()]
    references = [{"id": ex["id"], "answers": hf_data["id" == ex["id"]]["answers"]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def get_indices_from_offset_mapping(offset_mapping: list[tuple[int, int]], answer: dict[str, Union[int, str]],
                                    context: str) -> tuple[int, int]:
    start: int = 0
    end: int = 0
    answer_text: str = answer["text"][0]
    answer_start_char_idx: int = int(answer["answer_start"][0]) + 9
    answer_ids: list[int] = Config.tokenizer().encode(answer_text)
    answer_len: int = len(answer_ids)
    for i in range(len(offset_mapping)):
        chunk: list[tuple[int, int]] = offset_mapping[i:i + answer_len + 1]
        end_char_idx: int = chunk[-1][1]
        if not end_char_idx:
            continue
        start_char_idx: int = chunk[0][0]
        selected_context_span: str = context[start_char_idx:end_char_idx + 1]
        if answer_text in selected_context_span:
            old_chunk: list[tuple[int, int]] = chunk
            # remove superfluous tokens
            for j in range(2):
                while answer_text in selected_context_span:
                    old_chunk: list[tuple[int, int]] = chunk
                    chunk = chunk[1:] if j else chunk[:-1]
                    if len(chunk) == 0:
                        break
                    selected_context_span = context[chunk[0][0]:chunk[-1][1] + 1]
                chunk = old_chunk
                selected_context_span = context[chunk[0][0]:chunk[-1][1] + 1]
            start, end = chunk[0][0], chunk[-1][1]
            token_start_idx: int = next(k for k in range(len(offset_mapping)) if offset_mapping[k][0] == start)
            token_end_idx: int = next(k for k in range(len(offset_mapping)) if offset_mapping[k][1] == end)
            return token_start_idx, token_end_idx
    if not start:
        print(f"Answer not found: {answer_text}")  # || Question: {question}
    return 0, 0


def preprocess_function(examples):
    questions = [f"\n\nFrage: {q.strip()}" for q in examples["question"]]
    contexts = [f"Kontext: {x}" for x in examples["context"]]
    inputs = Config.tokenizer().batch_encode_plus(
        [(contexts[i], questions[i]) for i in range(len(questions))],
        max_length=1024 - 50,  # 384
        truncation="only_first",  # only_second
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = inputs["offset_mapping"]
    answers = examples["answers"]
    start_positions = []
    end_positions = []
    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        # start, end = get_position_indices(contexts[i], answer["text"], questions[i])
        start, end = get_indices_from_offset_mapping(offset_mapping[i], answer, contexts[i])
        start_positions.append(start)
        end_positions.append(end)
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["id"] = inputs["example_id"] = examples["id"]
    return inputs


def train_baseline(dataset: GermanQuADdataset):
    data_loader: DataLoader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    Config.model().train()
    optimizer = torch.optim.Adam(params=Config.model().parameters(), lr=LEARNING_RATE)
    for idx_train, batch in enumerate(data_loader, 0):
        optimizer.zero_grad()
        qamo: QuestionAnsweringModelOutput = Config.model()(**batch)  # (batch_size, dim, num_labels)
        qamo.loss.backward()
        optimizer.step()


def train_gpt2():
    eval_steps: int = 2  # 10 128 256 8 16
    batch_size: int = 1  # 4
    eval_batch_size: int = 4  # 4 16 64
    accumulation_steps: int = 1  # 16 1
    # pad on the left to enable batched generation; we always condition on the last token to predict the next one
    Config.tokenizer().padding_side = "left"
    # manually add a pad token because GPT2 does not have one by default
    Config.tokenizer().pad_token = Config.tokenizer().eos_token
    train_dataset: Dataset
    test_dataset: Dataset
    train_dataset, test_dataset = load_dataset("deepset/germanquad", "plain_text", split=[f"train", f"test[:100]"])
    train_dataset, test_dataset = train_dataset.shuffle(), test_dataset.shuffle()
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,
                                              remove_columns=test_dataset.column_names)
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True,
                                                remove_columns=train_dataset.column_names)
    model: GPT2ForQuestionAnswering = GPT2ForQuestionAnswering.from_pretrained(Config.model_name)
    data_collator: DefaultDataCollator = DefaultDataCollator()
    experiment_data_dir: str = os.path.abspath("./exp")  # checkpoints
    train_args: TrainingArguments = TrainingArguments(
        output_dir=experiment_data_dir, dataloader_pin_memory=False, logging_steps=eval_steps,
        logging_strategy=IntervalStrategy.STEPS, evaluation_strategy=IntervalStrategy.STEPS, eval_steps=eval_steps,
        per_device_train_batch_size=batch_size, per_device_eval_batch_size=eval_batch_size, num_train_epochs=1,
        save_strategy=IntervalStrategy.STEPS, save_steps=eval_steps, gradient_accumulation_steps=accumulation_steps,
        seed=42, warmup_steps=60, no_cuda=Config.device == torch.device("cpu"), save_total_limit=3, learning_rate=1e-1,
        label_names=["start_positions", "end_positions"], remove_unused_columns=False)
    # 4e-3 OptimizerNames.ADAFACTOR optim=OptimizerNames.ADAMW_HF, lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
    trainer: Trainer = QuestionAnsweringTrainer(
        model=model, args=train_args, train_dataset=tokenized_train_dataset, callbacks=[TensorBoardCallback],
        eval_dataset=tokenized_test_dataset, compute_metrics=compute_metrics, data_collator=data_collator,
        post_process_function=post_processing_function, hf_data=test_dataset,
        eval_examples=tokenized_test_dataset)  # , optimizers=(optimizer, lrst) MultilabelTrainer
    trainer.train()
    # trainer.evaluate()


# train_baseline(dataset)
train_gpt2()
