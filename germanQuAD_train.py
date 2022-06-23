import random
from answer_length_predictor import MultilabelTrainer, compute_metrics
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers import GPT2Model, GPT2PreTrainedModel, GPTJForQuestionAnswering, TrainingArguments, IntervalStrategy, Trainer
from transformers.integrations import TensorBoardCallback
from config import Config
from data import GermanQuADdataset

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
        question_indices=None
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
        ), question_indices


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
    eval_steps: int = 4  # 128 256 8 16
    batch_size: int = 1  # 4
    eval_batch_size: int = 4  # 16 64
    accumulation_steps: int = 1  # 16 1
    train_dataset: GermanQuADdataset = GermanQuADdataset(Config.germanquad_train_path)
    test_dataset: Dataset = GermanQuADdataset(Config.germanquad_test_path)
    all_indices: list[int] = list(range(len(test_dataset)))
    # reduce size of test dataset to increase evaluation speed
    test_dataset = Subset(test_dataset, random.sample(all_indices, len(all_indices) // 100))
    model = GPT2ForQuestionAnswering.from_pretrained(Config.model_name)
    train_args: TrainingArguments = TrainingArguments(
        output_dir=".", dataloader_pin_memory=False, logging_steps=eval_steps, logging_strategy=IntervalStrategy.STEPS,
        evaluation_strategy=IntervalStrategy.STEPS, eval_steps=eval_steps, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size, num_train_epochs=1, save_strategy=IntervalStrategy.STEPS,
        save_steps=eval_steps, gradient_accumulation_steps=accumulation_steps, seed=42, no_cuda=Config.device == torch.device("cpu"), save_total_limit=3, learning_rate=1e-1, weight_decay=0.01, label_names=["start_positions", "end_positions"])  #
    # 4e-3 OptimizerNames.ADAFACTOR optim=OptimizerNames.ADAMW_HF, lr_scheduler_type=SchedulerType.COSINE_WITH_RESTARTS,
    from transformers_qa import post_processing_function
    trainer: Trainer = MultilabelTrainer(
        model=model, args=train_args, train_dataset=train_dataset, callbacks=[TensorBoardCallback],
        eval_dataset=test_dataset, compute_metrics=compute_metrics, post_process_function=post_processing_function)  # , optimizers=(optimizer, lrst)
    # trainer.train()
    trainer.evaluate()


dataset: GermanQuADdataset = GermanQuADdataset(Config.germanquad_train_path)
# train_baseline(dataset)
train_gpt2()
