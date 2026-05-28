import os
import torch
import torch.nn as nn
from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass
import random
from collections import defaultdict

import random
from collections import defaultdict
from torch.utils.data import DataLoader, Sampler

from transformers import Trainer, GenerationConfig
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy,
    has_length,
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS
)
from transformers.trainer_utils import EvalLoopOutput
from torch.utils.data import DataLoader
from train.train_utils import get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3

from constants import IGNORE_INDEX


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


@dataclass
class GenerativeEvalPrediction:
    """Container for generative evaluation predictions."""
    predictions: List[str]
    references: List[str]

def get_arcface_head_from_model(model):
    if hasattr(model, "arcface_head"):
        return model.arcface_head

    if hasattr(model, "module") and hasattr(model.module, "arcface_head"):
        return model.module.arcface_head

    if hasattr(model, "base_model") and hasattr(model.base_model, "arcface_head"):
        return model.base_model.arcface_head

    if hasattr(model, "model") and hasattr(model.model, "arcface_head"):
        return model.model.arcface_head

    return None
class LabelGroupedBatchSampler(Sampler):
    """
    Create batches with P labels and K samples per label.

    batch_size = labels_per_batch * samples_per_label
    """

    def __init__(
        self,
        dataset,
        labels_per_batch: int = 4,
        samples_per_label: int = 2,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.labels_per_batch = labels_per_batch
        self.samples_per_label = samples_per_label
        self.drop_last = drop_last

        self.label_to_indices = defaultdict(list)

        for idx, item in enumerate(dataset.list_data_dict):
            if "label_id" not in item:
                raise ValueError(
                    "arc_group_batch=True nhưng có sample không có label_id."
                )
            self.label_to_indices[int(item["label_id"])].append(idx)

        self.labels = list(self.label_to_indices.keys())

        valid_labels = [
            label for label in self.labels
            if len(self.label_to_indices[label]) >= self.samples_per_label
        ]

        if len(valid_labels) < self.labels_per_batch:
            raise ValueError(
                f"Không đủ label hợp lệ để tạo batch. "
                f"Cần ít nhất {self.labels_per_batch} label, "
                f"mỗi label có >= {self.samples_per_label} sample."
            )

        self.labels = valid_labels

        self.batch_size = self.labels_per_batch * self.samples_per_label

        total_samples = sum(len(self.label_to_indices[label]) for label in self.labels)
        self.num_batches = total_samples // self.batch_size

    def __iter__(self):
        labels = self.labels[:]
        random.shuffle(labels)

        for _ in range(self.num_batches):
            selected_labels = random.sample(labels, self.labels_per_batch)

            batch = []
            for label in selected_labels:
                indices = self.label_to_indices[label]

                if len(indices) >= self.samples_per_label:
                    chosen = random.sample(indices, self.samples_per_label)
                else:
                    chosen = random.choices(indices, k=self.samples_per_label)

                batch.extend(chosen)

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches

class PairGroupedBatchSampler(Sampler):
    """
    Batch sampler ưu tiên đưa RGB và segment có cùng pair_id vào cùng batch.

    Mỗi batch có dạng:
        arc_pairs_per_batch pair_id × arc_samples_per_pair sample / pair_id

    Ví dụ:
        arc_pairs_per_batch = 4
        arc_samples_per_pair = 2

    Batch thật:
        4 pair × 2 sample = 8 sample

    Nếu dữ liệu mỗi pair_id có:
        - 1 RGB
        - 1 segment

    thì mỗi batch sẽ gồm:
        pair A: RGB + segment
        pair B: RGB + segment
        pair C: RGB + segment
        pair D: RGB + segment
    """

    def __init__(
        self,
        dataset,
        pairs_per_batch: int = 4,
        samples_per_pair: int = 2,
        drop_incomplete_pairs: bool = True,
    ):
        self.dataset = dataset
        self.pairs_per_batch = int(pairs_per_batch)
        self.samples_per_pair = int(samples_per_pair)
        self.drop_incomplete_pairs = bool(drop_incomplete_pairs)

        if self.pairs_per_batch <= 0:
            raise ValueError("pairs_per_batch must be > 0")

        if self.samples_per_pair <= 0:
            raise ValueError("samples_per_pair must be > 0")

        if not hasattr(dataset, "list_data_dict"):
            raise ValueError(
                "PairGroupedBatchSampler requires dataset.list_data_dict. "
                "Your SFT dataset must keep raw JSON records in list_data_dict."
            )

        self.pair_to_indices = defaultdict(list)
        missing_pair_id = 0

        for idx, item in enumerate(dataset.list_data_dict):
            pair_id = item.get("pair_id", None)

            if pair_id is None:
                missing_pair_id += 1
                continue

            pair_id = str(pair_id)
            self.pair_to_indices[pair_id].append(idx)

        if missing_pair_id > 0:
            print(
                f"[PairGroupedBatchSampler] Warning: "
                f"{missing_pair_id} samples do not have pair_id and will be ignored."
            )

        if self.drop_incomplete_pairs:
            self.valid_pair_ids = [
                pair_id
                for pair_id, indices in self.pair_to_indices.items()
                if len(indices) >= self.samples_per_pair
            ]
        else:
            self.valid_pair_ids = list(self.pair_to_indices.keys())

        if len(self.valid_pair_ids) < self.pairs_per_batch:
            raise ValueError(
                f"Not enough valid pair_id groups. "
                f"Need at least {self.pairs_per_batch}, "
                f"but got {len(self.valid_pair_ids)}."
            )

        total_valid_samples = sum(
            len(self.pair_to_indices[pair_id])
            for pair_id in self.valid_pair_ids
        )

        self.batch_size = self.pairs_per_batch * self.samples_per_pair

        # Số batch xấp xỉ theo số sample hợp lệ.
        # Vì mỗi batch lấy theo pair, ta lấy floor để tránh epoch quá dài.
        self.num_batches = max(1, total_valid_samples // self.batch_size)

        print("[PairGroupedBatchSampler] Enabled")
        print("================================")
        print(f"Valid pair_id groups      : {len(self.valid_pair_ids)}")
        print(f"Pairs per batch           : {self.pairs_per_batch}")
        print(f"Samples per pair          : {self.samples_per_pair}")
        print(f"Real batch size           : {self.batch_size}")
        print(f"Estimated batches / epoch : {self.num_batches}")
        print("================================")

    def __iter__(self):
        pair_ids = self.valid_pair_ids[:]
        random.shuffle(pair_ids)

        cursor = 0

        for _ in range(self.num_batches):
            # Nếu gần hết list pair_id thì shuffle lại để tiếp tục epoch
            if cursor + self.pairs_per_batch > len(pair_ids):
                random.shuffle(pair_ids)
                cursor = 0

            selected_pair_ids = pair_ids[cursor: cursor + self.pairs_per_batch]
            cursor += self.pairs_per_batch

            batch_indices = []

            for pair_id in selected_pair_ids:
                indices = self.pair_to_indices[pair_id]

                if len(indices) >= self.samples_per_pair:
                    chosen = random.sample(indices, self.samples_per_pair)
                else:
                    # Chỉ xảy ra khi drop_incomplete_pairs=False
                    chosen = random.choices(indices, k=self.samples_per_pair)

                batch_indices.extend(chosen)

            random.shuffle(batch_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches
      
class QwenSFTTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super(QwenSFTTrainer, self).__init__(*args, **kwargs)
        # processing_class is set by parent Trainer from the constructor argument
        # We can access it via self.processing_class (same as processor)

    def _pool_arc_features(self, hidden_states, inputs):
        """
        hidden_states: [B, L, D]
        return: [B, D]
        """
        attention_mask = inputs["attention_mask"].bool()
        labels = inputs["labels"]

        pooling_mode = getattr(self.args, "arc_pooling", "prompt")

        if pooling_mode == "answer":
            mask = (labels != IGNORE_INDEX) & attention_mask

        elif pooling_mode == "all":
            mask = attention_mask

        elif pooling_mode == "last":
            lengths = attention_mask.long().sum(dim=1) - 1
            batch_idx = torch.arange(
                hidden_states.size(0),
                device=hidden_states.device
            )
            return hidden_states[batch_idx, lengths]

        else:
            # prompt = phần input/question/image context
            # trong SFT dataset, prompt token có labels == IGNORE_INDEX
            mask = (labels == IGNORE_INDEX) & attention_mask

        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask[empty] = attention_mask[empty]

        mask = mask.unsqueeze(-1).to(hidden_states.dtype)

        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        arc_labels = inputs.pop("arc_labels", None)

        arcface_head = get_arcface_head_from_model(model)

        use_arc_loss = (
            getattr(self.args, "use_arc_loss", False)
            and arc_labels is not None
            and arcface_head is not None
        )

        if use_arc_loss:
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = model(
                **inputs,
                return_dict=True,
            )

        sft_loss = outputs.loss

        if not use_arc_loss:
            return (sft_loss, outputs) if return_outputs else sft_loss

        hidden_states = outputs.hidden_states[-1]
        arc_features = self._pool_arc_features(hidden_states, inputs)

        arc_labels = arc_labels.to(arc_features.device)

        arc_logits = arcface_head(
            arc_features.float(),
            arc_labels
        )

        arc_loss = nn.CrossEntropyLoss()(
            arc_logits.float(),
            arc_labels
        )

        arc_weight = getattr(self.args, "arc_loss_weight", 0.05)
        total_loss = sft_loss + arc_weight * arc_loss

        if self.state.global_step % max(getattr(self.args, "logging_steps", 10), 1) == 0:
            self.log({
                "loss_sft": sft_loss.detach().float().item(),
                "loss_arc": arc_loss.detach().float().item(),
                "loss_total": total_loss.detach().float().item(),
            })

        return (total_loss, outputs) if return_outputs else total_loss
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if not getattr(self.args, "arc_pair_batch", False):
            return super().get_train_dataloader()

        batch_sampler = PairGroupedBatchSampler(
            dataset=self.train_dataset,
            pairs_per_batch=getattr(self.args, "arc_pairs_per_batch", 4),
            samples_per_pair=getattr(self.args, "arc_samples_per_pair", 2),
            drop_incomplete_pairs=getattr(self.args, "arc_drop_incomplete_pairs", True),
        )

        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        return self.accelerator.prepare(dataloader)
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if not getattr(self.args, "arc_pair_batch", False):
            return super().get_train_dataloader()

        batch_sampler = PairGroupedBatchSampler(
            dataset=self.train_dataset,
            pairs_per_batch=getattr(self.args, "arc_pairs_per_batch", 4),
            samples_per_pair=getattr(self.args, "arc_samples_per_pair", 2),
            drop_incomplete_pairs=getattr(self.args, "arc_drop_incomplete_pairs", True),
        )

        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        return self.accelerator.prepare(dataloader)
    
    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

                if visual_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                            },
                        ]
                    )

                if merger_parameters:
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        super()._save_checkpoint(model, trial)

        if not self.args.lora_enable:
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        non_lora = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters(),
            require_grad_only=True,
        )


        if self.args.should_save:
            torch.save(non_lora, os.path.join(output_dir, "non_lora_state_dict.bin"))
            self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        labels = inputs.get("labels") if "labels" in inputs else None

        # arc_labels không phải input hợp lệ của Qwen forward,
        # nên phải bỏ ra trước khi gọi model(**inputs)
        inputs = dict(inputs)
        inputs.pop("arc_labels", None)

        with torch.no_grad():

            outputs = model(**inputs)

            loss = outputs.loss if hasattr(outputs, "loss") else None

            logits = outputs.logits if hasattr(outputs, "logits") else None

            if prediction_loss_only:

                return (loss, None, None)

            return (loss, logits, labels)

    def _extract_prompt_and_reference(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer
    ) -> tuple:
        """
        Extract prompt (question only) and reference (answer) from input_ids and labels.

        In SFT dataset, labels == IGNORE_INDEX for prompt tokens, and labels == token_id for answer tokens.

        Returns:
            prompt_ids: tensor of prompt token ids (question part only)
            reference_text: decoded answer text
        """
        # Find where labels are not IGNORE_INDEX (answer starts)
        label_mask = labels != IGNORE_INDEX

        if label_mask.any():
            answer_start_idx = label_mask.nonzero(as_tuple=True)[0][0].item()
        else:
            # No answer found, use full input as prompt
            answer_start_idx = len(input_ids)

        # Extract prompt (everything before answer)
        prompt_ids = input_ids[:answer_start_idx]

        # Extract reference answer
        answer_ids = labels[label_mask]
        reference_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

        return prompt_ids, reference_text

    def _prepare_generation_inputs(
        self,
        batch_prompt_ids: List[torch.Tensor],
        original_inputs: Dict[str, torch.Tensor],
        tokenizer,
        device
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for generation by padding prompts and including vision inputs.
        """
        batch_size = len(batch_prompt_ids)

        # Pad prompts to same length (left padding for generation)
        max_prompt_len = max(p.shape[0] for p in batch_prompt_ids)

        padded_prompts = torch.full(
            (batch_size, max_prompt_len),
            tokenizer.pad_token_id,
            dtype=batch_prompt_ids[0].dtype,
            device=device
        )
        attention_masks = torch.zeros(
            (batch_size, max_prompt_len),
            dtype=torch.long,
            device=device
        )

        # Right padding (Qwen uses right padding)
        for i, prompt in enumerate(batch_prompt_ids):
            prompt_len = len(prompt)
            padded_prompts[i, :prompt_len] = prompt
            attention_masks[i, :prompt_len] = 1

        gen_inputs = {
            "input_ids": padded_prompts,
            "attention_mask": attention_masks,
        }

        if "mm_token_type_ids" in original_inputs:
            padded_mm_token_type_ids = torch.zeros(
                (batch_size, max_prompt_len),
                dtype=original_inputs["mm_token_type_ids"].dtype,
                device=device,
            )
            for i, prompt in enumerate(batch_prompt_ids):
                prompt_len = len(prompt)
                padded_mm_token_type_ids[i, :prompt_len] = original_inputs["mm_token_type_ids"][i, :prompt_len]
            gen_inputs["mm_token_type_ids"] = padded_mm_token_type_ids

        # Add vision inputs if present
        if "pixel_values" in original_inputs:
            gen_inputs["pixel_values"] = original_inputs["pixel_values"]
        if "image_grid_thw" in original_inputs:
            gen_inputs["image_grid_thw"] = original_inputs["image_grid_thw"]
        if "pixel_values_videos" in original_inputs:
            gen_inputs["pixel_values_videos"] = original_inputs["pixel_values_videos"]
        if "video_grid_thw" in original_inputs:
            gen_inputs["video_grid_thw"] = original_inputs["video_grid_thw"]
        if "second_per_grid_ts" in original_inputs:
            gen_inputs["second_per_grid_ts"] = original_inputs["second_per_grid_ts"]

        return gen_inputs

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Override evaluation_loop to support generation-based evaluation.

        If compute_metrics is provided and prediction_loss_only is False,
        this method will use model.generate() to produce text outputs
        and pass them to compute_metrics as GenerativeEvalPrediction.

        Your compute_metrics function should accept either:
        - GenerativeEvalPrediction with .predictions (List[str]) and .references (List[str])
        - Or a dict with 'predictions' and 'references' keys
        """
        args = self.args

        # Determine if we should do generation-based evaluation
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        # If no compute_metrics or loss_only, fall back to default behavior
        if prediction_loss_only or self.compute_metrics is None:
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix
            )

        # Generation-based evaluation
        logger.info(f"\n***** Running {description} (Generation Mode) *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        logger.info(f"  Batch size = {self.args.eval_batch_size}")

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        # Get processor/tokenizer
        tokenizer = self.processing_class.tokenizer

        # Setup generation config
        generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=getattr(args, 'generation_max_new_tokens', 512),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Unwrap model for generation
        unwrapped_model = self.accelerator.unwrap_model(model)

        all_predictions = []
        all_references = []
        all_losses = []

        for step, inputs in enumerate(dataloader):
            # Move inputs to device
            inputs = self._prepare_inputs(inputs)

            batch_input_ids = inputs["input_ids"]
            batch_labels = inputs["labels"]
            batch_size = batch_input_ids.shape[0]

            # Compute loss using forward pass (optional, for logging)
            with torch.no_grad():
                outputs = model(**inputs)
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    loss = outputs.loss.detach()
                    # Gather loss across processes
                    loss = self.accelerator.gather(loss.repeat(batch_size))
                    all_losses.append(loss.cpu())

            # Extract prompts and references for each item in batch
            batch_prompt_ids = []
            batch_references = []

            for i in range(batch_size):
                prompt_ids, reference_text = self._extract_prompt_and_reference(
                    batch_input_ids[i],
                    batch_labels[i],
                    tokenizer
                )
                batch_prompt_ids.append(prompt_ids)
                batch_references.append(reference_text)

            # Prepare generation inputs
            gen_inputs = self._prepare_generation_inputs(
                batch_prompt_ids,
                inputs,
                tokenizer,
                batch_input_ids.device
            )

            # Generate
            with torch.no_grad():
                generated_ids = unwrapped_model.generate(
                    **gen_inputs,
                    generation_config=generation_config,
                )

            # Decode generated tokens (excluding prompt)
            for i in range(batch_size):
                prompt_len = len(batch_prompt_ids[i])
                new_tokens = generated_ids[i][prompt_len:]
                pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_predictions.append(pred_text)

            all_references.extend(batch_references)

            # Log progress
            if step % 10 == 0:
                logger.info(f"  Eval step {step}/{len(dataloader)}")

        # Gather predictions across processes if distributed
        if self.args.world_size > 1:
            # For distributed evaluation, we need to gather all predictions
            all_predictions = self._gather_predictions(all_predictions)
            all_references = self._gather_predictions(all_references)

        # Compute metrics
        eval_prediction = GenerativeEvalPrediction(
            predictions=all_predictions,
            references=all_references
        )

        metrics = self.compute_metrics(eval_prediction)

        # Add loss to metrics if available
        if all_losses:
            avg_loss = torch.cat(all_losses).mean().item()
            metrics[f"{metric_key_prefix}_loss"] = avg_loss

        # Prefix all metrics
        metrics = {
            f"{metric_key_prefix}_{k}" if not k.startswith(metric_key_prefix) else k: v
            for k, v in metrics.items()
        }

        self.log(metrics)

        return EvalLoopOutput(
            predictions=all_predictions,
            label_ids=all_references,
            metrics=metrics,
            num_samples=len(all_predictions),
        )

    def _gather_predictions(self, predictions: List[str]) -> List[str]:
        """Gather string predictions across all processes."""
        import torch.distributed as dist

        if not dist.is_initialized():
            return predictions

        world_size = dist.get_world_size()

        # Gather all predictions to rank 0
        gathered = [None] * world_size
        dist.all_gather_object(gathered, predictions)

        # Flatten the list
        all_predictions = []
        for preds in gathered:
            all_predictions.extend(preds)

        return all_predictions
