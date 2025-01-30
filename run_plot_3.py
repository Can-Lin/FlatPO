import argparse
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '7,8,9'
import random
import pandas as pd
import openpyxl
import wandb
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from metrics import calculate_metric
from modeling_mistral import (
    MistralForCausalLM,
    MistralConfig
)
from tasks import get_task
from trainer import OurTrainer
from utils import *
from sam import SAM
import torch.nn.functional as F
os.environ["TRANSFORMERS_CACHE"] = "/data/zhuyifan/lc/model"


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AutoConfig.register("mistral", MistralConfig)
AutoModelForCausalLM.register(MistralConfig, MistralForCausalLM)


@dataclass
class OurArguments(TrainingArguments):
    # dataset and sampling strategy
    task_name: str = "SST2"  # task name should match the string before Dataset in the Dataset class name. We support the following task_name: SST2, RTE, CB, BoolQ, WSC, WIC, MultiRC, Copa, ReCoRD, SQuAD, DROP

    # Number of examples
    num_train: int = 0  # ICL mode: number of demonstrations; training mode: number of training samples
    num_dev: int = None  # (only enabled with training) number of development samples
    num_eval: int = None  # number of evaluation samples
    num_train_sets: int = None  # how many sets of training samples/demos to sample; if None and train_set_seed is None, then we will sample one set for each evaluation sample
    train_set_seed: int = 0  # designated seed to sample training samples/demos
    result_file: str = None  # file name for saving performance; if None, then use the task name, model name, and config

    # Model loading
    model_name: str = "facebook/opt-125m"  # HuggingFace model name
    load_float16: bool = False  # load model parameters as float16
    load_bfloat16: bool = False  # load model parameters as bfloat16
    load_int8: bool = False  # load model parameters as int8
    max_length: int = 2048  # max length the model can take
    no_auto_device: bool = False  # do not load model by auto device; should turn this on when using FSDP

    # Calibration
    sfc: bool = False  # whether to use SFC calibration
    icl_sfc: bool = False  # whether to use SFC calibration for ICL samples

    template_ver: int = 0  # template. For some tasks (SST2, RTE, Copa), we add template ver=1 as the empty template.

    # Training
    trainer: str = "none"
    ## options
    ## - none: no training -- for zero-shot or in-context learning (ICL)
    ## - regular: regular huggingface trainer -- for fine-tuning
    ## - zo_sgd: zeroth-order SGD (MeZO) training
    ## - zo_conserv: zeroth-order SGD conservative training
    ## - zo_adam: zeroth-order Adam training
    ## - zo_sign_opt: zeroth-order sign sgd training
    ## - forward_grad: forward gradient
    optimizer: str = "adamw"
    ## options
    ## - sgd
    ## - adam
    ## - adamw # this is huggingface default
    only_train_option: bool = True  # whether to only train the option part of the input
    train_as_classification: bool = False  # take the log likelihood of all options and train as classification
    momentum: float = 0.0  # only work for SGD optimizer
    sam_rho: float = 0.0
    # MeZO
    zo_eps: float = 1e-3  # eps in MeZO
    perturbation_mode: str = "two_side"
    q: int = 1  # number of Gaussian samples for zeroth-order trainers
    #sam: bool = False
    baseline: bool = False
    use_algorithm: str = "sam"
    beta: float = 1.0
    gamma: float = 1.0
    # Prefix tuning
    prefix_tuning: bool = False  # whether to use prefix tuning
    num_prefix: int = 5  # number of prefixes to use
    no_reparam: bool = True  # do not use reparameterization trick
    prefix_init_by_real_act: bool = True  # initialize prefix by real activations of random words

    # prompt tuning hyperparameters
    prompt_tuning: bool = False  # whether to use prompt tuning
    num_virtual_tokens: int = 10  # number of prompt tokens to use
    prompt_init_by_real_tokens: bool = False  # whether to sample random tokens from Embedding layer

    # LoRA
    lora: bool = False  # whether to use LoRA
    lora_alpha: int = 16  # alpha in LoRA
    lora_r: int = 8  # r in LoRA

    # Generation
    sampling: bool = False  # whether to use sampling
    temperature: float = 1.0  # temperature for generation
    num_beams: int = 1  # number of beams for generation
    top_k: int = None  # top-k for generation
    top_p: float = 0.95  # top-p for generation
    max_new_tokens: int = 50  # max number of new tokens to generate
    eos_token: str = "\n"  # end of sentence token

    # Saving
    save_model: bool = False  # whether to save the model
    no_eval: bool = False  # whether to skip evaluation
    tag: str = ""  # saving tag

    # Linear probing
    linear_probing: bool = False  # whether to do linear probing
    lp_early_stopping: bool = False  # whether to do early stopping in linear probing
    head_tuning: bool = False  # head tuning: only tune the LM head

    # Untie emb/lm_head weights
    untie_emb: bool = False  # untie the embeddings and LM head

    # Display
    verbose: bool = False  # verbose output

    # Non-diff objective
    non_diff: bool = False  # use non-differentiable objective (only support F1 for SQuAD for now)

    # Auto saving when interrupted
    save_on_interrupt: bool = False  # save model when interrupted (useful for long training)

    clean_model_at_end: bool = True  # remove everthing at the end.

    # sparse gradient pruning
    gradient_sparsity: float = None
    sparse_gradient_resample_steps: int = 1
    sparse_gradient_group: str = "layer"
    """
    Options
    ## - global: global sparsity will assign different sparsity to each layer, based on the pretrained weight magnitude
    ## - layer: each layer has the same sparsity
    """

    # module-wise perturbation
    module_wise_perturbation: bool = False
    perturbed_module_level: str = "transformer-block"
    coordinate_perturbation: bool = True  # If True, will update weight right after the gradient is computed
    """
    Options
    ## - transformer-block: perturb one transformer block at a time
    ## - mlp-attn: perturb one mlp/attention layer at a time
    ## - linear: perturb one linear layer at a time
    """


def parse_args():
    parser = argparse.ArgumentParser()
    parser = HfArgumentParser(OurArguments)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 这个类仅用于存放预测结果和对应的评分（logits或对数概率等）
class Prediction:
    def __init__(self, correct_candidate, predicted_candidate, scores=None):
        """
        correct_candidate: int 或 list(int) ，表示正确答案的下标
        predicted_candidate: int 或 str ，表示预测的答案/文本
        scores: list(float) 或 None ，用于存放对所有候选的打分（可被当做 logits）
        """
        self.correct_candidate = correct_candidate
        self.predicted_candidate = predicted_candidate
        self.scores = scores

class Framework:
    
    def __init__(self, args, task):
        self.args = args
        self.task = task
        self.model, self.tokenizer = self.load_model()
            
    def load_model(self):
        """
        Load HuggingFace models
        """
        with count_time("Loading model with FP%d" % (16 if self.args.load_float16 else 32)):
            free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
            print(free_in_GB)
            config = AutoConfig.from_pretrained(self.args.model_name)
            if self.args.untie_emb:
                # Untie embeddings/LM head
                logger.warn("Untie embeddings and LM head")
                config.tie_word_embeddings = False
            if self.args.head_tuning:
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                # Head tuning
                if "opt" in self.args.model_name.lower():
                    from modeling_opt import OPTForCausalLM
                    model = OPTForCausalLM.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map='auto',
                        torch_dtype=torch_dtype,
                        max_memory={i: f'{free_in_GB - 5}GB' for i in
                                    range(torch.cuda.device_count())},
                    )

                elif "llama" in self.args.model_name.lower():
                    from modeling_llama import LlamaForCausalLMWithHeadTuning
                    model = LlamaForCausalLMWithHeadTuning.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map='auto',
                        torch_dtype=torch_dtype,
                        max_memory={i: f'{free_in_GB - 5}GB' for i in
                                    range(torch.cuda.device_count())},
                    )
                elif "mistral" in self.args.model_name.lower():
                    from modeling_mistral import MistralForCausalLMWithHeadTuning
                    model = MistralForCausalLMWithHeadTuning.from_pretrained(
                        self.args.model_name,
                        config=config,
                        device_map='auto',
                        torch_dtype=torch_dtype,
                        max_memory={i: f'{free_in_GB - 5}GB' for i in
                                    range(torch.cuda.device_count())},
                    )
                else:
                    raise NotImplementedError(f"Head tuning is not supported for {self.args.model_name}")
            elif self.args.no_auto_device:
                # No auto device (use for FSDP)
                model = AutoModelForCausalLM.from_pretrained(self.args.model_name, config=config, )
            else:
                # Auto device loading
                torch_dtype = torch.float32
                if self.args.load_float16:
                    torch_dtype = torch.float16
                elif self.args.load_bfloat16:
                    torch_dtype = torch.bfloat16
                model = AutoModelForCausalLM.from_pretrained(self.args.model_name, config=config, device_map='auto',
                                                             torch_dtype=torch_dtype,
                                                             max_memory={i: f'{free_in_GB - 5}GB' for i in
                                                                         range(torch.cuda.device_count())},
                                                             load_in_8bit=self.args.load_int8, )
            model.eval()

        # Load tokenizer
        #  In mezo, use_fast is set to False. But TypeError will occur when running SQuaD. Setting to be True can fix.
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name, use_fast=True)

        # HF tokenizer bug fix
        if "opt" in self.args.model_name:
            tokenizer.bos_token_id = 0

        if ("llama" in self.args.model_name) or ("mistral" in self.args.model_name.lower()):
            # LLaMA padding token
            tokenizer.pad_token_id = 0  # technically <unk>

        # Prefix tuning/LoRA
        if self.args.prefix_tuning:
            from prefix_tuning import PrefixTuning
            PrefixTuning(model, num_prefix=self.args.num_prefix, reparam=not self.args.no_reparam,
                         float16=self.args.load_float16, init_by_real_act=self.args.prefix_init_by_real_act)
        if self.args.lora:
            from lora import LoRA
            #print("lora_parameters_before:")
            #print(len(list(model.parameters())))
            LoRA(model, r=self.args.lora_r, alpha=self.args.lora_alpha, float16=self.args.load_float16)
            #print("lora_parameters_after:")
            #print(len(list(model.parameters())))
        if self.args.prompt_tuning:
            from prompt_tuning import PromptTuning
            print("Adding Prompt Tuning to model...")
            PromptTuning(
                model,
                num_virtual_tokens=self.args.num_virtual_tokens,
                init_by_real_tokens=self.args.prompt_init_by_real_tokens,
                hide_virtual_token_logits=True,  # a workaround for the other loss/prediction functions
            )
            print("Total/Trainable number of parameters: {}/{}".format(
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            ))

        if self.args.head_tuning:
            if model.config.model_type in ["opt", "llama", "mistral"]:
                head_name = "lm_head" if self.args.untie_emb else "embed_tokens"
            else:
                raise NotImplementedError
            for n, p in model.named_parameters():
                if head_name not in n:
                    p.requires_grad = False
                else:
                    logger.info(f"Only tuning {n}")
        
        return model, tokenizer

    def forward(self, input_ids, option_len=None, generation=False):
        """
        Given input_ids and the length of the option, return the log-likelihood of each token in the option.
        For generation tasks, return the generated text.
        This function is only for inference
        """
        # 包装为 batch_size=1 的张量
        input_ids = torch.tensor([input_ids]).to(self.model.device)

        if generation:
            args = self.args
            # Autoregressive generation
            outputs = self.model.generate(
                input_ids,
                do_sample=args.sampling,
                temperature=args.temperature,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=min(args.max_new_tokens, args.max_length - input_ids.size(1)),
                num_return_sequences=1,
                eos_token_id=[
                    self.tokenizer.encode(args.eos_token, add_special_tokens=False)[-1],
                    self.tokenizer.eos_token_id
                ],
            )
            # For generation, directly return the text output
            output_text = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True).strip()
            return output_text
        else:
            with torch.inference_mode():
                self.model.eval()
                logits = self.model(input_ids=input_ids).logits
            # 下面的 labels、log_probs 是用于计算选项 token 的 log-likelihood
            labels = input_ids[0, 1:]
            logits = logits[0, :-1]
            log_probs = F.log_softmax(logits, dim=-1)

            selected_log_probs = log_probs[torch.arange(len(labels)).to(labels.device), labels]
            selected_log_probs = selected_log_probs.cpu().detach()
            # 仅保留选项（candidate）的部分
            return selected_log_probs[-option_len:]

    def one_step_pred(self, train_samples, eval_sample, verbose=False):
        """
        Return the prediction on the eval sample. In ICL, use train_samples as demonstrations
        """
        verbose = verbose or self.args.verbose

        # 对样本进行编码，若是分类/多选，会对所有候选都做encode
        # encode_prompt 是外部依赖函数，需要你自行定义
        encoded_candidates, option_lens = encode_prompt(
            self.task,
            self.task.get_template(template_version=self.args.template_ver),
            train_samples,
            eval_sample,
            self.tokenizer,
            max_length=self.args.max_length,
            generation=self.task.generation,
            max_new_tokens=self.args.max_new_tokens
        )

        # 如果使用了 surface form competition（SFC），再多encode一遍
        if self.args.sfc or self.args.icl_sfc:
            sfc_encoded_candidates, sfc_option_lens = encode_prompt(
                self.task,
                self.task.get_template(template_version=self.args.template_ver),
                train_samples,
                eval_sample,
                self.tokenizer,
                max_length=self.args.max_length,
                sfc=self.args.sfc,
                icl_sfc=self.args.icl_sfc,
                generation=self.task.generation,
                max_new_tokens=self.args.max_new_tokens
            )

        outputs = []
        # 如果是生成任务，直接调用 forward 做生成
        if self.task.generation:
            output_text = self.forward(encoded_candidates[0], generation=True)
            return Prediction(
                correct_candidate=eval_sample.correct_candidate,
                predicted_candidate=output_text
            )
        else:
            # 否则是多选/分类
            for candidate_id, encoded_candidate in enumerate(encoded_candidates):
                selected_log_probs = self.forward(encoded_candidate, option_len=option_lens[candidate_id])

                if verbose:
                    # logger.info(...) 此处省略日志输出，需要时可自行补充
                    pass

                if self.args.sfc or self.args.icl_sfc:
                    sfc_selected_log_probs = self.forward(
                        sfc_encoded_candidates[candidate_id],
                        option_len=sfc_option_lens[candidate_id]
                    )
                else:
                    sfc_selected_log_probs = None

                outputs.append({
                    "log_probs": selected_log_probs,
                    "sfc_log_probs": sfc_selected_log_probs
                })

            # 根据是否启用 sfc 算分
            if self.args.sfc or self.args.icl_sfc:
                # Calibrated probabilities
                # log p(candidate | input) = log p_lm(candidate | input) - log p_lm(candidate | sfc prompt)
                scores = [
                    x['log_probs'].sum().item() - x['sfc_log_probs'].sum().item()
                    for x in outputs
                ]
            else:
                # 默认，用长度平均的 log_prob 作为 candidate 分数
                scores = [
                    x['log_probs'].mean().item()
                    for x in outputs
                ]

            if isinstance(eval_sample.correct_candidate, list):
                # 如果有多个正确答案
                correct_candidate_id = [
                    eval_sample.candidates.index(c)
                    for c in eval_sample.correct_candidate
                ]
            else:
                correct_candidate_id = eval_sample.candidates.index(eval_sample.correct_candidate)

            # 选出得分最高的 candidate
            predicted_candidate_id = int(np.argmax(scores))

            # 在这里把 scores 一并返回
            return Prediction(
                correct_candidate=correct_candidate_id,
                predicted_candidate=predicted_candidate_id,
                scores=scores
            )

    def evaluate(self, train_samples, eval_samples, one_train_set_per_eval_sample=False, description=None):
        """
        Evaluate function.
        这里主要新增对 cross-entropy loss 的计算逻辑
        """
        if one_train_set_per_eval_sample:
            logger.info(f"There are {len(eval_samples)} validation samples and one train set per eval sample")
        else:
            logger.info(f"There are {len(train_samples)} training samples and {len(eval_samples)} validation samples")

        # Prediction loop
        predictions = []

        # 准备 CrossEntropyLoss
        loss_fct = CrossEntropyLoss()
        total_loss = 0.0
        count_loss = 0

        for eval_id, eval_sample in enumerate(tqdm(eval_samples, desc=description)):
            # 根据是否 one_train_set_per_eval_sample，选择传入的 demonstration
            prediction = self.one_step_pred(
                train_samples[eval_id] if one_train_set_per_eval_sample else train_samples,
                eval_sample,
                verbose=False
            )
            predictions.append(prediction)

            # 如果是生成任务，就不计算候选交叉熵
            if not self.task.generation:
                # prediction.scores: [num_candidates]
                # prediction.correct_candidate: 正确候选下标 (int) 或 list(多标签)

                if isinstance(prediction.correct_candidate, list):
                    # 多正确答案情况，需要你自行定制逻辑
                    # 这里演示：简单跳过
                    continue
                else:
                    # 把 list 的 scores 转成 tensor，当做 logits
                    scores_tensor = torch.tensor(prediction.scores, dtype=torch.float).unsqueeze(0)
                    # 正确 label
                    label_tensor = torch.tensor([prediction.correct_candidate], dtype=torch.long)
                    # 计算交叉熵
                    sample_loss = loss_fct(scores_tensor, label_tensor)
                    total_loss += sample_loss.item()
                    count_loss += 1

        # 计算平均 Loss
        avg_loss = total_loss / count_loss if count_loss > 0 else 0.0

        # 计算任务指标（accuracy 等），calculate_metric 需要你自行定义
        metric_name = getattr(self.task, "metric_name", "accuracy")
        metrics = {metric_name: calculate_metric(predictions, metric_name)}

        # 把 loss 放进 metrics
        metrics["loss"] = avg_loss
        return metrics


    def train(self, train_samples, dev_samples, eval_samples):
        """
        Training function
        if self.num_dev is not None, eval_samples are dev_samples
        """
        logger.info(f"Eval sample length is {len(eval_samples)}")
        # Set tokenizer to left padding (so that all the options are right aligned)
        self.tokenizer.padding_side = "left"

        class HFDataset(Dataset):

            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        def _convert(samples):
            """
            Convert samples to HF-compatible dataset
            """
            data = []
            for sample in samples:
                encoded_candidates, option_lens = encode_prompt(self.task, self.task.get_template(
                    template_version=self.args.template_ver), [], sample,
                                                                self.tokenizer, max_length=self.args.max_length,
                                                                generation=self.task.generation,
                                                                generation_with_gold=True,
                                                                max_new_tokens=self.args.max_new_tokens)
                if self.task.generation:
                    correct_candidate_id = 0
                elif isinstance(sample.correct_candidate, list):
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate[0])
                else:
                    correct_candidate_id = sample.candidates.index(sample.correct_candidate)

                if self.args.non_diff:
                    # For non-differentiable objective, there is no teacher forcing thus the 
                    # current answer part is removed
                    encoded_candidates[correct_candidate_id] = encoded_candidates[correct_candidate_id][
                                                               :-option_lens[correct_candidate_id]]

                if self.args.train_as_classification:
                    # For classification, we provide the label as the correct candidate id
                    data.append([{"input_ids": encoded_candidates[_i], "labels": correct_candidate_id,
                                  "option_len": option_lens[_i], "num_options": len(sample.candidates)} for _i in
                                 range(len(encoded_candidates))])
                elif self.args.only_train_option:
                    # Otherwise, it is just LM-style teacher forcing
                    if self.args.non_diff:
                        # For non-differentiable objective, we need to provide the gold answer to calculate F1/acc
                        data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                     "labels": encoded_candidates[correct_candidate_id],
                                     "option_len": option_lens[correct_candidate_id], "gold": sample.correct_candidate})
                    else:
                        data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                     "labels": encoded_candidates[correct_candidate_id],
                                     "option_len": option_lens[correct_candidate_id]})
                else:
                    data.append({"input_ids": encoded_candidates[correct_candidate_id],
                                 "labels": encoded_candidates[correct_candidate_id]})
            return data

        with count_time("Tokenizing training samples"):
            train_dataset = HFDataset(_convert(train_samples))
            eval_dataset = HFDataset(_convert(eval_samples))
            dev_dataset = HFDataset(_convert(dev_samples))

        if self.args.only_train_option and not self.args.non_diff:
            # If --only_train_option and not with a non-differentiable objective, we wrap the forward function
            self.model.original_forward = self.model.forward
            self.model.forward = forward_wrap_with_option_len.__get__(self.model, type(self.model))

        if self.args.non_diff:
            collator = NondiffCollator
        else:
            collator = DataCollatorForTokenClassification

        if self.args.gradient_sparsity is not None:
            logger.info(
                f"[Sparse gradient] sparsity is {self.args.gradient_sparsity}, resampling per {self.args.sparse_gradient_resample_steps} steps"
            )

            if self.args.sparse_gradient_group == "global":
                logger.info(f"[Sparse gradient] global-ratio random pruning is enabled, "
                            f"sparsity of each layer is computed based on the pretrained weight magnitude.")
            elif self.args.sparse_gradient_group == "layer":
                logger.info(f"[Sparse gradient] layer-wise random pruning is enabled, "
                            f"sparsity of each layer is the same.")
            else:
                raise NotImplementedError(f"Unknown sparse gradient group: {self.args.sparse_gradient_group}")

        perturb_module_regex = None
        if self.args.module_wise_perturbation:
            if "opt" in self.args.model_name:
                assert self.args.perturbed_module_level in OPT_PERTURBATION_LEVEL_TO_REGEX.keys(), f"Unknown perturbed module group {self.args.perturbed_module_level}"
                perturb_module_regex = OPT_PERTURBATION_LEVEL_TO_REGEX[self.args.perturbed_module_level]
            else:
                raise NotImplementedError(f"Unimplemented model {self.args.model_name} for module-wise perturbation")

        trainer = OurTrainer(model=self.model,
                             args=self.args,
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             tokenizer=self.tokenizer,
                             data_collator=DataCollatorWithPaddingAndNesting(self.tokenizer,
                                                                             pad_to_multiple_of=8) if self.args.train_as_classification else collator(
                                 self.tokenizer, pad_to_multiple_of=8),
                             eval_samples=eval_samples,
                             dev_samples=dev_samples,
                             evaluate_func=self.evaluate,
                             perturb_module_regex=perturb_module_regex,
                             )
        if self.args.save_on_interrupt:
            trainer.add_callback(SIGUSR1Callback())

        # Resume training from a last checkpoint
        last_checkpoint = None
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(self.args.output_dir) and not self.args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.args.output_dir)
        if last_checkpoint is not None and self.args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
        if self.args.resume_from_checkpoint is not None:
            last_checkpoint = self.args.resume_from_checkpoint

        # This calls the trainer._inner_training_loop()
        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Explicitly save the model
        if self.args.save_model:
            logger.info("Save model..")
            trainer.save_model()

        # FSDP compatibility
        self.model = trainer.model

        # Reset the forward function for evaluation
        if self.args.only_train_option and not self.args.non_diff:
            if type(self.model) == FSDP:
                logger.info("This is an FSDP model now. Be careful when assigning back the original forward function")
                self.model._fsdp_wrapped_module.forward = self.model._fsdp_wrapped_module.original_forward
            else:
                self.model.forward = self.model.original_forward

    def delete_checkpoints(self):
        import shutil
        print(f"\nWARNING: Removing everything at end: {self.args.output_dir}")
        deleted_folders = [folder for folder in os.listdir(self.args.output_dir)
                           if os.path.isdir(os.path.join(self.args.output_dir, folder))
                           and folder.startswith("checkpoint-")]
        for f in deleted_folders:
            shutil.rmtree(os.path.join(self.args.output_dir, f))
        print(f"deleted folders: ", deleted_folders)


def result_file_tag(args):
    """
    Get the result file tag
    """
    save_model_name = args.model_name.split("/")[-1]
    sfc_tag = "-sfc" if args.sfc else ""
    icl_sfc_tag = "-icl_sfc" if args.icl_sfc else ""
    sample_eval_tag = "-sampleeval%d" % args.num_eval if args.num_eval is not None else ""
    sample_train_tag = "-ntrain%d" % args.num_train if args.num_train > 0 else ""
    sample_dev_tag = "-ndev%d" % args.num_dev if args.num_dev is not None else ""
    customized_tag = f"-{args.tag}" if len(args.tag) > 0 else ""
    return f"{args.task_name}-{save_model_name}" + sfc_tag + icl_sfc_tag + sample_eval_tag + sample_train_tag + sample_dev_tag + customized_tag


def main():
    # 解析命令行参数
    args = parse_args()
    args.trainer = "none"
    args.num_dev = None
    args.num_train = 0
    # 根据参数设置模式
    if args.prefix_tuning:
        args.mode = "prefix"
    elif args.lora:
        args.mode = "lora"
    elif args.prompt_tuning:
        args.mode = "prompt"
    else:
        args.mode = "ft"
    # 构建标签
    args.tag = f"{args.trainer}-{args.task_name}-{args.template_ver}-{args.model_name.split('/')[-1]}-OPTIM_{args.mode}-STEP{args.max_steps}-{args.optimizer}-LR{args.learning_rate}-{args.lr_scheduler_type}-ZOEPS{args.zo_eps}-Q{args.q}"
    args.tag = "momen" + args.tag if args.momentum > 0 else args.tag
    args.tag = f"sparse_grad-{args.gradient_sparsity}-{args.sparse_gradient_group}-{args.sparse_gradient_resample_steps}-" + args.tag if args.gradient_sparsity is not None else args.tag
    args.tag = f"module_perturb-{args.perturbed_module_level}-" + args.tag if args.module_wise_perturbation else args.tag
    # 设置运行名称和输出目录
    args.run_name = args.tag
    args.output_dir = f"result/{args.tag}"
    args.result_file = f"result/{args.tag}/results.json"
    # 创建输出目录和日志目录
    os.makedirs(args.output_dir, exist_ok=True)
    args.logging_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(args.logging_dir, exist_ok=True)
    # 初始化wandb
    wandb.init(project='zo-bench', name=args.tag, config=args)
    # 设置随机种子
    set_seed(args.seed)
    # 获取数据集
    task = get_task(args.task_name)

    # This function samples both training and validation samples. The validation (dev) samples are also stored in "train_sets"
    # Later the train_samples and dev_samples are separated
    # 抽样训练集和验证集
    train_sets = task.sample_train_sets(num_train=args.num_train, num_dev=args.num_dev, num_eval=args.num_eval,
                                        num_train_sets=args.num_train_sets, seed=args.train_set_seed)

    # Initialize trainer and load model
    # 初始化训练框架和加载模型
    #framework_after = Framework(args, task)
    #SGD
    #learnt_model = '/data/zhuyifan/lc/ZO-LLM/zo-bench/model_state_dict.pth'
    #SAM
    #learnt_model = '/data/zhuyifan/lc/ZO-LLM/zo-bench/result/momenregular-SST2-0-opt-1.3b-OPTIM_lora-STEP500-sgd-LR0.0001-cosine-ZOEPS0.001-Q1/model_state_dict.pth'
    #OURS
    #learnt_model = '/data/zhuyifan/lc/ZO-LLM/zo-bench/result/momenregular-SST2-0-opt-1.3b-OPTIM_lora-STEP500-sgd-LR0.0001-cosine-ZOEPS0.001-Q1/model_state_dict111.pth'
    #framework_after.model.load_state_dict(torch.load(learnt_model))
    #theta_ast = Params2Vec(framework_after.model.parameters())
    
        
    #theta = Params2Vec(framework_before.model.parameters())
    #Vec2Params(0.5 * theta + 0.5 * theta_ast, framework_before.model.parameters())
    # ZO-Bench Added
    # We add these parameters to evaluate the model during the training.
    # These two parameters will be used in the training loop
    # args.task = task
    # args.framework = framework
    # 如果设置了训练集种子或训练集数量
    if args.train_set_seed is not None or args.num_train_sets is not None:
        #print(11111111111111)
        # Training goes to this way

        # Eval samples share one (or multiple) training set(s)
        # 遍历每个训练集
        for train_set_id, train_samples in enumerate(train_sets):
            # 设置训练集种子
            train_set_seed = train_set_id if args.train_set_seed is None else args.train_set_seed

            # Sample eval samples
            # 抽样评估样本
            if args.num_eval is not None:
                eval_samples = task.sample_subset(data_split="valid", seed=train_set_seed, num=args.num_eval)
            else:
                eval_samples = task.valid_samples

            if args.trainer != "none":
                # Here the training samples are seperated
                if args.num_dev is not None:
                    # Dev samples
                    # assert args.num_dev + args.num_train <= len(train_samples), f"num_dev({args.num_dev})+num_train({args.num_train}) is more than actual num of training samples ({len(train_samples)})."
                    dev_samples = train_samples[-args.num_dev:]
                    train_samples = train_samples[:-args.num_dev]
                    logger.info("Dev samples: %d" % len(dev_samples))
                    logger.info("Train samples: %d" % len(train_samples))
                else:
                    dev_samples = None
                    logger.info("Train samples: %d" % len(train_samples))
                    logger.info("No dev samples")

                args.dev_samples = dev_samples
                args.eval_samples = eval_samples
                
                # Training
                framework.train(train_samples, dev_samples if dev_samples is not None else eval_samples, eval_samples)
                # 保存模型
                torch.save(framework.model.state_dict(), os.path.join(args.output_dir, 'model_state_dict.pth'))
                # 如果不禁止评估
                if not args.no_eval:  # This is True
                    #print(22222222222222)
                    metrics = framework.evaluate([], eval_samples, description="Evaluating on the Test Set")
                    _keys = list(metrics.keys())
                    for m in _keys:
                        metrics["test_" + m] = metrics[m]
                    if dev_samples is not None:
                        dev_metrics = framework.evaluate(
                            [], dev_samples, description="Evaluating on the Validation Set"
                        )
                        _keys = list(dev_metrics.keys())
                        for m in _keys:
                            metrics["val_" + m] = dev_metrics[m]
            else:
                assert args.num_dev is None
                # Zero-shot / in-context learning
                #print(3333333333)
                # 初始化一个空的列表，用于存储每个 alpha 值对应的 metrics
                framework_after = Framework(args, task)
                #SGD
                #learnt_model = '/data/zhuyifan/lc/ZO-LLM/zo-bench/model_state_dict_sgd.pth'
                #SAM
                learnt_model = '/data/zhuyifan/lc/ZO-LLM/zo-bench/model_state_dict_sam.pth'
                #OURS
                #learnt_model = '/data/zhuyifan/lc/ZO-LLM/zo-bench/model_state_dict_our.pth'
                framework_after.model.load_state_dict(torch.load(learnt_model))
                
                framework_before = Framework(args, task)
                framework_before.model.load_state_dict(torch.load(learnt_model))
                framework_before.model.eval()  # 评估模式
                framework_after.model.eval()  # 评估模式
                # 1. 设定参数
                before_params = [p.clone().detach() for p in framework_before.model.parameters()]
                after_params  = [p.clone().detach() for p in framework_after.model.parameters()]
                reference_params = before_params
                direction1 = []
                direction2 = []

                for ref_p, after_p in zip(before_params, after_params):
                    # 方向1：随机高斯向量
                    d1 = torch.randn_like(ref_p)
                    #d1 = after_p - ref_p
                    # 方向2：随机高斯向量
                    d2 = torch.randn_like(ref_p)

                    direction1.append(d1)
                    direction2.append(d2)
                for i in range(len(direction1)):
                    direction1[i] = direction1[i] / (direction1[i].norm() + 1e-8)
                    direction2[i] = direction2[i] / (direction2[i].norm() + 1e-8)
      

                # 后面 alpha/beta 网格计算同之前一致
                alpha_list = np.linspace(-1.0, 1.0, 21)
                beta_list  = np.linspace(-1.0, 1.0, 21)
                #alpha_list = np.arange(-1.0, 1.0 + 0.001, 0.5)
                #beta_list  = np.arange(-1.0, 1.0 + 0.001, 0.5)
                df = pd.DataFrame(columns=["alpha", "beta", "loss"])

                for alpha in alpha_list:
                    for beta in beta_list:
                        with torch.no_grad():
                            for p_model, p_ref, d1, d2 in zip(framework_before.model.parameters(),
                                                            reference_params,
                                                            direction1,
                                                            direction2):
                                p_model.data.copy_(p_ref + alpha * d1 + beta * d2)

                        metrics = framework_before.evaluate(train_samples, eval_samples)
                        current_loss = metrics['loss']
                        
                        df.loc[len(df)] = [alpha, beta, current_loss]
                        print(f"alpha={alpha}, beta={beta}, metrics={metrics}")
                # 3. 导出到 Excel
                df.to_excel("loss_landscape_sam_prompt.xlsx", index=False)    
            logger.info(metrics)
            wandb.log(metrics)

            if not args.no_eval:
                logger.info("===== Train set %d =====" % train_set_seed)
                logger.info(metrics)
                wandb.log(metrics)
                if args.local_rank <= 0:
                    write_metrics_to_file(metrics, "result/" + result_file_tag(
                        args) + f"-trainset{train_set_id}.json" if args.result_file is None else args.result_file)
            if args.trainer != "none" and args.clean_model_at_end:
                framework.delete_checkpoints()

    else:
        # For each eval sample, there is a training set. no training is allowed
        # This is for in-context learning (ICL)
        # 每个评估样本都有一个训练集，不允许训练
        # 这是为了上下文学习（ICL）
        assert args.trainer == "none"
        # 抽样评估样本
        if args.num_eval is not None:
            eval_samples = task.sample_subset(data_split="valid", seed=0, num=args.num_eval)
        else:
            eval_samples = task.valid_samples
        # 评估模型
        metrics = framework.evaluate(train_sets, eval_samples, one_train_set_per_eval_sample=True)
        logger.info(metrics)
        wandb.log(metrics)
        # 如果是主进程，写入指标到文件
        if args.local_rank <= 0:
            write_metrics_to_file(metrics, "result/" + result_file_tag(
                args) + "-onetrainpereval.json" if args.result_file is None else args.result_file)


if __name__ == "__main__":
    main()
