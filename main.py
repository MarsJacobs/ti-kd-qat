
# This code is implemented base on "TernaryBERT: Distillation-aware Ultra-low Bit BERT" (Zhang et al, EMNLP2020)
# https://arxiv.org/abs/2009.12812

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import pickle
import copy
import collections
import math

import numpy as np
import numpy
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset

from torch.nn import CrossEntropyLoss, MSELoss, CosineEmbeddingLoss

from transformer import BertForSequenceClassification,WEIGHTS_NAME, CONFIG_NAME
from transformer.modeling_quant import BertForSequenceClassification as QuantBertForSequenceClassification
from transformer import BertTokenizer
from transformer import BertAdam
from transformer import BertConfig
from utils_glue import *

import numpy as np

import torch.nn.functional as F
import time

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0 
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_seq_lengths = torch.tensor([f.seq_length for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_label_ids, all_seq_lengths)
    return tensor_data, all_label_ids

def do_eval(model, task_name, eval_dataloader,
            device, output_mode, eval_labels, num_labels, teacher_model=None):
    eval_loss = 0
    nb_eval_steps = 0
    preds = []

    for batch_ in eval_dataloader:
        batch_ = tuple(t.to(device) for t in batch_)
        
        with torch.no_grad():
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch_

            if teacher_model is not None:
                teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_values = teacher_model(input_ids, segment_ids, input_mask)
                logits, student_atts, student_reps, student_probs, student_values = model(input_ids, segment_ids, input_mask, teacher_outputs=(teacher_probs, teacher_values, teacher_reps, teacher_logits, teacher_atts))
            else:
                logits, _, _, _, _ = model(input_ids, segment_ids, input_mask)
        
        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, eval_labels.numpy())
    result['eval_loss'] = eval_loss
    return result

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return torch.sum((- targets_prob * student_likelihood), dim=-1).mean()

def main():

    # ================================================================================  #
    # ArgParse
    # ================================================================================ #
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default='data',
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_dir",
                        default='models',
                        type=str,
                        help="The model dir.")
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        help="The models directory.")
    parser.add_argument("--task_name",
                        default='sst-2',
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--save_fp_model',
                        action='store_true',
                        help="Whether to save fp32 model")
                        
    parser.add_argument('--save_quantized_model',
                        default=False, type=str2bool,
                        help="Whether to save quantized model")
    
    parser.add_argument("--input_bits",
                        default=8,
                        type=int,
                        help="Quantization bits for activation.")
    
    parser.add_argument("--tc_top_k",
                        default=3,
                        type=int,
                        help="Top-K Coverage")

    parser.add_argument("--gpus",
                        default=1,
                        type=int,
                        help="Number of GPUs to use")
    parser.add_argument("--clip_val",
                        default=2.5,
                        type=float,
                        help="Initial clip value.")
    
    parser.add_argument('--qk_FP',
                        default=False, type=str2bool,
                        )
    
    parser.add_argument('--qkv_FP',
                        default=False, type=str2bool,
                        )
    
    parser.add_argument('--neptune',
                        default=True, type=str2bool,
                        help="neptune logging option")
    
    #MSKIM Quantization Range Option
    parser.add_argument('--quantize',
                        default =True, type=str2bool,
                        help="Whether to quantize student model")

    parser.add_argument('--ffn_1',
                        default =True, type=str2bool,
                        help="Whether to quantize Feed Forward Network")
    
    parser.add_argument('--ffn_2',
                        default =True, type=str2bool,
                        help="Whether to quantize Feed Forward Network")
    
    parser.add_argument('--qkv',
                        default =True, type=str2bool,
                        help="Whether to quantize Query, Key, Value Mapping Weight Matrix")
    
    parser.add_argument('--emb',
                        default =True, type=str2bool,
                        help="Whether to quantize Embedding Layer")

    parser.add_argument('--cls',
                        default =True, type=str2bool,
                        help="Whether to quantize Classifier Dense Layer")
    
    parser.add_argument('--aug_train',
                        default =False, type=str2bool,
                        help="Whether to use augmented data or not")

    parser.add_argument('--clipping',
                        default =False, type=str2bool,
                        help="Whether to use FP Weight Clipping")
    
    
    parser.add_argument("--mean_scale",
                        default=0.7,
                        type=float,
                        help="Ternary Clipping Value Scale Value")
    
    parser.add_argument("--exp_name",
                        default="",
                        type=str,
                        help="Output Directory Name")
    
    parser.add_argument("--training_type",
                        default="qat_normal",
                        type=str,
                        help="QAT Method")

    parser.add_argument("--aug_N",
                        default=30,
                        type=int,
                        help="Data Augmentation N Number")

    parser.add_argument('--pred_distill',
                        default =False, type=str2bool,
                        help="prediction distill option")

    parser.add_argument('--attn_distill',
                        default =True, type=str2bool,
                        help="attention Score Distill Option")

    parser.add_argument('--rep_distill',
                        default =True, type=str2bool,
                        help="Transformer Layer output Distill Option")

    parser.add_argument('--output_distill',
                        default =False, type=str2bool,
                        help="Context Value Distill Option")
        
    parser.add_argument('--gt_loss',
                        default =False, type=str2bool,
                        help="Ground Truth Option")

    # Teacher Intervention Options
    parser.add_argument('--teacher_attnmap',
                        default =False, type=str2bool,
                        help="Teacher Intervention Option (TI-M)")
    parser.add_argument('--teacher_output',
                        default =False, type=str2bool,
                        help="Teacher Intervention Option (TI-O)")
    parser.add_argument('--teacher_gradual',
                        default =False, type=str2bool,
                        help="Teacher Intervention Option (TI-G)")
    
    parser.add_argument('--teacher_stochastic',
                        default =False, type=str2bool,
                        help="Teacher Intervention Option (Stochastic Mixed)")
    
    parser.add_argument('--teacher_inverted',
                        default =False, type=str2bool,
                        help="Teacher Intervention Option (Stochastic Mixed)")

    parser.add_argument('--teacher_context',
                        default =False, type=str2bool,
                        help="Teacher Intervention Option (Context)")
    
    parser.add_argument('--step1_option',
                        default ="GRAD", type=str,
                        help="Teacher Intervention Step-1 Option (For step-2 model init)")
    
    parser.add_argument('--bert',
                        default ="base", type=str,
    )

    args = parser.parse_args() 
    
    # ================================================================================  #
    # Logging setup
    # ================================================================================ #
    run = None

    # Use Neptune for logging
    if args.neptune:
        import neptune.new as neptune
        run = neptune.init_run(project='niceball0827/' + args.task_name.upper(),
                    api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLC\
                    JhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0YjM\
                    0ZTYwMi1kNjQwLTQ4NGYtOTYxMy03Mjc5ZmVkMzY2YTgifQ==')
        
        # run = neptune.init(project='Neptune_ID/ProjectName',
        #             api_token='Neptune_API_Token')

    # ================================================================================  #
    # Load Directory
    # ================================================================================ #
    
    # Exp Name
    exp_name = args.exp_name 

    exp_name += f"_{args.bert}"

    if args.training_type == "qat_step1":
        if args.teacher_attnmap:
            exp_name += f"_MI"
        if args.teacher_context:
            exp_name += f"_CI"
        if args.teacher_output:
            exp_name += f"_OI"
        if args.teacher_gradual:
            exp_name += f"_GRAD"
        if args.teacher_inverted:
            exp_name += f"_INVERTED"
        if args.teacher_stochastic:
            exp_name += f"_STOCHASTIC"
    
    else:
        if args.gt_loss:
            exp_name += "_G"
        if args.attn_distill:
            exp_name += "_S"
        if args.rep_distill:
            exp_name += "_R"
        if args.output_distill:
            exp_name += "_O"
        exp_name += f"_{args.seed}"            
    

    if args.training_type == "qat_step2":
        exp_name += f"_{args.step1_option}"
    
    args.exp_name = exp_name
    
    if args.aug_train:
        logger.info(f'DA QAT')        
        
    logger.info(f'EXP SET: {exp_name}')
    logger.info(f'TASK: {args.task_name}')
    logger.info(f"SIZE: {args.bert}")
    logger.info(f"SEED: {args.seed}")
    logger.info(f'EPOCH: {args.num_train_epochs}')
    
    # GLUE Dataset Setting
    task_name = args.task_name.lower()
    data_dir = os.path.join(args.data_dir,task_name)
    processed_data_dir = os.path.join(data_dir,'preprocessed')
    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    
    # BERT Large Option
    if args.bert == "large":
        args.model_dir = os.path.join(args.model_dir, "BERT_large")
        args.output_dir = os.path.join(args.output_dir, "BERT_large")
    
    if args.bert == "tiny-4l":
        args.model_dir = os.path.join(args.model_dir, "BERT_Tiny_4l")
        args.output_dir = os.path.join(args.output_dir, "BERT_Tiny_4l")
    
    if args.bert == "tiny-6l":
        args.model_dir = os.path.join(args.model_dir, "BERT_Tiny_6l")
        args.output_dir = os.path.join(args.output_dir, "BERT_Tiny_6l")
    
    # Model Save Directory
    output_dir = os.path.join(args.output_dir,task_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    if args.save_quantized_model:
        output_quant_dir = os.path.join(output_dir, 'exploration')
        if not os.path.exists(output_quant_dir):
            os.mkdir(output_quant_dir)

        if not os.path.exists(output_quant_dir):
            os.makedirs(output_quant_dir)
        
        output_quant_dir = os.path.join(output_quant_dir, args.exp_name)
        if not os.path.exists(output_quant_dir):
            os.makedirs(output_quant_dir)

    # ================================================================================  #
    # Load Pths
    # ================================================================================ #
    # Student Model Pretrained FIle    
    
    if args.training_type == "qat_normal":
        args.student_model = os.path.join(args.model_dir,task_name) 
    elif args.training_type == "qat_step1":
        args.student_model = os.path.join(args.model_dir, task_name) 
    elif args.training_type == "qat_step2":        
        args.student_model = os.path.join(args.output_dir, task_name, "exploration", f"TI_step1_{args.bert}_{args.step1_option}")
    else:
        raise ValueError("Choose Training Type {downsteam, qat_normal, qat_step1, qat_step2, qat_step3, gradual}")

    # Teacher Model Pretrained FIle    
    args.teacher_model = os.path.join(args.model_dir,task_name)
    
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification"
    }

    default_params = {
        "cola": {"max_seq_length": 64,"batch_size":16,"eval_step": 2000 if args.aug_train else 50}, # No Aug : 50 Aug : 400
        "mnli": {"max_seq_length": 128,"batch_size":32,"eval_step":8000},
        "mrpc": {"max_seq_length": 128,"batch_size":32,"eval_step":1000 if args.aug_train else 50},
        "sst-2": {"max_seq_length": 64,"batch_size":32,"eval_step":100},
        "sts-b": {"max_seq_length": 128,"batch_size":32,"eval_step":2000 if args.aug_train else 100},
        "qqp": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "qnli": {"max_seq_length": 128,"batch_size":32,"eval_step":1000},
        "rte": {"max_seq_length": 128,"batch_size":32,"eval_step":1000 if args.aug_train else 20}
    }

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    # ================================================================================  #
    # prepare devices
    # ================================================================================ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = args.gpus
    
    # ================================================================================  #
    # prepare seed
    # ================================================================================ #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
 
    if task_name in default_params:
        args.batch_size = default_params[task_name]["batch_size"]
        if n_gpu > 0:
            args.batch_size = int(args.batch_size*n_gpu)
        args.max_seq_length = default_params[task_name]["max_seq_length"]
        args.eval_step = default_params[task_name]["eval_step"]
    
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # ================================================================================  #
    # Load Vocab FIle -> Tokenization 
    # ================================================================================ #
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model, do_lower_case=True)
    
    # ================================================================================  #
    # Dataset Setup (with DA)
    # ================================================================================ #
    if args.aug_train: # Data Augmentation
        try:
            train_file = os.path.join(processed_data_dir,f'aug_data_{args.aug_N}.pkl')
            train_features = pickle.load(open(train_file,'rb'))
        except:
            train_examples = processor.get_aug_examples(data_dir, args.aug_N)
            train_features = convert_examples_to_features(train_examples, label_list,
                                            args.max_seq_length, tokenizer, output_mode)
            with open(train_file, 'wb') as f:
                pickle.dump(train_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        try:
            train_file = os.path.join(processed_data_dir,'data.pkl')
            train_features = pickle.load(open(train_file,'rb'))
            
        except:
            train_examples = processor.get_train_examples(data_dir)
            train_features = convert_examples_to_features(train_examples, label_list,
                                            args.max_seq_length, tokenizer, output_mode)
            
            with open(train_file, 'wb') as f:
                pickle.dump(train_features, f, protocol=pickle.HIGHEST_PROTOCOL)

    num_train_epochs = args.num_train_epochs 
    num_train_optimization_steps = math.ceil(len(train_features) / args.batch_size) * num_train_epochs
    
    # TI Step-2 Iteration Number Setting
    if "tiny-4l" in args.bert or task_name == "cola":
        ti_step_1_total_step = 120
    else:
        ti_step_1_total_step = 60

    if args.training_type == "qat_step1": 
        args.eval_step = 10
        num_train_optimization_steps = ti_step_1_total_step 

    # We keep total two-step QAT iteration number identical to baseline TernaryBERT QAT setting
    # Total Iteration Step = N
    # TI step-1 iteration step = s
    # TI step-2 iteration step = N - s

    if args.training_type == "qat_step2" : 
        num_train_optimization_steps = num_train_optimization_steps - ti_step_1_total_step  
    
    train_data, _ = get_tensor_data(output_mode, train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
    
    # Dev Data load
    try:
        dev_file = train_file = os.path.join(processed_data_dir,'dev.pkl')
        eval_features = pickle.load(open(dev_file,'rb'))
    except:
        eval_examples = processor.get_dev_examples(data_dir)
        eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
        with open(dev_file, 'wb') as f:
                pickle.dump(eval_features, f, protocol=pickle.HIGHEST_PROTOCOL)
 
    eval_data, eval_labels = get_tensor_data(output_mode, eval_features)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    if task_name == "mnli":
        processor = processors["mnli-mm"]()
        try:
            dev_mm_file = train_file = os.path.join(processed_data_dir,'dev-mm_data.pkl')
            mm_eval_features = pickle.load(open(dev_mm_file,'rb'))
        except:
            mm_eval_examples = processor.get_dev_examples(data_dir)
            mm_eval_features = convert_examples_to_features(
                mm_eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
            with open(dev_mm_file, 'wb') as f:
                pickle.dump(mm_eval_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        mm_eval_data, mm_eval_labels = get_tensor_data(output_mode, mm_eval_features)
        # logger.info("  Num examples = %d", len(mm_eval_features))

        mm_eval_sampler = SequentialSampler(mm_eval_data)
        mm_eval_dataloader = DataLoader(mm_eval_data, sampler=mm_eval_sampler,
                                        batch_size=args.batch_size)


    # ================================================================================ #
    # Build Teacher Model
    # ================================================================================ # 
    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model, num_labels=num_labels)
    
    teacher_model.to(device)
    teacher_model.eval()
    
    if n_gpu > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
    
    result = do_eval(teacher_model, task_name, eval_dataloader,
                    device, output_mode, eval_labels, num_labels)
    
    # ================================================================================  #
    # Save Teacher Model Peroformance for KD Training
    # ================================================================================ #
    if task_name in acc_tasks:
        if task_name in ['sst-2','mnli','qnli','rte']:
            fp32_performance = f"acc:{result['acc']}"
            fp32_score = result['acc']
        elif task_name in ['mrpc','qqp']:
            fp32_performance = f"f1/acc:{result['f1']}/{result['acc']} avg : {(result['f1'] + result['acc'])*50}"
            fp32_score = (result['f1'] + result['acc'])*50
    if task_name in corr_tasks:
        fp32_performance = f"pearson/spearmanr:{result['pearson']}/{result['spearmanr']} corr:{result['corr']}"
        fp32_score = result['corr']*100

    if task_name in mcc_tasks:
        fp32_performance = f"mcc:{result['mcc']}"
        fp32_score = result['mcc']

    if task_name == "mnli":
        result = do_eval(teacher_model, 'mnli-mm', mm_eval_dataloader,
                            device, output_mode, mm_eval_labels, num_labels)
        fp32_performance += f"  mm-acc:{result['acc']}"
        fp32_score = result['acc']
    fp32_performance = task_name +' fp32   ' + fp32_performance
    
    # ================================================================================  #
    # Build Student Model
    # ================================================================================ #
    student_config = BertConfig.from_pretrained(args.student_model, 
                                                clip_val = args.clip_val,
                                                quantize = args.quantize,
                                                ffn_q_1 = args.ffn_1,
                                                ffn_q_2 = args.ffn_2,
                                                qkv_q = args.qkv,
                                                emb_q = args.emb,
                                                cls_q = args.cls,
                                                mean_scale = args.mean_scale,
                                                teacher_attnmap = args.teacher_attnmap,
                                                teacher_context = args.teacher_context,
                                                teacher_output = args.teacher_output,
                                                )
    
    student_model = QuantBertForSequenceClassification.from_pretrained(args.student_model, config = student_config, num_labels=num_labels)
    student_model.to(device)
    
    # ================================================================================  #
    # Training Setting
    # ================================================================================ #
    if n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
    param_optimizer = list(student_model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in (no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    schedule = 'warmup_linear'
    optimizer = BertAdam(optimizer_grouped_parameters,
                            schedule=schedule,
                            lr=args.learning_rate,
                            warmup=0.1,
                            t_total=num_train_optimization_steps)
    
    
    norm_func = torch.linalg.norm
    loss_cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    global_step = 0
    best_dev_acc = 0.0
    previous_best = None
    
    # ================================================================================  #
    # Training Start
    # ================================================================================ #

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    
    # Loss Init AverageMeter
    l_gt_loss = AverageMeter()
    l_att_loss = AverageMeter()
    l_rep_loss = AverageMeter()
    l_cls_loss = AverageMeter()
    l_output_loss = AverageMeter()
    l_loss = AverageMeter()
    
    mixed_status = None
    ce_loss_func = CrossEntropyLoss()
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    cos_loss_func  = CosineEmbeddingLoss()
    loss_mse = MSELoss()
    
    for epoch_ in range(int(num_train_epochs)):
        
        for batch in train_dataloader:
            
            # Gradual TI (You could try other TI options - Stochastic/Inverted)
            if args.training_type == "qat_step1" and args.teacher_gradual:
                if global_step < num_train_optimization_steps / 6:
                    student_config.teacher_output = True
                    mixed_status = "OI"
                elif global_step < num_train_optimization_steps / 3:
                    student_config.teacher_output = False
                    student_config.teacher_context = True
                    mixed_status = "CI"
                else:
                    student_config.teacher_output = False
                    student_config.teacher_context = False
                    student_config.teacher_attnmap = True
                    mixed_status = "MI"
            
            if args.training_type == "qat_step1" and args.teacher_stochastic:
                rand_int = torch.randint(1,4,(1,))[0].item()
                
                if rand_int == 1 :
                    student_config.teacher_output = True
                    student_config.teacher_context = False
                    student_config.teacher_attnmap = False
                    mixed_status = "OI"
                elif rand_int == 2 :
                    student_config.teacher_output = False
                    student_config.teacher_context = True
                    student_config.teacher_attnmap = False
                    mixed_status = "CI"
                else:
                    student_config.teacher_output = False
                    student_config.teacher_context = False
                    student_config.teacher_attnmap = True
                    mixed_status = "MI"

            if args.training_type == "qat_step1" and args.teacher_inverted:
                if global_step < num_train_optimization_steps / 3:
                    student_config.teacher_attnmap = True
                    mixed_status = "MI"
                elif global_step < num_train_optimization_steps * 2/ 3:
                    student_config.teacher_attnmap = False
                    student_config.teacher_context = True
                    mixed_status = "CI"
                else:
                    student_config.teacher_attnmap = False
                    student_config.teacher_context = False
                    student_config.teacher_output = True
                    mixed_status = "OI"

            student_model.train()
            
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, seq_lengths = batch
            
            # tmp loss init
            att_loss = 0.
            rep_loss = 0.
            cls_loss = 0.
            attscore_loss = 0.
            output_loss = 0.
            loss = 0.
            
            with torch.no_grad():
                teacher_logits, teacher_atts, teacher_reps, teacher_probs, teacher_attn_blocks = teacher_model(input_ids, segment_ids, input_mask)
        
            student_logits, student_atts, student_reps, student_probs, student_attn_blocks = student_model(input_ids, segment_ids, input_mask, teacher_outputs=(teacher_probs, teacher_attn_blocks))

            # We did not use GT-Loss for fair comparison to TernaryBERT QAT (note that GT-loss helps boosting resulting accuracy in some tasks)
            if args.gt_loss:
                if output_mode == "classification":
                    loss = ce_loss_func(student_logits, label_ids)
                    
                elif output_mode == "regression":
                    loss = loss_mse(student_logits, teacher_logits)
                
                l_gt_loss.update(loss.item())

            # Pred Loss (TernaryBERT Loss)
            if args.pred_distill:
                if output_mode == "classification":
                    cls_loss = soft_cross_entropy(student_logits,teacher_logits)
                elif output_mode == "regression":
                    cls_loss = MSELoss()(student_logits, teacher_logits)
                else:
                    cls_loss = soft_cross_entropy(student_logits,teacher_logits)
                l_cls_loss.update(cls_loss.item())

            # Output Loss 
            if args.output_distill:
                for i, (student_attn_block, teacher_attn_block) in enumerate(zip(student_attn_blocks, teacher_attn_blocks)):    
                    tmp_loss = MSELoss()(student_attn_block[1], teacher_attn_block[1]) # 1 : Attention Output 0 : Layer Context
                    output_loss += tmp_loss
                l_output_loss.update(output_loss.item())
            
            # Attention Score Loss (TernaryBERT Loss)
            if args.attn_distill:
                for i, (student_att, teacher_att) in enumerate(zip(student_atts, teacher_atts)):    
                            
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to("cuda"),
                                                student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to("cuda"),
                                                teacher_att)
                    tmp_loss = MSELoss()(student_att, teacher_att)
                    attscore_loss += tmp_loss
                l_att_loss.update(attscore_loss.item())

            # Rep Distill (TernaryBERT Loss)
            if args.rep_distill:
                for i, (student_rep, teacher_rep) in enumerate(zip(student_reps, teacher_reps)):
                    tmp_loss = MSELoss()(student_rep, teacher_rep)
                    rep_loss += tmp_loss
                l_rep_loss.update(rep_loss.item())

            loss += cls_loss + rep_loss + output_loss + attscore_loss 
            l_loss.update(loss.item())

            if n_gpu > 1:
                loss = loss.mean()           
                
            # Zero Step Loss Update
            if global_step == 0: 
                if run is not None:           
                    run["loss/total_loss"].log(value=l_loss.avg, step=global_step)
                    run["loss/gt_loss_loss"].log(value=l_gt_loss.avg, step=global_step)
                    run["loss/att_loss_loss"].log(value=l_att_loss.avg, step=global_step)
                    run["loss/rep_loss_loss"].log(value=l_rep_loss.avg, step=global_step)
                    run["loss/cls_loss_loss"].log(value=l_cls_loss.avg, step=global_step)
                    run["loss/output_loss_loss"].log(value=l_output_loss.avg, step=global_step)
                    
                    run["metrics/lr"].log(value=optimizer.get_lr()[0], step=global_step)
 
            loss.backward()
            optimizer.step() 
            optimizer.zero_grad()
            
            global_step += 1
            # ================================================================================  #
            #  Evaluation
            # ================================================================================ #

            if global_step % args.eval_step == 0 or global_step == num_train_optimization_steps-1: # period or last step
                
                student_model.eval()
                
                result = do_eval(student_model, task_name, eval_dataloader,
                                    device, output_mode, eval_labels, num_labels, teacher_model=teacher_model)
            
                result['global_step'] = global_step
                result['cls_loss'] = l_cls_loss.avg
                result['att_loss'] = l_att_loss.avg
                result['rep_loss'] = l_rep_loss.avg
                result['loss'] = l_loss.avg
                
                # Basic Logging (Training Loss, Clip Val)
                if run is not None:
                    
                    run["loss/total_loss"].log(value=l_loss.avg, step=global_step)
                    run["loss/gt_loss_loss"].log(value=l_gt_loss.avg, step=global_step)
                    run["loss/att_loss_loss"].log(value=l_att_loss.avg, step=global_step)
                    run["loss/rep_loss_loss"].log(value=l_rep_loss.avg, step=global_step)
                    run["loss/cls_loss_loss"].log(value=l_cls_loss.avg, step=global_step)
                    run["loss/output_loss_loss"].log(value=l_output_loss.avg, step=global_step)        
                    run["metrics/lr"].log(value=optimizer.get_lr()[0], step=global_step)

                if task_name=='cola':
                    eval_score = result["mcc"]
                    if run is not None:
                        run["metrics/mcc"].log(value=result['mcc'], step=global_step)

                    eval_result = result["mcc"]  
                    # logger.info(f"Eval Result is {result['mcc']}")
                elif task_name in ['sst-2','mnli','mnli-mm','qnli','rte','wnli']:
                    eval_score = result["acc"]
                    if run is not None:
                        run["metrics/acc"].log(value=result['acc'],step=global_step)
                        
                    logger.info(f"Eval Result is {result['acc']}")
                    eval_result = result["acc"]
                elif task_name in ['mrpc','qqp']:
                    eval_score = result["acc_and_f1"]
                    if run is not None:
                        run["metrics/acc_and_f1"].log(value=result['acc_and_f1'],step=global_step)
                        
                    # logger.info(f"Eval Result is {result['acc']}, {result['f1']}")
                    eval_result = result["acc_and_f1"]
                else:
                    eval_score = result["corr"]
                    if run is not None:
                        run["metrics/corr"].log(value=result['corr'],step=global_step)
                        
                    # logger.info(f"Eval Result is {result['corr']}")
                    eval_result = result["corr"]

                if args.training_type == "qat_step1":        
                    logger.info(f"Gradual-{mixed_status}-{global_step}-SAVE : {eval_result*100}")
                    model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                    quant_model = copy.deepcopy(model_to_save)
                     
                    output_model_file = os.path.join(output_quant_dir, WEIGHTS_NAME)
                    output_config_file = os.path.join(output_quant_dir, CONFIG_NAME)

                    torch.save(quant_model.state_dict(), output_model_file)
                    model_to_save.config.to_json_file(output_config_file)
                    tokenizer.save_vocabulary(output_quant_dir)
                
                # Save Model
                save_model = False

                if task_name in acc_tasks and result['acc'] > best_dev_acc:
                    if task_name in ['sst-2','mnli','qnli','rte']:
                        previous_best = f"{result['acc']*100}"
                    elif task_name in ['mrpc','qqp']:
                        previous_best = f"{(result['f1'] + result['acc'])*50}"
                    best_dev_acc = result['acc']
                    save_model = True

                if task_name in corr_tasks and result['corr'] > best_dev_acc:
                    previous_best = f"{result['corr']*100}"
                    best_dev_acc = result['corr']
                    save_model = True

                if task_name in mcc_tasks and result['mcc'] > best_dev_acc:
                    previous_best = f"{result['mcc']*100}"
                    best_dev_acc = result['mcc']
                    save_model = True

                if save_model:
                    # logger.info("====> Best Score *****")
                    # Test mnli-mm
                    if task_name == "mnli":
                        result = do_eval(student_model, 'mnli-mm', mm_eval_dataloader,
                                            device, output_mode, mm_eval_labels, num_labels, teacher_model=teacher_model)
                        previous_best+= f"mm-acc:{result['acc']}"

                    if args.training_type == "qat_step1":
                        logger.info(fp32_performance)
                        logger.info(previous_best)

                    if args.save_fp_model:
                        # logger.info("***** Save full precision model *****")
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                        output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(output_dir, CONFIG_NAME)

                        torch.save(model_to_save.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(output_dir)

                    if args.save_quantized_model and not args.training_type == "qat_step1":
                        # logger.info("====> Save quantized model *****")

                        # output_quant_dir = os.path.join(output_dir, 'quant')
                        output_quant_dir = os.path.join(output_dir, 'exploration')
                        if not os.path.exists(output_quant_dir):
                            os.mkdir(output_quant_dir)

                        if not os.path.exists(output_quant_dir):
                            os.makedirs(output_quant_dir)
                        
                        output_quant_dir = os.path.join(output_quant_dir, args.exp_name)
                        if not os.path.exists(output_quant_dir):
                            os.makedirs(output_quant_dir)
                        
                        model_to_save = student_model.module if hasattr(student_model, 'module') else student_model
                        quant_model = copy.deepcopy(model_to_save)
                            
                        output_model_file = os.path.join(output_quant_dir, WEIGHTS_NAME)
                        output_config_file = os.path.join(output_quant_dir, CONFIG_NAME)

                        torch.save(quant_model.state_dict(), output_model_file)
                        model_to_save.config.to_json_file(output_config_file)
                        tokenizer.save_vocabulary(output_quant_dir)
                        
        
                
                # TI QAT Step-1
                if global_step >= num_train_optimization_steps and args.training_type == "qat_step1":
                    
                    if global_step >= ti_step_1_total_step:
                        logger.info(f"==> TI-step1 Last Result = {eval_result}")
                        best_txt = os.path.join(output_quant_dir, "best_info.txt")
                        with open(best_txt, "w") as f_w:
                            f_w.write(previous_best)
                        return
                    
    logger.info(f"==> Previous Best = {previous_best}")
    
    # Save Best Score
    if args.save_quantized_model:
        best_txt = os.path.join(output_quant_dir, "best_info.txt")
        last_txt = os.path.join(output_quant_dir, "last_info.txt")
        with open(best_txt, "w") as f_w:
            f_w.write(previous_best)
        with open(last_txt, "w") as f_w:
            f_w.write(f"{eval_result*100}")
            # f_w.write(f"Last Result = {result}")

if __name__ == "__main__":
    main()
