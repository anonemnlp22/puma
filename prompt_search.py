from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2TokenizerFast, T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
import numpy as np
import random
## loading edge scoring dataset
import csv, json, time
import os, sys
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import sys
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, get_linear_schedule_with_warmup, default_data_collator

torch.autograd.set_detect_anomaly(True)
sys.path.append("PathRetriever/")
from retriever.tfidf_vectorizer_article import TopTfIdf
from pipeline.tfidf_retriever import TfidfRetriever
from nltk.tokenize import sent_tokenize
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger("transformers.tokenization_utils_base").setLevel(
    logging.ERROR)
from src.eval_utils import evaluate_retrieval, threaded_eval
from src.prompt_utils import *
from src.data_utils import *
from src.gpt_prompt_tuning import GPT2PromptTuningLM


## set seed
def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_metrics(metrics, best=False, test=False):
    for k, v in metrics.items():
        if "count" in k or "stuff" in k:
            continue
        if best:
            k = "best " + k
        elif test:
            k = "test " + k

        if not isinstance(v, float):
            continue
        print("%s: %.4f" % (k, v))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="google/t5-large-lm-adapt")
    parser.add_argument(
        "--train_data",
        type=str)
    parser.add_argument(
        "--val_data",
        type=str)

    parser.add_argument("--test_data", type=str, default=None)

    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument(
        "--prompt_template_file",
        type=str,
        default=None
    )
    parser.add_argument("--ensemble_prompts", action="store_true")
    parser.add_argument("--ensembling_method",
                        type=str,
                        default="mean",
                        choices=["mean", "max", "token"])
    parser.add_argument("--n_train_examples",
                        type=int,
                        default=100,
                        help="Number of positive examples")
    parser.add_argument("--unlikelihood_loss",
                        action="store_true",
                        help="Use unlikelihood loss on negative")
    parser.add_argument("--contrastive_loss",
                        action='store_true',
                        help="Use contrastive CE loss")
    parser.add_argument(
        "--scoring_method",
        type=str,
        default="conditional_plp",
        help="method used to compute relevance score",
        choices=["prompt_plp", "conditional_plp", "raw", "tfidf"])
    parser.add_argument("--contrastive_loss_weight",
                        type=float,
                        default=0.1,
                        help="weight for contrastive representation loss")
    parser.add_argument("--contrastive_loss_type",
                        type=str,
                        choices=["ce", "ul"],
                        default="ce")
    parser.add_argument("--pooling_method",
                        type=str,
                        default="mean",
                        help="method used to compute representation")
    parser.add_argument("--unlikelihood_loss_weight",
                        type=float,
                        default=0.1,
                        help="weight for unlikelihood loss")
    parser.add_argument("--train_only", action="store_true", help="train only")
    parser.add_argument("--eval_mode",
                        type=str,
                        default="hotpot",
                        choices=["hotpot", "nq"],
                        help="eval model")
    parser.add_argument("--eval_batch_size", type=int, default=50)
    parser.add_argument("--n_eval_examples", type=int, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tfidf_pool_size", type=int, default=40)
    parser.add_argument("--top_k_second_hops", type=int, default=3)
    parser.add_argument("--top_k_first_hops", type=int, default=5)
    parser.add_argument(
        "--use_raw_score",
        action="store_true",
        help="use unnormalized raw score (logits) instead of logprobs")
    parser.add_argument("--negative_method",
                        type=str,
                        default="tfidf_links",
                        choices=["sf", "tfidf", "tfidf_links", "sf_tfidf"])
    parser.add_argument("--n_negs", type=int, default=20)
    parser.add_argument("--combine_hops_method",
                        type=str,
                        default="separate",
                        choices=["separate", "sep-concat", "concat"])
    parser.add_argument("--prepend_title",
                        action="store_true",
                        help="prepend titles to paragraphs")
    parser.add_argument("--max_doc_len", type=int, default=230)
    parser.add_argument("--max_prompt_len", type=int, default=600)
    parser.add_argument("--prefix_tuning",
                        action="store_true",
                        help="use prefix tuning")
    parser.add_argument("--prefix_len", type=int, default=10)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--rand_train_data",
                        action='store_true',
                        help="sample random train data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--db_path",
        type=str)
    parser.add_argument(
        "--tfidf_retriever_path",
        type=str)
    parser.add_argument("--lower_case", action="store_true", help="lower case")
    parser.add_argument("--use_bm25",
                        action="store_true",
                        help="rerank with bm25")
    parser.add_argument("--load_params", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--last_layer_only",
                        action="store_true",
                        help="finetune last layer only")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--metric_out_path", type=str, required=True)
    parser.add_argument("--H", type=int, default=2, help="number of hops")
    parser.add_argument("--beam_sizes", type=str, default="5", help="H-1 beam sizes separated by commas")
    parser.add_argument("--top_k_links" , type=int, default=3, help="top k links to expand per path")

    args = parser.parse_args()
    init_seed(args.seed)

    ws_dir = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/data"

    print("Scoring Model used ====> {} ".format(args.model))

    if 'gpt2' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                     cache_dir=os.path.join(
                                                         ws_dir, "cache"))
        tokenizer = GPT2TokenizerFast.from_pretrained(args.model,
                                                      cache_dir=os.path.join(
                                                          ws_dir, "cache"))
        tokenizer.pad_token = tokenizer.eos_token

    elif 't5' in args.model:
        model = T5ForConditionalGeneration.from_pretrained(
            args.model, cache_dir=os.path.join(ws_dir, "cache"))
        tokenizer = T5Tokenizer.from_pretrained(args.model,
                                                cache_dir=os.path.join(
                                                    ws_dir, "cache"))

    elif 'bart' in args.model:
        model = BartForConditionalGeneration.from_pretrained(
            args.model, cache_dir=os.path.join(ws_dir, "cache"))
        tokenizer = BartTokenizer.from_pretrained(args.model,
                                                  cache_dir=os.path.join(
                                                      ws_dir, "cache"))

    model = model.to(args.device)

    db_path = args.db_path
    tfidf_retriever = TfidfRetriever(
        db_path,
        args.tfidf_retriever_path,
    )

    MAX_TEMPLATES = 200
    ## open prmopt tempalte file
    with open(args.prompt_template_file, 'r') as f:
        prompt_templates = [l.replace("Task:","").strip() for l in f.readlines()]
        for i, t in enumerate(prompt_templates):
            if not t.endswith("Question:"):
                prompt_templates[i] = t + " Question:"
        ## adding baseline template
        prompt_templates.insert(
            0,
            "Read the following documents and answer the question. Document: <P> Question:"
        )
        prompt_templates = prompt_templates[:MAX_TEMPLATES]
        print ("Number of prompt templates = {}".format(len(prompt_templates)))

    print("Starting search...")

    best_rec_2 = 0
    best_template = None
    all_metrics = {}

    prompt_templates = ["<P>"] + prompt_templates
    for i, template in enumerate(prompt_templates):
        print("***********************************************************")
        print("[{}] Template: {}".format(i+1, template))
        setattr(args, "prompt_template", template)
        assert "<P>" in template, "invalid prompt format!"

        if args.val_data.endswith(".json"):
            with open(args.val_data) as f:
                val_data = json.load(f)

            if args.eval_mode == "nq":
                # convert to hotpot format
                val_data = convert_nq_to_hotpot(val_data)

        #start_time = time.time()
        metrics = evaluate_retrieval(args, model,
        val_data,
        tokenizer,
        tfidf_retriever=tfidf_retriever,
        get_retrieved=False)

        rec_2 = metrics["REC_AT_2"]
        print("REC@2: {}".format(round(rec_2, 2)))
        if rec_2 > best_rec_2:
            print("New best REC@2!")
            best_rec_2 = rec_2
            best_template = template

        all_metrics[template] = metrics

    print("Best template: {}".format(best_template))
    print("Best REC@2: {}".format(best_rec_2))

    ## save all metrics to json file
    with open(args.metric_out_path, 'w') as f:
        json.dump(all_metrics, f)
