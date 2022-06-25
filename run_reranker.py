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
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
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

retriever_in_context_template1 = """
 Read the following documents and answer the question.
 Document: Lars Kristinus Larsen is a Danish businessman, owner and founder of the Jysk retail chain.
 Document: Jysk is a Danish retail chain, selling household goods such as mattresses, furniture and interior décor.
 Question: What does the retail chain founded by Lars Larsen sell?
 Read the following documents and answer the question.
 Document: Michael John Moorcock (born 18 December 1939) is an English writer, primarily of science fiction and fantasy, who has also published literary novels.
 He is best known for his novels about the character Elric of Melniboné, a seminal influence on the field of fantasy in the 1960s and 1970s.
 Document: Elizabeth Jane Howard, CBE, FRSL (26 March 1923 – 2 January 2014), was an English novelist.
 She had previously been an actress and a model.
 Question: Is Elizabeth Jane Howard and Michael Moorcock both English writers or novelist?
 Read the following documents and answer the question.
 Document: <P> Question:
""".strip().replace("\n", "").replace("\t", "")
retriever_in_context_template1 = " ".join(retriever_in_context_template1.split())

retriever_in_context_template2 = """
Read the following documents and answer the question.
 Document: Frank Vernon Ramsey is an American former professional basketball player and coach.
 Ramsey was also a head coach for the Kentucky Colonels of the ABA during the 1970–1971 season.
 Question: which team was Ramsey a head coach for during the 1970–1971 season?
 Read the following documents and answer the question.
 Document: Jysk is a Danish retail chain, selling household goods such as mattresses and interior décor.
 Question: What does Jysk retail chain sell?
Read the following documents and answer the question.
 Document: <P> Question:
""".strip().replace("\n", "").replace("\t", "")
retriever_in_context_template2 = " ".join(
    retriever_in_context_template2.split())


prompt_templates_for_ensembling = [
    'Document: <P> Review previous documents and ask some question. Question:',
    'Document: <P> Review the previous documents and answer question. Question:',
    'Document: <P> Read the previous documents and write the following question. Question:',
    'Document: <P> Search previous documents and ask the question. Question:',
    'To analyze the documents and ask question. Document: <P> Question:',
    #'Document: <P> To read the previous documents and write a question. Question:',
    #'Document: <P> Read previous documents and write your exam question. Question:',
    #'Document: <P> Read the previous documents and ask this question. Question:',
    #'Read two documents and answer a question. Document: <P> Question:',
    #'Identify all documents and ask question. Document: <P> Question:'
]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2-medium")
    parser.add_argument("--demos_file", type=str, default=None)
    parser.add_argument(
        "--val_data",
    )

    parser.add_argument(
        "--test_data",
        type=str,
        default=None)

    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--warmup_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--outdir",
        type=str,
        default=None)
    parser.add_argument("--save_retrieved_path", type=str, default=None)
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=
        'Document: <P> Review previous documents and ask some question. Question:'
    )
    parser.add_argument("--ensemble_prompts", action="store_true")
    parser.add_argument("--ensembling_method", type=str, default="mean")
    parser.add_argument("--n_train_examples",
                        type=int,
                        default=100,
                        help="Number of positive examples")

    parser.add_argument("--demos_ids", type=str, default="0,1", help="demos ids to use")
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
    parser.add_argument("--contrastive_loss_type", type=str, choices=["ce", "ul"], default="ce")
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
    parser.add_argument("--top_k_second_hops",
                        type=int,
                        default=3)
    parser.add_argument("--top_k_first_hops", type=int, default=5)
    parser.add_argument("--use_raw_score", action="store_true", help="use unnormalized raw score (logits) instead of logprobs")
    parser.add_argument("--negative_method",
                        type=str,
                        default="tfidf_links",
                        choices=["sf", "tfidf", "tfidf_links", "sf_tfidf"])
    parser.add_argument("--n_negs", type=int, default=20)
    parser.add_argument("--combine_hops_method", type=str, default="separate", choices=["separate", "sep-concat", "concat"])
    parser.add_argument("--prepend_title",
                        action="store_true",
                        help="prepend titles to paragraphs")
    parser.add_argument("--max_doc_len", type=int, default=230)
    parser.add_argument("--demo_max_doc_len", type=int, default=100)
    parser.add_argument("--max_prompt_len", type=int, default=600)
    parser.add_argument("--demo_max_prompt_len", type=int, default=1024)
    parser.add_argument("--prefix_tuning", action="store_true", help="use prefix tuning")
    parser.add_argument("--prefix_len", type=int, default=10)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--rand_train_data", action='store_true', help="sample random train data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--db_path", type=str,)
    parser.add_argument("--tfidf_retriever_path", type=str)
    parser.add_argument("--lower_case", action="store_true", help="lower case")
    parser.add_argument("--use_bm25", action="store_true", help="rerank with bm25")
    parser.add_argument("--load_params", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=2)
    parser.add_argument("--last_layer_only", action="store_true", help="finetune last layer only")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--reverse_path", action="store_true", help="reverse document paths")
    parser.add_argument("--truncate_demos",
                        action="store_true",
                        help="truncate demos to max_doc_len")
    parser.add_argument("--n_ensemble_demos", type=int, default=3)

    args = parser.parse_args()
    init_seed(args.seed)

    ws_dir = os.environ["WORKSPACE"] if "WORKSPACE" in os.environ else "/data"

    print("Scoring Model used ====> {} ".format(args.model))

    if 'gpt2' in args.model:
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                                cache_dir=os.path.join(ws_dir, "cache"))
        tokenizer = GPT2TokenizerFast.from_pretrained(args.model,
                                                cache_dir=os.path.join(ws_dir, "cache"))
        tokenizer.pad_token = tokenizer.eos_token

    elif 't5' in args.model:
        model = T5ForConditionalGeneration.from_pretrained(args.model, cache_dir=os.path.join(ws_dir, "cache"))
        tokenizer = T5Tokenizer.from_pretrained(args.model, cache_dir=os.path.join(ws_dir, "cache"))

    elif 'bart' in args.model:
        model = BartForConditionalGeneration.from_pretrained(args.model, cache_dir=os.path.join(ws_dir, "cache"))
        tokenizer = BartTokenizer.from_pretrained(args.model, cache_dir=os.path.join(ws_dir, "cache"))

    model = model.to(args.device)

    if args.load_params:
        print("initializing model params from {}".format(args.load_params))
        ckpt = torch.load(args.load_params)
        state_dict = ckpt["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'module' in k:
                name = k[7:] # remove `module.`
                new_state_dict[name] = v

        # load params
        model.load_state_dict(new_state_dict)

    prompt_template = args.prompt_template  # use first prompt format for now
    assert "<P>" in prompt_template, "invalid prompt format!"

    if args.demos_file: ## append in-context demos to prompt
        print("Loading in-context demos from {}".format(args.demos_file))

        with open(args.demos_file, "r") as f:
            demos = json.load(f)

        if args.truncate_demos:
            for d in demos:
                d["gold_path_texts"] = [shorten_paragraph_if_necessary(p, args.demo_max_doc_len, tokenizer=tokenizer) for p in d["gold_path_texts"]]

        #import ipdb; ipdb.set_trace()
        if args.ensemble_prompts:
            print("Ensemble in-context demos!")
            ## sample
            prompt_template_ensemble = []
            even_ids = [i for i in range(0, len(demos), 2)] # for comparison questions
            odd_ids = [i for i in range(1, len(demos), 2)] # for bridge questions

            ## sample args.n_ensemble_demos even and odd ids
            even_ids = np.random.choice(even_ids, args.n_ensemble_demos, replace=False)
            odd_ids = np.random.choice(odd_ids, args.n_ensemble_demos, replace=False)
            ids = zip(even_ids, odd_ids)
            #ids = [(0,1), (2,3), (0,3), (1,2)]

            for ids_tuple in ids:
                exs = []
                print(ids_tuple)
                for i in ids_tuple:
                    exs.append(create_prompt(demos[i]["gold_path_texts"],
                                        prompt_template=prompt_template
                                        ) + " " + demos[i]["question"])
                t = " ".join(exs) + " " + prompt_template
                prompt_template_ensemble.append(t)

            prompt_template = prompt_template_ensemble

        else:
            print("Using in-context demos with ids {} -- no ensemble".format(args.demos_ids))

            ids = [int(i) for i in args.demos_ids.split(",")]

            demos = [demos[i] for i in ids]
            examples = []
            for d in demos:
                example = create_prompt(d["gold_path_texts"],
                                        prompt_template=prompt_template
                                        ) + " " + d["question"]
                examples.append(example)

            prompt_template = " ".join(examples) + " " + prompt_template

        setattr(args, "max_prompt_len", args.demo_max_prompt_len)

    elif args.ensemble_prompts:
        print("Ensemble prompts -- no in-context demos")
        prompt_template = prompt_templates_for_ensembling


    setattr(args, "prompt_template", prompt_template)
    print("Prompt template(s) used: ", prompt_template)

    db_path = args.db_path
    tfidf_retriever = TfidfRetriever(
        db_path,
        args.tfidf_retriever_path,
    )

    if args.val_data.endswith(".json"):
        with open(args.val_data) as f:
            val_data = json.load(f)

        if args.eval_mode == "nq":
            # convert to hotpot format
            val_data = convert_nq_to_hotpot(val_data)

    print("-- For evaluation, using {} examples".format(args.n_eval_examples))

    if args.n_train_examples == 0:
        #start_time = time.time()
        print("******* Zero-shot retrieval *******")
        metrics_docs = threaded_eval(args, model,
        val_data,
        tokenizer,
        tfidf_retriever=tfidf_retriever,
        n_threads=args.n_workers)

        log_metrics(metrics_docs, best=True)

        if args.save_retrieved_path:
            with open(args.save_retrieved_path, "w") as f:
                json.dump(metrics_docs, f)

            with open(args.save_retrieved_path[:-5] + ".args", "w") as f:
                json.dump(vars(args), f)

        sys.exit()

    ## fine-tuning the model on the training data
    print("\n\n-- Fine-tuning the model on the training data...")

    json_train_data = json.load(open(args.train_data))

    if args.n_train_examples is not None:
        if args.rand_train_data:
            print("randomly sampling {} examples".format(
                args.n_train_examples))
            json_train_data = np.random.choice(json_train_data,
                                         args.n_train_examples,
                                         replace=False)
        else:
            json_train_data = json_train_data[:args.
                                  n_train_examples]  # take the first n_examples

    if args.eval_mode == "nq":
        # convert to hotpot format
        json_train_data = convert_nq_to_hotpot(json_train_data)

    ## loading training data
    processed_train_data = process_data(args,
                                        json_train_data,
                                        tokenizer,
                                        retriever=tfidf_retriever)

    print("-- For training: we have {} positive and negative examples".format(
        len(processed_train_data)))

    ## focus on positive examples only for now --
    train_dataset = PromptRetrievalDataset(processed_train_data,
                                           tokenizer=tokenizer,
                                           max_prompt_len= args.max_prompt_len,)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

    ## optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    #metrics = threaded_eval(args, model,
    #                val_data,
    #                tokenizer,
    #                tfidf_retriever=tfidf_retriever,
    #                n_threads=args.n_workers,)
    if args.last_layer_only:
        for n, p in model.named_parameters():
            if 'base' in args.model and '9' not in n and '10' not in n and '11' not in n and 'lm' not in n and 'final' not in n:
                p.requires_grad = False

            elif 'large' in args.model and '22' not in n and '23' not in n and 'lm' not in n and 'final' not in n:
                p.requires_grad = False

            elif 'xl' in args.model and '22' not in n and '23' not in n and 'lm' not in n and 'final' not in n:
                p.requires_grad = False


    logger.info("***** Running Retriever training *****")
    best_metric_value_so_far = 0
    best_all_metrics = {}

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0
        model.train()
        cur_step = 0

        for step, batch in tqdm(enumerate(train_dataloader)):
            pos_input_ids = batch["pos_input_ids"].to(model.device)
            pos_attn_masks = batch["pos_attn_masks"].to(model.device)

            neg_input_ids = torch.cat([t.unsqueeze(1).to(model.device) for t in batch["neg_input_ids"]], dim=1) # (B, NNEG, L)
            neg_attn_masks = torch.cat([t.unsqueeze(1).to(model.device) for t in batch["neg_attn_masks"]], dim=1) # (B, NNEG, L)

            question_input_ids = batch["question_input_ids"].to(model.device)

            B, T = pos_input_ids.shape
            nneg, _  = neg_input_ids.shape[1:]

            labels = question_input_ids
            labels[labels == tokenizer.pad_token_id] = -100


            pos_out = model(pos_input_ids,
                            attention_mask=pos_attn_masks,
                            labels=labels)

            pos_logits = pos_out.logits
            pos_loss = pos_out.loss

            if not args.contrastive_loss: # MLE loss
                loss = pos_loss
                epoch_loss += loss.item()

            else:
                ## flatten
                neg_input_ids = neg_input_ids.view(-1, neg_input_ids.shape[-1])
                neg_attn_masks = neg_attn_masks.view(-1, neg_attn_masks.shape[-1])

                neg_logits = []
                shift = True if "gpt" in model.config._name_or_path else False

                neg_loss = 0

                for i in range(0, neg_input_ids.shape[0], args.batch_size):
                    neg_out = model(neg_input_ids[i:i+args.batch_size],
                                        attention_mask=neg_attn_masks[i:i+args.batch_size],
                                        labels=labels)

                    neg_loss += -neg_out.loss
                    neg_logits.append(neg_out.logits)

                neg_logits = torch.cat(neg_logits, dim=0)
                ## obtaining scores for negative examples

                ## repeat labels nneg times
                labels_rep = labels.repeat(nneg, 1)
                neg_log_probs = batch_get_logprob_from_logits(neg_logits, labels_rep, shift=shift)
                neg_log_probs = neg_log_probs.view(B, nneg) # (B, NNEG)

                ## obtaining scores for positive examples
                pos_log_probs = batch_get_logprob_from_logits(pos_logits, labels, shift=shift)
                pos_loss = -pos_log_probs.mean()

                if args.contrastive_loss_type == "ce":
                    temp = 1.0
                    log_probs = torch.cat([pos_log_probs.unsqueeze(1) / temp , neg_log_probs / temp], dim=1).to(model.device) # (B, NNEG+1)
                    ce_labels = torch.zeros(log_probs.shape[0], dtype=torch.long).to(model.device)
                    loss_fct = torch.nn.CrossEntropyLoss()
                    loss =  loss_fct(log_probs, ce_labels) #  48.00
                elif args.contrastive_loss_type == "ul":
                    loss = pos_loss - torch.log(1 - torch.exp(neg_loss)) # 49.40

                epoch_loss += loss.item()

            loss.backward()

            if cur_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            cur_step += 1

        print("Finished epoch {}: \n\t\t loss: {}".format(
            epoch, epoch_loss / cur_step))

        # evaluate model on validation data
        if not args.train_only:
            if epoch % args.eval_every == 0:
                print(" -- Evaluating on validation data...")
                metrics = threaded_eval(args, model,
                    val_data,
                    tokenizer,
                    tfidf_retriever=tfidf_retriever,
                    n_threads=args.n_workers,)

                retrieved_docs = metrics["questions_docs"]
                del metrics["questions_docs"]

                log_metrics(metrics)
                best_metric = "REC_AT_2"
                best_metric_value = metrics[best_metric]

                if best_metric_value > best_metric_value_so_far:
                    print("-- Obtained better {}: {}".format(
                        best_metric, best_metric_value))

                    best_metric_value_so_far = best_metric_value
                    best_all_metrics = metrics

                    if args.outdir:
                        # save model
                        path = args.outdir
                        os.makedirs(path, exist_ok=True)
                        torch.save({'model_state_dict': model.state_dict()},
                                os.path.join(path, 'model.pt'))
                        ## save metrics
                        with open(os.path.join(path, 'metrics.json'), 'w+') as f:
                            json.dump(best_all_metrics, f)

                        ## save retrieved docs
                        with open(os.path.join(path, 'val_retrieved_docs.json'), 'w+') as f:
                            json.dump(retrieved_docs, f)

                        ## save args
                        with open(os.path.join(path, 'args.json'), 'w+') as f:
                            json.dump(vars(args), f)

                        ## save prompt template
                        with open(os.path.join(path, 'prompt_template.txt'), 'w+') as f:
                            f.write(prompt_template)

        else:
            print("-- Skipping evaluation on validation data")
            if args.outdir:
                # save model
                path = os.path.join(
                    args.outdir,
                    '{}_{}ex_{}'.format(args.model, args.n_train_examples,
                                        prompt_template.replace(" ", "_")))

                os.makedirs(path, exist_ok=True)
                torch.save({'model_state_dict': model.state_dict()},
                           os.path.join(path, 'model.pt'))

    print("Finished training retriever!")
    log_metrics(best_all_metrics, best=True)

    ## Loading the best model
    if args.outdir and args.test_data:
        print("Loading the best model and evaluating on test set")
        ckpt = torch.load(os.path.join(path, 'model.pt'))
        model.load_state_dict(ckpt["model_state_dict"])

        # evaluating on test set
        with open(args.test_data, "r") as f:
            test_data = json.load(f)

        print(" -- Evaluating on test set...")
        setattr(args, 'n_eval_examples', None)
        metrics = threaded_eval(args, model,
        test_data,
        tokenizer,
        tfidf_retriever=tfidf_retriever,
        n_threads=args.n_workers)

        ## saving the metrics
        with open(os.path.join(path, 'eval_metrics.json'), 'w+') as f:
            json.dump(metrics, f)

        log_metrics(metrics, test=True)
