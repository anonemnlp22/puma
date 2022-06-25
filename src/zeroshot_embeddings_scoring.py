from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

## loading edge scoring dataset
import csv
from collections import defaultdict
from tqdm import tqdm
import sys
import torch
from argparse import ArgumentParser

sys.path.append("../../../")
from retriever.tfidf_vectorizer_article import TopTfIdf
from nltk.tokenize import sent_tokenize


def get_embeddings(model, text, tokenizer):
    input_ids = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(model.device)
    with torch.no_grad():
        embeddings = model(**input_ids,
                           output_hidden_states=True)[-1][0].mean(dim=1)

    return embeddings


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default='roberta-base')
    parser.add_argument("--eval_data",
                        type=str,
                        default="../data/db_0_2000_bert_edge_scorer.csv")
    parser.add_argument("--tf_idf",
                        action="store_true",
                        help="whether to evaluate using tf-df baseline")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--ensemble_prompts",
                        action='store_true',
                        help="compute ensemble prompt scores")
    parser.add_argument("--load_params", type=str, default=None)
    parser.add_argument("--hop", type=int, choices=[1,2], default=1, help="which hop to use for evaluation")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              cache_dir="/data/cache")
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 cache_dir="/data/cache")
    model = model.to(args.device).eval()

    if args.load_params:
        ckpt = torch.load(args.load_params)
        model.load_state_dict(ckpt["model_state_dict"])

    labels = []
    questions_to_pos_paragraphs = defaultdict(list)
    questions_to_neg_paragraphs = defaultdict(list)

    with open(args.eval_data, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            labels.append(row["label"])
            ll = row["input"].split("</s>")
            if len(ll) == 2 and args.hop == 1:
                q, p = ll
                if int(row["label"]) == 1:
                    questions_to_pos_paragraphs[q].append(p)
                else:
                    questions_to_neg_paragraphs[q].append(p)

            elif len(ll) == 3 and args.hop == 2:
                q, _, p = ll
                if int(row["label"]) == 1:
                    questions_to_pos_paragraphs[q].append(p)
                else:
                    questions_to_neg_paragraphs[q].append(p)

        sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        total_n = 0
        correct_n = 0
        recall = 0

        for q in tqdm(questions_to_pos_paragraphs.keys(), disable=False):
            # encode question
            pos_p = questions_to_pos_paragraphs[q][0]
            neg_para = questions_to_neg_paragraphs[q]
            recalled_correctly = True

            question_embeddings = get_embeddings(model, q, tokenizer)
            pos_paragraph_embeddings = get_embeddings(model, pos_p, tokenizer)
            pos_p_score = sim(question_embeddings,
                              pos_paragraph_embeddings)[0].item()

            for neg_p in neg_para:
                if pos_p == neg_p:
                    continue

                neg_paragraph_embeddings = get_embeddings(
                    model, neg_p, tokenizer)
                neg_p_score = sim(question_embeddings,
                                  neg_paragraph_embeddings)[0].item()
                if pos_p_score > neg_p_score:
                    correct_n += 1
                else:
                    recalled_correctly = False

                total_n += 1

            if recalled_correctly:
                recall += 1

        recall /= len(questions_to_pos_paragraphs.keys())

        print("Acc= {}".format(correct_n / total_n))
        print("Recall@1 negatives= {}".format(recall))