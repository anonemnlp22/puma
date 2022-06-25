from tqdm import tqdm
import time, os
from .prompt_utils import shorten_paragraph_if_necessary
from .prompt_utils import (create_prompt, score_prompt_batch,
                        shorten_paragraph_if_necessary, prepend_title_doc,
                        create_qa_prompt, trim_passages_for_reader, score_paths, all_prompt_templates)
import threading, queue
import numpy as np
import pickle
from transformers import GPT2TokenizerFast, AutoTokenizer
from torch.utils.data.dataloader import default_collate
from rank_bm25 import BM25Okapi, BM25Plus, BM25L
import torch
from .qa_eval import evaluate_answers
import regex
import string
import unicodedata
from retriever.tfidf_vectorizer_article import TopTfIdf
from scipy.stats import entropy
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict

def evaluate_retrieval(
    args,
    model,
    eval_data,
    tokenizer,
    tfidf_retriever,
    get_retrieved=False,
):
    """
    Evaluates the model on the given evaluation data
    """
    model.eval()

    batch_size = args.eval_batch_size
    eval_mode = args.eval_mode
    tfidf_pool_size = args.tfidf_pool_size
    n_examples = getattr(args, "n_eval_examples", None)
    use_bm25 = getattr(args, "use_bm25", False)
    prompt_template = getattr(args, "prompt_template", getattr(args, "reranker_prompt_template", None))

    max_prompt_len = getattr(args, "max_prompt_len", getattr(args, "reranker_max_prompt_len", None))

    REC_AT_2 = 0.0
    REC_AT_10 = 0.0
    REC_AT_20 = 0.0
    REC_AT_50 = 0.0

    ANS_RECALL_AT_10 = 0.0
    ANS_RECALL_AT_2 = 0.0
    ANS_RECALL_AT_20 = 0.0
    ANS_RECALL_AT_50 = 0.0
    ANS_RECALL_AT_100 = 0.0

    N_BRIDGE_QUESTIONS = 0

    if n_examples is not None:
        eval_data = eval_data[:n_examples]

    all_retrieved_docs = []

    top_k_first_hops = args.top_k_first_hops
    top_k_second_hops = args.top_k_second_hops


    tfidf_pruner = TopTfIdf(n_to_select=top_k_second_hops,
                          filter_dist_one=True,
                          rank=True)


    for ex in tqdm(eval_data):

        if not ex["question"].endswith("?"):
            ex["question"] += "?"

        question = ex["question"]
        answer = ex["answer"]
        qid = ex["_id"]
        qtype = ex["type"]


        if "gold_path_titles" in ex:
            gold_docs = ex["gold_path_titles"]
        else:
            gold_docs = []
            for title, _ in ex["supporting_facts"]:
                if title not in gold_docs:
                    gold_docs.append(title)

        assert len(gold_docs) > 0, "No gold documents found for question {}".format(question)

        if "retrieved_docs_text" not in ex:  # if we haven't already pre-retrieved docs
            tfidf_paragraphs, _ = tfidf_retriever.ranker.closest_docs(
                question, tfidf_pool_size)

            retrieved_titles = [
                t.replace("_0", "") for t in tfidf_paragraphs
            ]
            retrieved_docs_text = tfidf_retriever.load_abstract_para_text(
                retrieved_titles)
            retrieved_docs_text = list(retrieved_docs_text.items())

        else:  # use pre-retrieved docs
            retrieved_docs_text = ex["retrieved_docs_text"]
            retrieved_docs_text = retrieved_docs_text[:tfidf_pool_size]

        if use_bm25:
            ## rerank with bm25
            tokenized_corpus = [
                word_tokenize(t[1]) for t in retrieved_docs_text
            ]
            bm25_index = BM25Okapi(tokenized_corpus, k1=2 , b=0.75)
            doc_scores = bm25_index.get_scores(word_tokenize(question))

            ## sort by score
            doc_scores = sorted(zip(doc_scores, retrieved_docs_text),
                                key=lambda x: x[0],
                                reverse=True)
            retrieved_docs_text = [t[1] for t in doc_scores]


        retrieved_docs_text = dict(retrieved_docs_text)
        title_score = {}
        title_to_text = {}

        if args.scoring_method == "tfidf":
            retrieved_titles = [
                t[0].replace("_0", "")
                for t in retrieved_docs_text.items()
            ]
            title_to_text = retrieved_docs_text
            scores = np.linspace(0.1, 1.0, num=len(retrieved_titles))# actual tfidf score don't matter
            scores = scores[::-1]
            title_score = dict(zip(retrieved_titles, scores))
        else:
            retrieved_docs_text_dict = {}
            for doc_title, doc_text in retrieved_docs_text.items():
                if not (isinstance(doc_text, str) and doc_text.strip()):
                    continue

                doc_text = shorten_paragraph_if_necessary(
                    doc_text, args.max_doc_len, tokenizer=tokenizer)

                doc_title = doc_title.replace("_0", "")
                if args.prepend_title:
                    doc_text = prepend_title_doc(doc_text, doc_title)

                retrieved_docs_text_dict[doc_title] = doc_text

            retrieved_docs_text = list(
                t for t in retrieved_docs_text_dict.items()
                if isinstance(t[1], str))  # ignore empty docs

            title_to_text.update(dict(retrieved_docs_text))

            # score docs
            for j in range(0, len(retrieved_docs_text), batch_size):

                cur_batch = retrieved_docs_text[j:j + batch_size]
                batch_titles = [t[0] for t in cur_batch]
                batch_texts = [t[1] for t in cur_batch]

                batch_scores = score_paths(model ,
                prompt_template=prompt_template,
                question=question,
                path_list= [[t] for t in batch_texts],
                tokenizer=tokenizer,
                max_prompt_len=max_prompt_len,
                temperature=args.temperature,
                ensembling_method=args.ensembling_method,)

                for score, title in zip(batch_scores, batch_titles):
                    title_score[title.replace("_0", "")] = score

            ## sort by score
            assert len(title_score) == len(retrieved_docs_text)

            title_score_sorted = sorted(
                title_score.items(), key=lambda x: x[1],
                reverse=True)

            scores= [t[1] for t in title_score_sorted]
            #print("single-hop scores stats: mean={}, std={}".format(np.mean(scores), np.std(scores)))

            retrieved_titles = [t[0] for t in title_score_sorted]

            ## multi-hop prompts
            do_expand = True if args.eval_mode == "hotpot" else False

            if do_expand: ## add linked docs from all retrieved docs
                linked_title_to_text = ex["all_linked_paras_dic"]
                multi_prompts = []
                multi_prompts_titles = []

                ## NOTE TO REMOVE: expand tfidf without re-ranking first hops
                #retrieved_titles = [t[0] for t in retrieved_docs_text]

                for first_hop_title in retrieved_titles[:
                                                        top_k_first_hops]:  ## only expand top k2 docs
                    first_hop_text = title_to_text[first_hop_title]

                    first_hop_text = shorten_paragraph_if_necessary(first_hop_text, args.max_doc_len, tokenizer=tokenizer)
                    linked_paragraph_titles = ex["all_linked_para_title_dic"][first_hop_title]
                    linked_paragraph_texts = [linked_title_to_text[t] for t in linked_paragraph_titles]

                    ## select top k links using tfidf pruner
                    rank_idx = tfidf_pruner.prune(question=question,
                                                            paragraphs=linked_paragraph_texts,
                                                            return_scores=False)

                    rank_idx = rank_idx[:top_k_second_hops]
                    linked_paragraph_titles = [linked_paragraph_titles[i] for i in rank_idx]

                    ## add self to the linked paragraphs to allow retrieval of single hop
                    #import ipdb; ipdb.set_trace()
                    #linked_paragraph_titles.insert(0, first_hop_title)
                    #linked_title_to_text[first_hop_title] = ""

                    for t in linked_paragraph_titles:
                        second_hop_title = t.replace("_0","")
                        second_hop_text = linked_title_to_text[t]

                        ## shorten
                        second_hop_text = shorten_paragraph_if_necessary(second_hop_text, args.max_doc_len, tokenizer=tokenizer)
                        assert isinstance(second_hop_text, str)

                        title_to_text[second_hop_title] = linked_title_to_text[t]
                        
                        if getattr(args, "reverse_path", False):
                            multi_prompts.append(
                                [second_hop_text, first_hop_text])
                            multi_prompts_titles.append((second_hop_title, first_hop_title))
                        else:
                            multi_prompts.append([first_hop_text, second_hop_text])
                            multi_prompts_titles.append((first_hop_title, second_hop_title))

                #print("we have {} multi-hop prompts".format(len(multi_prompts)))

                scores_mutlihop_titles = {}
                if len(multi_prompts) > 0:
                    for j in range(0, len(multi_prompts), batch_size):
                        paths_batch = multi_prompts[j:j + batch_size]
                        batch_titles = multi_prompts_titles[j:j + batch_size]

                        batch_scores = score_paths(
                            model,
                            prompt_template=prompt_template,
                            question=question,
                            path_list=paths_batch,
                            tokenizer=tokenizer,
                            max_prompt_len=max_prompt_len,
                            temperature=args.temperature,
                        )

                        for i, (t1, t2) in enumerate(batch_titles):
                            scores_mutlihop_titles[(t1, t2)] = batch_scores[i]     # score of full path

                    # printing scores stats (mean and std)
                    #scores_mutlihop_titles_list = list(scores_mutlihop_titles.values())
                    #print("multi-hop scores stats: mean={}, std={}".format(np.mean(scores_mutlihop_titles_list), np.std(scores_mutlihop_titles_list)))

                    #if qtype == "comparison":
                    #    import ipdb; ipdb.set_trace()
                    ## sort by multi-hop prompt score
                    scores_mutlihop_titles = sorted(scores_mutlihop_titles.items(), key=lambda x: x[1], reverse=True)

                    retrieved_titles = []
                    #for k in title_score:
                    #    title_score[k] = np.exp(title_score[k])

                    for (title1, title2), score in scores_mutlihop_titles:
                        for t in [title1, title2]:
                            title_score[t] = max(title_score.get(t, -1e5), score)
                            #title_score[t] = title_score.get(t, 0) + np.exp(score)
                            #if t not in retrieved_titles:
                            #    retrieved_titles.append(t)

                    title_score_sorted = sorted(
                        title_score.items(), key=lambda x: x[1],
                        reverse=True)
                    retrieved_titles = [t[0] for t in title_score_sorted]

        if set(retrieved_titles[:2]) & set(gold_docs) == set(gold_docs):
            REC_AT_2 += 1

        if set(retrieved_titles[:10]) & set(gold_docs) == set(gold_docs):
            REC_AT_10 += 1

        if set(retrieved_titles[:20]) & set(gold_docs) == set(gold_docs):
            REC_AT_20 += 1

        if set(retrieved_titles[:50]) & set(gold_docs) == set(gold_docs):
            REC_AT_50 += 1

        ## answer recall
        if qtype == "bridge":
            N_BRIDGE_QUESTIONS += 1
            for i, title in enumerate(retrieved_titles):
                try:
                    cur_paragraph_text = title_to_text[title + '_0']
                except KeyError:
                    cur_paragraph_text = title_to_text[title]

                if isinstance(cur_paragraph_text, list):
                    cur_paragraph_text = " ".join(cur_paragraph_text).strip()
                if has_answer(cur_paragraph_text, answer):
                    if i < 50:
                        ANS_RECALL_AT_50 += 1
                    if i < 20:
                        ANS_RECALL_AT_20 += 1
                    if i < 10:
                        ANS_RECALL_AT_10 += 1
                    if i < 2:
                        ANS_RECALL_AT_2 += 1
                    break

        if get_retrieved:
            retrieved_text_title = []
            for title in retrieved_titles:
                doc = {}
                doc["title"] = title.replace("_0", "")
                try:
                    cur_paragraph_text = title_to_text[title + '_0']
                except KeyError:
                    cur_paragraph_text = title_to_text[title]
                doc['text'] = cur_paragraph_text
                doc['score'] = title_score[doc["title"]]
                retrieved_text_title.append(doc)

            d = {
                "id": qid,
                "question": question,
                "answer": answer,
                "reranked_docs": retrieved_text_title
            }
            all_retrieved_docs.append(d)

    metrics = {
        "REC_AT_2": REC_AT_2 / len(eval_data),
        "REC_AT_10": REC_AT_10 / len(eval_data),
        "REC_AT_20": REC_AT_20 / len(eval_data),
        "REC_AT_50": REC_AT_50 / len(eval_data),
        "ANS_RECALL_AT_10": ANS_RECALL_AT_10 / N_BRIDGE_QUESTIONS,
        "ANS_RECALL_AT_2": ANS_RECALL_AT_2 / N_BRIDGE_QUESTIONS,
        "ANS_RECALL_AT_20": ANS_RECALL_AT_20 / N_BRIDGE_QUESTIONS,
        "ANS_RECALL_AT_50": ANS_RECALL_AT_50 / N_BRIDGE_QUESTIONS,
        "REC_AT_2_count": REC_AT_2,
        "REC_AT_10_count": REC_AT_10,
        "REC_AT_20_count": REC_AT_20,
    }
    if get_retrieved:
        metrics["questions_docs"] = all_retrieved_docs

    return metrics


def retrieve_docs_fid(args, model, eval_data, tokenizer, tfidf_retriever):
    '''
    retrieves docs in FiD (Izacard and Grave, 2021) format
    returns  
        {
        'id': '0',
        'question': 'What element did Marie Curie name after her native land?',
        'target': 'Polonium',
        'answers': ['Polonium', 'Po (chemical element)', 'Po'],
        'ctxs': [
                {
                    "title": "Marie Curie",
                    "text": "them on visits to Poland. She named the first chemical element that she discovered in 1898 \"polonium\", after her native country. Marie Curie died in 1934, aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, of aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I. Maria Sk\u0142odowska was born in Warsaw, in Congress Poland in the Russian Empire, on 7 November 1867, the fifth and youngest child of well-known teachers Bronis\u0142awa, \"n\u00e9e\" Boguska, and W\u0142adys\u0142aw Sk\u0142odowski. The elder siblings of Maria"
                },
                {
                    "title": "Marie Curie",
                    "text": "was present in such minute quantities that they would eventually have to process tons of the ore. In July 1898, Curie and her husband published a joint paper announcing the existence of an element which they named \"polonium\", in honour of her native Poland, which would for another twenty years remain partitioned among three empires (Russian, Austrian, and Prussian). On 26 December 1898, the Curies announced the existence of a second element, which they named \"radium\", from the Latin word for \"ray\". In the course of their research, they also coined the word \"radioactivity\". To prove their discoveries beyond any"
                }
                ]
        }
    '''
    metrics = evaluate_retrieval(args,
                                 model,
                                 eval_data,
                                 tokenizer,
                                 tfidf_retriever,
                                 get_retrieved=True)

    retrieved_stuff = metrics["retrieved_stuff"]

    rets = []
    for d in retrieved_stuff:
        question = d["question"]
        answer = d["answer"]
        text_title = d["retrieved_text_title"]
        qid = d["id"]

        ctxs = []
        for title in text_title:
            text = text_title[title]
            ctxs.append({"title": title, "text": text})

        rets.append({
            "id": qid,
            "question": question,
            "target": answer,
            "answers": [answer],
            "ctxs": ctxs
        })

    return rets


def retrieve_docs(args, model, eval_data, tokenizer, tfidf_retriever):
    docs_metrics = evaluate_retrieval(args,
                                 model,
                                 eval_data,
                                 tokenizer,
                                 tfidf_retriever,
                                 get_retrieved=True)

    return docs_metrics


def eval_retrieval_queue(args, model, eval_data, tokenizer, tfidf_retriever,
                         queue):
    metrics = evaluate_retrieval(args, model, eval_data, tokenizer,
                                 tfidf_retriever)
    queue.put(metrics)


def threaded_eval(args,
                  model,
                  eval_data,
                  tokenizer,
                  tfidf_retriever,
                  n_threads=4):

    ##
    q = queue.Queue()

    if n_threads == 1:
        metrics = evaluate_retrieval(args, model, eval_data, tokenizer,
                                     tfidf_retriever, get_retrieved=True)
        return metrics

    if args.n_eval_examples:
        eval_data = eval_data[:args.n_eval_examples]
        setattr(args, "n_eval_examples", None)

    split_data = np.array_split(eval_data, n_threads)
    threads = []
    for i in range(n_threads):
        cur_model = pickle.loads(pickle.dumps(model)).to(i % 4)
        ## split data
        ws_dir = os.environ[
            "WORKSPACE"] if "WORKSPACE" in os.environ else "/data"

        tokenizer = AutoTokenizer.from_pretrained(args.model,
                                                  cache_dir=os.path.join(
                                                      ws_dir, "cache"))
        tokenizer.pad_token = tokenizer.eos_token
        t = threading.Thread(target=eval_retrieval_queue,
                             args=(args, cur_model, split_data[i], tokenizer,
                                   tfidf_retriever, q))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    ##

    metrics = []
    while not q.empty():
        metrics.append(q.get())

    keys = metrics[0].keys()

    ## average metrics accross threads
    final_metrics = {}
    for k in keys:
        if "count" in k:
            values = [m[k] for m in metrics]
            newk = k.replace("_count", "")
            final_metrics[newk] = sum(values) / len(eval_data)

    return final_metrics


def evaluate_reader_gold(args, model, eval_data, tokenizer):
    '''
    Evaluates reader given GOLD passages -- cannot be used when using 
    reranked passages.
    '''
    model.eval()

    batch_size = args.eval_batch_size
    eval_mode = args.eval_mode
    prompt_template = getattr(args, "prompt_template", getattr(args, "reader_prompt_template", None))

    if eval_mode == "hotpot":

        gold_answers = []
        gen_answers = []

        for ex in tqdm(eval_data, disable=os.environ.get("NO_TQDM", False)):
            question = ex["question"]
            answer = ex["answer"]

            gold_docs_texts = ex["gold_path_texts"]
            prompt = create_qa_prompt(question, gold_docs_texts, prompt_template)

            if args.append_support_facts:
                title_to_text = dict(zip(ex["gold_path_titles"], ex["gold_path_texts"]))
                sf =  ex["supporting_facts"]

                sf_idx = defaultdict(list)
                for title, idx in sf:
                    sf_idx[title].append(idx) ## TODO needs checking

                sf = []
                for title in sf_idx:
                    tokenized_passage = sent_tokenize(title_to_text[title])
                    for idx in sf_idx[title]:
                        if idx >= len(tokenized_passage):
                            continue
                        sf.append(tokenized_passage[idx])

                sf = " ".join(sf)
                prompt += " " + sf + " Answer:"

            if not prompt.endswith("Answer:"):
                prompt += " Answer:"

            generated_answer = generate_answer(args, model, tokenizer, prompt)
            if isinstance(generated_answer, list):
                generated_answer = generated_answer[0]
            generated_answer = process_answer(generated_answer)
            gold_answers.append(answer)
            gen_answers.append(generated_answer)

        return evaluate_answers(gen_answers, gold_answers)


def generate_answer_greedy(args, model, tokenizer, prompt):
    model.eval()

    max_len = getattr(args, "max_prompt_len",
                      getattr(args, "reader_max_prompt_len", None))
    max_ans_len = args.max_ans_len

    max_len = min(max_len - max_ans_len, len(tokenizer.encode(prompt)))
    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       max_length=max_len,
                       truncation=True).to(model.device)

    input_len = inputs.input_ids.size(1)

    gen_output = model.generate(**inputs,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_new_tokens=max_ans_len,
                                min_length=input_len + 1,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                do_sample=False,
                                num_beams=1,)


    ans_tokens = gen_output.sequences[0][input_len:]
    answer = tokenizer.decode(ans_tokens, skip_special_tokens=True).strip()

    return answer


def generate_answer(args, model, tokenizer, prompt):
    model.eval()

    max_len = getattr(args, "max_prompt_len", getattr(args, "reader_max_prompt_len", None))
    max_ans_len = args.max_ans_len
    temp = getattr(args, "reader_temperature", 0.05)
    topk = getattr(args, "reader_topk_sampling", None)
    do_sample = True if topk is not None else False

    num_return_sequences = 1 if getattr(args, "self_consistency_sampling", None) is None else args.self_consistency_sampling
    if num_return_sequences > 1:
        assert topk is not None, "Must specify topk for self-consistency sampling"

    max_len = min(max_len - max_ans_len, len(tokenizer.encode(prompt)))
    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       max_length=max_len,
                       truncation=True).to(model.device)

    input_len = inputs.input_ids.size(1)


    gen_output = model.generate(**inputs,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_new_tokens=max_ans_len,
                                min_length=input_len + 1,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                temperature=temp,
                                do_sample=do_sample,
                                top_k=topk,
                                num_return_sequences=num_return_sequences)

    samples = []
    for ans in gen_output.sequences:
        ids = ans[input_len:]
        samples.append(tokenizer.decode(ids, skip_special_tokens=True))

    #ans_tokens = gen_output.sequences[0][input_len:]
    #answer = tokenizer.decode(ans_tokens, skip_special_tokens=True).strip()

    return samples


def generate_answer_with_confidence(args, model, tokenizer, prompt):
    model.eval()
    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       max_length=1024,
                       truncation=True).to(model.device)

    max_ans_len = args.max_ans_len
    input_len = inputs.input_ids.size(1)

    gen_output = model.generate(**inputs,
                                return_dict_in_generate=True,
                                output_scores=True,
                                max_new_tokens=max_ans_len,
                                min_length=input_len + 1,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                temperature=0.05,
                                num_beams=3)

    ans_tokens = gen_output.sequences[0][input_len:]
    answer = tokenizer.decode(ans_tokens, skip_special_tokens=True).strip()

    ## obtaining answer token logprobs
    ans_logprobs = []

    vocab_logits = torch.stack(gen_output.scores[:len(ans_tokens)],
                               dim=0)  # (|a|, batch_size, |V|)
    ## selecting first element
    vocab_logits = vocab_logits[:, 0, :]
    vocab_probs = torch.softmax(vocab_logits, dim=-1)
    vocab_logprobs = torch.log_softmax(vocab_logits, dim=-1)

    ## selecting logprobs of answer tokens
    ans_logprobs = vocab_logprobs[range(len(ans_tokens)),
                                  ans_tokens].cpu().numpy()  # (|a|)
    confidence = {"first_token_logprob": ans_logprobs[0]}
    all_ans_logprob = sum(ans_logprobs)
    confidence["ans_logprob"] = all_ans_logprob
    confidence["avg_ans_logprob"] = all_ans_logprob / len(ans_tokens)

    vocab_probs = vocab_probs.cpu().numpy()
    ## average entropy
    ent = 0
    for i in range(vocab_probs.shape[0]):
        ent += entropy(vocab_probs[i, :], base=2)

    ent /= vocab_logits.shape[0]
    confidence["avg_entropy"] = -ent

    return answer, confidence


def answer_questions(args, model, data, tokenizer):
    '''
    generates answers for questions in data given the reranked documens in data

    data = {

        "id": "",
        "question": "",
        "reranked_docs": [ {"title": "", "text": ""}, ... ]
    }

    returns  {
        "id": "",
        "question": "",
        "predicted_answer": "",
        "gold_answer": "" (if available)

    }
    '''
    model.eval()

    n_total = args.self_consistency_sampling
    predictions = []

    for ex in tqdm(data):
        if not ex["question"].endswith("?"):
            ex["question"] += "?"

        question = ex["question"]
        gold_answer = ex["answer"] if "answer" in ex else None
        reranked_docs = ex["reranked_docs"]

        ## sort by score
        reranked_docs = sorted(reranked_docs, key=lambda x: x["score"])

        passages = [d['text'] for d in reranked_docs]
        if args.prepend_title:
            #TODO
            pass

        max_prompt_len = getattr(args, "max_prompt_len", getattr(args, "reader_max_prompt_len", None))

        if args.reader_top_n_passages is not None: # feed as many passages as possible
            passages = passages[-args.reader_top_n_passages:]


        letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        if args.symbolic_prompt:
            ## split passages to sentences and label them
            n=1
            new_passages = []
            for passage in passages:
                org_sents = sent_tokenize(passage)
                new_sents = []
                for sent in org_sents:
                    new_sents.append(f"{letters[n-1]}: {sent}")
                    n+=1
                new_passages.append(" ".join(new_sents))

            passages = new_passages

        #import ipdb; ipdb.set_trace()

        trimmed_docs = trim_passages_for_reader(question, passages, max_prompt_len,
        args.max_doc_len, tokenizer)

        prompt = create_qa_prompt(question, trimmed_docs, args.reader_prompt_template, passage_prefix="" if args.symbolic_prompt else "Passage: ")
        #import ipdb;ipdb.set_trace()

        #import ipdb; ipdb.set_trace()

        if args.self_consistency_sampling is None:
            answers = generate_answer(args, model, tokenizer, prompt)
            #import ipdb; ipdb.set_trace()
            generated_answer = answers[0]
            generated_answer = process_answer(generated_answer)

        else: # self-consistency sampling
            bsz = 1
            answers= []
            setattr(args, "self_consistency_sampling", bsz)
            for _ in range(n_total // bsz):
                ans = generate_answer(args, model, tokenizer, prompt)
                answers.extend(ans)

            #import ipdb; ipdb.set_trace()
            answers = [process_answer(a) for a in answers]
            ans_to_count = {}
            for ans in answers:
                ans_to_count[ans] = ans_to_count.get(ans, 0) + 1
            ## select answer with highest count
            max_count = max(ans_to_count.values())
            generated_answer = [ans for ans, count in ans_to_count.items() if count == max_count][0]

        #import ipdb; ipdb.set_trace()


        ex.update({
            "predicted_answer": generated_answer,
            "gold_answer": gold_answer,
            'passages': trimmed_docs,
            'prompt': prompt,
            'question': question,
        })

        predictions.append(ex)

    return predictions


def process_answer(answer):

    ## yes, no answers
    if answer.lower().startswith('yes'):
        answer = 'yes'

    if answer.lower().startswith('no'):
        answer = 'no'

    if 'The answer is' in answer:
        answer = answer.split('The answer is')[1]
        answer = answer.split('.')[0]

    if 'the answer is' in answer:
        answer = answer.split('the answer is')[1]
        answer = answer.split('.')[0]

    if 'Question' in answer:
        answer = answer.split('Question')[0]

    if 'Passage' in answer:
        answer = answer.split('Passage')[0]

    if 'Answer' in answer:
        answer = answer.split('Answer')[1]

    if 'Answer:' in answer:
        answer = answer.split('Answer:')[1]

    if ' = ' in answer:
        answer = answer.split(' = ')[1]

    if 'XX' in answer:
        answer = answer.split('XX')[0]



    answer = answer.strip()

    if answer.endswith('.'):
        answer = answer[:-1]

    return answer

def answer_questions_rias(args, model, data, tokenizer):
    '''
    Ranking-informed answer selection
    for each retrieved passage, generate an answer, and then rank the answers depending on passage score and answer confidence

    data = {

        "id": "",
        "question": "",
        "reranked_docs": [ {"title": "", "text": ""}, ... ]
    }

    returns  {
        "id": "",
        "question": "",
        "predicted_answer": "",
        "gold_answer": "" (if available)

    }
    '''
    model.eval()

    predictions = []
    for ex in tqdm(data):
        question = ex["question"]
        gold_answer = ex["answer"] if "answer" in ex else None
        reranked_docs = ex["reranked_docs"]

        ## sort by score in descending order
        reranked_docs = sorted(reranked_docs, key=lambda x: x["score"], reverse=True)
        K = 10
        reranked_docs = reranked_docs[:K]

        passages = [d['text'] for d in reranked_docs]
        passage_scores = [d['score'] for d in reranked_docs]

        max_prompt_len = getattr(args, "max_prompt_len",
                                 getattr(args, "reader_max_prompt_len", None))

        p_score, a_conf, answers = [], [] ,[]
        conf_measure = args.rias_conf_measure
        ps_wt = args.rias_passage_wt
        ac_wt = args.rias_answer_wt

        ## windows of passages
        #window_size = args.rias_window_size
        #if 'tsunami' in question:
        #    import ipdb; ipdb.set_trace()

        for p, sc in zip(passages, passage_scores):
            p = shorten_paragraph_if_necessary(p, args.max_doc_len, tokenizer)
            prompt = create_qa_prompt(question, [p],
                                    args.reader_prompt_template)

            generated_answer, conf = generate_answer_with_confidence(args, model, tokenizer, prompt)
            generated_answer = process_answer(generated_answer)
            p_score.append(sc)
            conf = conf[conf_measure]
            a_conf.append(conf)
            answers.append(generated_answer)

        answers_scores = [ps_wt*p + ac_wt*a for p, a in zip(p_score, a_conf)]
        best_answer = answers[np.argmax(answers_scores)]


        ex.update({
            "predicted_answer": best_answer,
            "gold_answer": gold_answer
        })

        predictions.append(ex)

    return predictions


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def has_answer(text, answers):
    """
    Adapted from DPR: https://github.com/facebookresearch/DPR
    """

    if isinstance(answers, str):
        answers = [answers]

    tokenizer = SimpleTokenizer()
    text = _normalize(text)

    tokenizer = SimpleTokenizer()
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def _normalize(text):
    return unicodedata.normalize('NFD', text)

#Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))