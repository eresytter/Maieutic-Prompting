import pickle
import os
from argparse import ArgumentParser, Namespace

import jsonlines
import torch
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, AutoTokenizer

from inference.inference import InferenceWrapper
from inference.verifier import NLIModel


def parse_args() -> Namespace:
    args = ArgumentParser()
    args.add_argument("--device_id", type=int)
    args.add_argument("--dataset_name", default="Maieutic", type=str)

    args = args.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.data_filename = f"./data/{args.dataset_name}/1_gen/dev.Q.json"
    args.G_filename = f"./data/{args.dataset_name}/1_gen/dev.G.pkl"

    return args


if __name__ == "__main__":
    args = parse_args()
    current_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_path, "models/roberta-large-mnli/")
    model = RobertaForSequenceClassification.from_pretrained(model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    InferenceWrapper.nli_model = NLIModel(model, tokenizer)
    
    with jsonlines.open(os.path.join(current_path, args.data_filename), "r") as f:
        samples = list(f)

    with open(os.path.join(current_path, args.G_filename), "rb") as f:
        G_samples = pickle.load(f)

    acc_result = [0, 0]  # [correct, incorrect]
    for sample, G in tqdm(zip(samples, G_samples), total=len(samples)):
        if G.size() == 1:
            inferred_answer = 1 if G["Q"].data["blf"][0] >= G["Q"].data["blf"][1] else -1
        elif G.size() > 1:
            score_list, correct_E_dict, graph2sat, belief, consistency = InferenceWrapper.infer(G)
            # score_list: correct statements and their NLI relationships with Q.
            # correct_E_dict: correct statements given by the SAT solver.
            # graph2sat: a mapping from node identifiers to ids in the SAT solver.
            # belief: unary clause for SAT, computed from equation (7), page 4.
            # consistency: binary clause for SAT, computed from section 3.4, page 4.
            sum_score = sum([score[1] for score in score_list])
            inferred_answer = 1 if sum_score >= 0 else -1
        else:
            inferred_answer = 1

        # Record results
        gt_answer = 1 if sample["A"] else -1
        acc_result[0 if inferred_answer == gt_answer else 1] += 1

    print(f"Acc: {acc_result[0] / (acc_result[0] + acc_result[1]) * 100:.1f}")
