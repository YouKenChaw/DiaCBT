import argparse
import json
import os
from tqdm import tqdm
from glob import glob
import re
import ast

from langchain_core.prompts import PromptTemplate

from code.utils.args import print_args
from code.utils.files import load_prompt
from code.helpers.backbone import Backbone
from code.examples import PATTERN_DEF, INTAKE_FORM_EN_EXAMPLE, INTAKE_FORM_ZH_EXAMPLE


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--result_dir", type=str, default="./outputs/results")
    parser.add_argument("--save_dir", type=str, default="./outputs/questionnaires")
    parser.add_argument("--dataset", type=str, required=True, choices=["C2D2", "PatternReframe"])
    parser.add_argument("--agent", type=str, required=True)
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--questionnaire", type=str, required=True, choices=["panas_before", "panas_after", "CTRS"])
    parser.add_argument("--npc_backbone", type=str, default="gpt-4o-mini")
    return parser.parse_args()


def load_and_prepare_data(args):
    with open(os.path.join(args.data_dir, args.dataset, "test.json"), mode="r", encoding="utf-8") as data_file:
        data = json.load(data_file)

    if args.questionnaire == "panas_before":
        return data
    
    with open(os.path.join(args.result_dir, args.agent, args.dataset, f"{args.backbone}_records.txt"), "r") as file:
        records = file.read().split("\n\n")[:-1]

    if args.questionnaire == "CTRS":
        outputs = []
        for record in records:
            dialogue = ast.literal_eval(record)["dialogue"]
            dialogue = "\n".join([f"{item[0]}: {item[-1]}" for item in dialogue])
            outputs.append(dialogue)
        return outputs

    all_model_data = []
    for idx in range(len(data)):
        with open(os.path.join(args.save_dir, "panas_before", args.dataset, f"{idx + 1}.json"), "r") as js_file:
            intake_form = json.load(js_file)["intake_form"]
        record = ast.literal_eval(records[idx])["dialogue"]
        record = "\n".join([f"{item[0]}: {item[-1]}" for item in record])
        all_model_data.append(dict(intake_form=intake_form, record=record))
    return all_model_data


def questionnaire_panas_before(args, data, save_path):
    prompt = PromptTemplate.from_template(load_prompt(role="scales", name="panas_before"))
    backbone = Backbone(args.npc_backbone, temperature=0.0, n=1)
    generation_model = prompt | backbone

    intake_prompt = PromptTemplate.from_template(load_prompt(role="npc", name="intake")).partial(example=INTAKE_FORM_ZH_EXAMPLE if args.dataset == "C2D2" else INTAKE_FORM_EN_EXAMPLE)
    intake_generation_model = intake_prompt | backbone

    pbar = tqdm(total=len(data), desc="Questionnaire...")
    for idx, profile in enumerate(data):
        intake_form = intake_generation_model.invoke({
            "persona": profile["场景"] if args.dataset == "C2D2" else profile["persona"],
            "thought": profile["思维"] if args.dataset == "C2D2" else profile["thought"],
            "patterns": f"{profile['标签']}: {PATTERN_DEF[profile['标签']]}" if args.dataset == "C2D2" else profile["pattern_def"]
        })
        response = generation_model.invoke({"intake_form": intake_form})
        
        result = dict(intake_form=intake_form, prediction=response)
        with open(os.path.join(save_path, f"{idx + 1}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        pbar.update()
    
    print("\n>>>>>> The questionnaire result is saved.")


def questionnaire_panas_after(args, data, save_path):
    prompt = PromptTemplate.from_template(load_prompt(role="scales", name="panas_after"))
    backbone = Backbone(args.npc_backbone, temperature=0.0, n=1)
    generation_model = prompt | backbone

    pbar = tqdm(total=len(data), desc="Questionnaire...")
    for idx, model_data in enumerate(data):
        response = generation_model.invoke({"intake_form": model_data["intake_form"], "dialogue": model_data["record"]})

        result = dict(prediction=response)
        with open(os.path.join(save_path, f"{idx + 1}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        pbar.update()
    
    print("\n>>>>>> The questionnaire result is saved.")


def questionnaire_CTRS(args, data, save_path):
    criteria_list = ["understanding", "interpersonal_effectiveness", "collaboration", "guided_discovery", "focus", "strategy"]
    for criteria in criteria_list:
        results = []

        prompt = PromptTemplate.from_template(load_prompt(role="scales", name=criteria))
        backbone = Backbone(args.npc_backbone, temperature=0.0, n=1)
        generation_model = prompt | backbone

        pbar = tqdm(total=len(data), leave=False, desc=f"Questionnaire CTRS-{criteria}")
        for idx, model_data in enumerate(data):
            response = generation_model.invoke({"conversation": model_data})
            results.append({"idx": idx + 1, "score": response})
            pbar.update()
        
        with open(os.path.join(save_path, f"{criteria}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"\n>>>>>> The questionnaire result of {criteria} is saved.")


def calculate_CTRS_score(save_path):
    total_results = {}
    criteria_list = ["understanding", "interpersonal_effectiveness", "collaboration", "guided_discovery", "focus", "strategy"]
    for criteria in criteria_list:
        with open(os.path.join(save_path, f"{criteria}.json"), "r") as f:
            dataset = json.load(f)
        
        avg_score = 0
        for data in dataset:
            try:
                score = int(data["score"].split(",")[0])
            except Exception as e:
                print(e)
                score = 0
            avg_score += score
        
        total_results[criteria] = avg_score / len(dataset)
    
    print(">>>>>>\n")
    for criteria in criteria_list:
        print(f"{criteria}: {round(total_results[criteria], 4)}")


def calculate_panas_score(save_path):
    criteria_list = ["Interested", "Excited", "Strong", "Enthusiastic", "Proud", "Alert", "Inspired", "Determined", "Attentive", "Active", "Distressed", "Upset", "Guilty", "Scared", "Hostile", "Irritable", "Ashamed", "Nervous", "Jittery", "Afraid"]
    score_dict = {}
    for cri in criteria_list:
        score_dict[cri] = []

    data_files = glob(os.path.join(save_path, "*.json"))
    for file_path in data_files:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        score_lines = data["prediction"].split("\n")
        for line in score_lines:
            if not line:
                continue
            criteria = line.split(",")[0].strip()
            score = int(re.findall(r'\d+', line.split(",")[-1].strip())[0])
            if criteria in criteria_list:
                score_dict[criteria].append(score)
    
    avg_score_dict = {}
    for key in score_dict.keys():
        avg_score_dict[key] = sum(score_dict[key]) / len(score_dict[key])
    
    positive_score = []
    for key in criteria_list[:10]:
        positive_score.append(avg_score_dict[key])
    
    negative_score = []
    for key in criteria_list[10:]:
        negative_score.append(avg_score_dict[key])
    
    postive_criteria_score = sum(positive_score) / len(positive_score)
    negative_criteria_score = sum(negative_score) / len(negative_score)
    print(f"\n>>>>>> postive_scrteria_score: {postive_criteria_score}, negative_criteria_score: {negative_criteria_score}")


def main():
    args = setup_args()
    print_args(args)

    all_model_data = load_and_prepare_data(args)
    if args.questionnaire == "panas_before":
        save_path = os.path.join(args.save_dir, args.questionnaire, args.dataset)
    else:
        save_path = os.path.join(args.save_dir, args.questionnaire, args.agent, args.backbone, args.dataset)
    if os.path.exists(save_path):
        print("The save path already exists. If you want to re-questionnaire, please change the save path.")
        if "panas" in args.questionnaire:
            calculate_panas_score(save_path)
        elif "CTRS" in args.questionnaire:
            calculate_CTRS_score(save_path)
        else:
            raise ValueError("Invalid questionnaire.")
        return
    os.makedirs(save_path)

    if args.questionnaire == "panas_before":
        questionnaire_panas_before(args, all_model_data, save_path)
    elif args.questionnaire == "panas_after":
        questionnaire_panas_after(args, all_model_data, save_path)
    elif args.questionnaire == "CTRS":
        questionnaire_CTRS(args, all_model_data, save_path)
    else:
        raise ValueError("Invalid questionnaire.")
    
    if "panas" in args.questionnaire:
        calculate_panas_score(save_path)
    elif "CTRS" in args.questionnaire:
        calculate_CTRS_score(save_path)


if __name__ == '__main__':
    main()
