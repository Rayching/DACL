import json
import os
import argparse 
from random import sample, shuffle
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from time import sleep
import openai


def open_jsonl_fix_data(name):
    with open('../../data/cloth-f-fit-options/cloth-f-fit-options_{}_fit-answer.jsonl'.format(name)) as f:
        data = [json.loads(line) for line in f]
    return data

def open_cloth_dataset(name):
    with open('../../data/CLOTH-F/clean_cloth-f_{}.json'.format(name)) as f:
        data = json.load(f)
    return data

# 將 option list 做成 str
def make_choice_sign(option_list):
    shuffle(option_list)
    options_str = str()
    for num,opt in enumerate(option_list):
        options_str += " {} {}".format(chr(65+num), opt)
    return options_str, option_list

# 做成給 chatgpt 的 prompt + Question
def chatgpt_prompt_fit_blank_format(one_data):
    chatgpt_prompt = " Questions: " + one_data["sentence"] 
    return chatgpt_prompt
    
# 送給 chatgpt request，將輸入想要 chatgpt 做事情的 system_prompt 與丟入題目的 prompt 分開
def get_chatgpt_respond(data, max_test_time = 3):
    prompt = chatgpt_prompt_fit_blank_format(data)
    system_prompt = "I will give you a sentence with a missing word and some options. Please help me select the options that have suitable meanings to be considered as answers."

    for _ in range(max_test_time):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "system", 
                     "content": system_prompt},
                    {"role": "user", 
                     "content": prompt}
                ],
                max_tokens=500,
                temperature=0,
            )
            response_text = response["choices"][0]['message']['content']
            break
        except Exception as e:
            print(f"data : {data['index']} error")
            print(e)
            response_text = "ERROR"
    data['prompt'] = prompt
    data['chatgpt_response'] = response_text
    return data

def main(args):
    openai.api_key = "your_key"
    data_name = ["train","valid","test"]
    for dn in data_name:
        # data = data_name[dn]
        data = open_cloth_dataset(dn)
        filepath = f"../../data/cloth-f-fit-options/cloth-f-fit-options_{dn}_fit-answer.jsonl"
        f = open(filepath, "a", encoding="utf-8")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            processing_data = sample(data, args.test_size) if args.test_size > 0 else data
            for r in tqdm(executor.map(get_chatgpt_respond, processing_data), total=len(processing_data), desc = f'{dn}_chatgpt_respond'):
                json_data = json.dumps(r, ensure_ascii=False) + "\n"
                f.write(json_data)
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--test_size', type=int, default=0)
    parser.add_argument('--max_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)
    