import torch
import json
import os
import argparse

from random import sample
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments

from utils.data_collator_for_SLiC import DataCollatorForSLiC
from utils.distractor_compare_dataset import DistractorComparelDataset
from utils.SLiC_trainer import SLiCTrainer

# 記得訓練要修改 Project Name，會顯示在 wandb 上 Project 的名字
project_name = " Your Project Name "
os.environ["WANDB_PROJECT"] = project_name

def main(args):
    # load data and tokenize
    with open(args.dataset_path, "r") as f: 
        data = json.load(f)
    train_data = sample(data['train'],len(data['train']))
    valid_data = sample(data['valid'],2000)
    eval_data = sample(data['eval'],2000)

    tokenizer = T5Tokenizer.from_pretrained(args.model_path)

    train_dataset = DistractorComparelDataset(train_data, tokenizer, sample_size = args.options_size)
    valid_dataset = DistractorComparelDataset(valid_data, tokenizer, sample_size = args.options_size)
    eval_dataset = DistractorComparelDataset(eval_data, tokenizer, sample_size = args.options_size)

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)

    # set hyper parameter
    batch_size = 3
    data_collator = DataCollatorForSLiC(tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        save_strategy = "epoch",
        evaluation_strategy = "epoch",
        learning_rate=args.learning_rate,
        logging_steps=500,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        load_best_model_at_end=True,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        eval_accumulation_steps = 1,
        remove_unused_columns = False,
        report_to="wandb" if os.getenv("WANDB_PROJECT") else "none"
    )
    # trainer sample 
    trainer = SLiCTrainer(
        model = model,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        sample_size = args.options_size,
        length_penalty = 1,
        gamma = args.gamma,
        margin = args.margin,
        compare_size = 3,
    )
    print(args.gamma)
    print(args.margin)
    print(args.output_dir)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="../model/t5_vanilla-DG/t5-base-clean-sent-ans-tripleD-,split")
    parser.add_argument('--dataset_path', type=str, default='../data/CLOTH-F/cloth-f-fit-options/cloth-f-fit-answer-no-ans_3.json')
    parser.add_argument('--output_dir', type=str, default="results")
    parser.add_argument('--options_size', type=int, default=6)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--margin', type=float, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    args = parser.parse_args()
    main(args)