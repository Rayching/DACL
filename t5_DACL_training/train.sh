# gamma 為修改 entropy loss 的比例，也就是原本的 DG 任務要在這次訓練中占多少比例
# margin 為修改每個 options 之間的距離，也就是 Distractor 與 Ambiguous Options 要離多遠
# options_size options 有幾個，通常為六個，前面三個為題目的 distractor，後面三個為 Ambiguous Options
# output_dir 為訓練出來的模型要存放的資料夾
# dataset_path 為要使用的資料集，必須使用具有 Ambiguous Options 的資料集
# gradient_accumulation_steps 要相加幾次 gradient 之後再回傳
python train.py --gamma 1.0 \
                --margin 1.0 \
                --options_size 6 \
                --model_path your_vanillaDG_model_path \
                --output_dir ../model/t5_DACL-DG/your_DACLDG_model_name \
                --dataset_path ../data/CLOTH-F/cloth-f-fit-answer-no-ans_3.json \
                --gradient_accumulation_steps 4