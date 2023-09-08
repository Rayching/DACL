DACL(Degenerating Ambiguous-options via Contrastive Learning)

干擾選項生成 (Distractor Generation, DG) 為選擇題自動生成研究中重要的一
環。文獻中也有許多干擾選項生成之研究被提出。然而，我們發現現有干擾選項
生成模型，面臨著產生模擬兩可答案之顧慮 (例如產生與答案選項同義字為干擾
選項)，造成學生答題上之爭議。有鑑於此，在本篇研究中，我們提出 DACL 框
架 (Degenerating Ambiguous-options via Contrastive Learning)，主要使用對比式學習
(Contrastive Learning) 的方式，於訓練 DG 生成模型時，除了原先之生成目標外，
也規範 (Regulate) 模型產生可以當作答案的選項的生成機率。

## 資料夾介紹

```
.
├── data // 存放 cloth data 以及使用 ChatGPT、Deberta 製作的 DACL training data
├── evaluate // 計算 DACL-DG 的 F1, Alikescore, Answerscore
├── make_data // 存放產生 Ambiguous Options 的程式碼，包括 ChatGPT 以及 Deberta
├── model // 存放所有 model，包含 Vanilla-DG 以及 DACL-DG，Vanilla-DG 為只有經過 DG 訓練的模型，DACL-DG 為有使用 DACL learning 後的模型
├── t5_DACL_training // 存放 DACL learning 的程式碼，用來做不產生 Ambiguous Options 的訓練
└── t5_vanilla-DG_training // 存放 Distractor Generation 的程式碼，訓練模型的 DG 能力

```
# 使用說明

1. 準備資料
首先可以進入 make_data 資料夾，使用 ChatGPT 或是 Deberta 產生想要的 Ambiguous Options 資料，也可以使用現存的資料
ChatGPT 產生的資料在 data/chatGPT_answer 資料夾中
Deberta 產生的資料在 data/deberta_negative 資料夾中

2. 開始訓練
+ 接著我們要準備 Vanilla-DG model，進入 t5_vanilla-DG_training 資料夾中，使用 train_t5_text2text_sentence_len3_triple_DG_,split.ipynb，產生出經過 DG 訓練的 model，epochs 會存在此目錄的 results 資料夾，最好的 model 會存放在 model/t5_vanilla-DG

t5_vanilla-DG_training
```
.
└── train_t5_text2text_sentence_len3_triple_DG_,split.ipynb // 訓練有 DG 能力的模型，使用 t5-base 做訓練

```
+ 接著需要做 DACL training，進入 t5_DACL_training 資料夾中，修改 train.sh 檔案中的參數，接著就可以執行 train.sh 開始訓練

t5_DACL_training
```
.
├── train.py // 主要執行的程式，會調用 utils 內自己定義的 Trainer
├── train.sh // 主要調整的地方，需要調整的參數大多都寫在上面
└── utils // 自行定義的 data_collator、Dataset function、Trainer，非必要不做更動
    ├── data_collator_for_SLiC.py 
    ├── distractor_compare_dataset.py
    └── SLiC_trainer.py // 自行定義的 Trainer，主要計算 loss 的地方位於 compute_loss funtion 內
```

3. 做評估
+ 使用 evaluate 資料夾中的檔案做評估，計算出模型的分數，主要會看的地方為 
F1 scroe -> 代表模型生成與題目相同 distractor 的能力
Alike score -> 代表模型生成出 Ambiguous Options 的機率
Answer score -> 代表模型生成出與題目相同答案的機率

evaluate
```
.
├── cal_best_model_copy.ipynb //  將 evaluate 之後的分數按照 F1 排序
├── evaluate_answer_prob_model_juan_predict_cloth-f.ipynb // 計算學姊 model 在 cloth-f 的表現，一次生成三個 distractor，所以在計算分數的地方有做處理
├── evaluate_answer_prob_model_slic_,split_multi_chatgpt.ipynb // 計算 DACL-DG 在生成 Ambiguous Options 的模型表現，Ambiguous Options 使用 ChatGPT 生成，一次生出一個 distractor
├── evaluate_answer_prob_model_slic_,split_multi_clothf_for_clean.ipynb // 計算模型在 cloth-f 的表現，一次生出一個 distractor
├── evaluate_answer_prob_model_slic_,split_multi_LLM_negative.ipynb // 計算 DACL-DG 在生成 Ambiguous Options 的模型表現，Ambiguous Options 使用 Deberta 生成，一次生出一個 distractor
└── evaluate_answer_prob_model_slic_,split.ipynb // 計算模型生成 Ambiguous Options 的模型表現，Ambiguous Options 使用 ChatGPT 生成，一次生出三個 distractor

```