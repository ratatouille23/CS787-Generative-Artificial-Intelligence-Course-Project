# CS787-Generative-Artificial-Intelligence-Course-Project
Improving Hybrid Attention Network (HAN) for Stock Movement Prediction using FinBERT-enhanced Embeddings
========================================================================================================

**Authors:** Anurag Patel (230173) | Shlok Misar (230653)

**Date:** \\today (or November 14, 2025, if you prefer a fixed date)

Abstract
--------

Financial news plays a crucial role in shaping short-term stock market movements, yet extracting meaningful signals from textual data remains a challenging task. This project investigates the effectiveness of a **Hybrid Attention Network (HAN)** for stock movement prediction and introduces an improved variant that incorporates **FinBERT**, a domain-specific language model pretrained on large-scale financial corpora. The baseline HAN uses Word2Vec embeddings and a hybrid attention structure that combines content-level (news) and temporal attention to aggregate information from multiple days of news. However, static embeddings and simple averaging can miss important contextual nuances.

To address this limitation, we propose **FinBERT-HAN**, a modified architecture that replaces word-level static embeddings with contextual FinBERT \[CLS\] representations and introduces a news-level attention mechanism before temporal modeling. Both models are trained and evaluated on a curated dataset aligned with stock price movements. Experimental results indicate that FinBERT-HAN consistently improves predictive performance on key metrics (**accuracy, F1-score, MCC**), demonstrating the benefit of domain-specific contextual embeddings when combined with hybrid temporal attention mechanisms. We also discuss computational trade-offs, ablation ideas, and practical considerations for deployment.

1\. Introduction
----------------

### 1.1 Motivation

Using financial news to predict short-term stock movements remains challenging due to the nuanced nature of financial language. Domain-specific language models like **FinBERT** offer significant advantages over generic word embeddings for this task.

### 1.2 Problem Statement

Binary classification of stock movement (up/down) using news data over previous $D$ days.

### 1.3 Contributions

1.  Implementation and evaluation of the project's original **Hybrid Attention Network (HAN)** architecture on the CMIN-US dataset.
    
2.  A modified HAN (**FinBERT-HAN**) that replaces word-level embeddings with FinBERT \[CLS\] embeddings.
    
3.  Empirical comparison and analysis of results, computational trade-offs, and suggestions for production deployment.
    

2\. Dataset
-----------

*   **Source:** CMIN-US price files and corresponding news dumps
    
*   **Time Range:** 01-01-2018 to 31-12-2021
    
*   **Splits:**
    
    *   **Train:** 01-01-2018 $\\rightarrow$ 30-04-2021
        
    *   **Dev:** 01-05-2021 $\\rightarrow$ 31-08-2021
        
    *   **Test:** 01-09-2021 $\\rightarrow$ 31-12-2021
        
*   **Stocks Selected:** Apple (AAPL), Amazon (AMZN), Google (GOOG), Microsoft (MSFT), Morgan Stanley (MS), Nvidia (NVDA), Netflix (NFLX), Tesla (TSLA), J. P. Morgan Chase (JPM), RTX (RTX)
    
*   **Samples:** Train: \[_**\] | Dev: \[**_\] | Test: \[\_\_\_\]
    
*   **Labeling:** Price change $> 0.55\\% \\rightarrow$ Up (**1**); Price change $< -0.5\\% \\rightarrow$ Down (**0**)
    

3\. Preprocessing
-----------------

### 3.1 Original HAN (Baseline)

*   Tokenization using a custom tokenizer (sentence / word-level splitting as needed).
    
*   Word2Vec embedding pretraining using gensim on the corpus of news text.
    
*   Sequence shape used by the model: $(\\text{batch}, \\text{days}, \\text{news\\\_items}, \\text{words})$.
    
*   For each news item we build a fixed-length word-index vector (padding/truncation).
    

### 3.2 FinBERT-HAN

*   Tokenization via AutoTokenizer.from\_pretrained() (FinBERT WordPiece tokenizer).
    
*   Input shape per sample: $(\\text{batch}, \\text{days}, \\text{max\\\_tweets}, 3, \\text{max\\\_tokens})$ (channels: input\_ids, token\_type\_ids, attention\_mask).
    
*   For each news item we precompute / dynamically compute the FinBERT output and take the \[CLS\] embedding (768-d).
    
*   Filtering: samples with $>1$ missing news day removed.
    

4\. Model Architectures
-----------------------

### 4.1 Original HAN (Baseline) -- Hybrid Attention Network

**Overview:** The project's original model is a **Hybrid Attention Network** that focuses on two complementary attention mechanisms:

1.  **Content-level (news) attention:** assigns importance weights to different news items within the same day.
    
2.  **Temporal attention:** models sequential importance across multiple days.
    

**High-level pipeline:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Input (batch, days, news_items, words)      └─> Word Embedding Layer (Word2Vec)          └─> Bi-GRU (news encoder) -> per-news vector              └─> News-level Attention  --> Day Vector      └─> Temporal Bi-GRU (over days) -> Temporal Attention --> Final Doc Vector      └─> Dense Layers -> Softmax (2 classes)   `

**Notes:**

*   Unlike a fully hierarchical HAN, the hybrid variant used here treats each news item as a single unit produced by a per-news encoder (word-level Bi-GRU) and emphasizes news aggregation plus temporal modeling.
    
*   The hybrid design is better suited when documents are short (news articles or tweets) and when temporal patterns across days are crucial.
    

### 4.2 FinBERT-HAN (Modified)

**Overview:** Replace the word-level static embeddings with contextual FinBERT embeddings for each news item and apply the same hybrid attention/temporal modeling pipeline on top of these richer vectors.

**High-level pipeline:**

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   Input (batch, days, max_tweets, token_ids/token_type/att_mask)    └─> FinBERT (AutoModel) per news -> [CLS] embedding (768-d)        └─> News Attention -> Day Vector            └─> Bi-GRU over days -> Temporal Attention -> Final Vector                └─> Dense -> Dropout -> Dense -> Output logits   `

**Notes:**

*   FinBERT outputs contextual embeddings. We use the \[CLS\] vector by default.
    
*   FinBERT may be _frozen_ (to reduce memory) or _fine-tuned_ (for potentially better performance).
    

5\. Training Setup and Hyperparameters
--------------------------------------

**ParameterOriginal HANFinBERT-HANLoss Function**CrossEntropyCrossEntropy**Optimizer**AdamWAdamW**Learning Rate**1e-4 (or 1e-3 warmup)1e-5 (if fine-tuning FinBERT)**Batch Size**168--16 (smaller when fine-tuning)**Epochs**\[insert\]\[insert\]**Dropout**0.30.3**Hidden Size**\[insert\]768 (or 256 lightweight)

**Table:** Training hyperparameters (example). Adjust according to experiments.

**Training tips:**

*   Use **class-weighted CrossEntropy** due to label imbalance.
    
*   For FinBERT fine-tuning: use smaller LR (e.g., 1e-5), gradient accumulation, mixed precision (AMP).
    
*   Clip gradients to prevent exploding gradients in GRU layers.
    

6\. Evaluation Metrics
----------------------

*   Accuracy, Precision, Recall, F1-score
    
*   **Matthews Correlation Coefficient (MCC)**
    
*   Confusion matrix (TP, FN, FP, TN)
    
*   Training and inference time (per epoch / per sample)
    

7\. Results
-----------

### 7.1 Training Curves

### 7.2 Confusion Matrices

**Original Hybrid HANFinBERT-HAN**

### 7.3 Quantitative Comparison

**MetricOriginal HANFinBERT-HANDifferenceAccuracy**\[\*\*\*\]\[\*\*\*\]\[\*\*\*\]Precision\[\*\*\*\]\[\*\*\*\]\[\*\*\*\]**Recall**\[\*\*\*\]\[\*\*\*\]\[\*\*\*\]F1-score\[\*\*\*\]\[\*\*\*\]\[\*\*\*\]**MCC**\[\*\*\*\]\[\*\*\*\]\[\*\*\*\]Dev Loss\[\*\*\*\]\[\*\*\*\]\[\*\*\*\]**Test Loss**\[\*\*\*\]\[\*\*\*\]\[\*\*\*\]Training Time/Epoch\[\*\*\*\]\[\*\*\*\]\[\*\*\*\]**Trainable Parameters**\[\*\*\*\]\[\*\*\*\]\[\_\_\_\]

**Table:** Performance comparison between models

8\. Analysis and Discussion
---------------------------

\[Summary of which model performed better and on what metrics. Discussion of statistical significance, error analysis, and ablation studies.\]

9\. Limitations
---------------

*   Dataset selection and preprocessing biases (sample filtering, thresholding).
    
*   Limited number of stocks and possible sampling bias.
    
*   Using the \[CLS\] token may discard fine-grained token-level information in some cases.
    
*   Computational constraints may limit full end-to-end FinBERT fine-tuning experiments.
    

10\. Future Work
----------------

*   End-to-end FinBERT fine-tuning with gradient accumulation and mixed precision.
    
*   Replace GRU with **transformer encoders** for temporal modeling.
    
*   Multi-task learning (predict direction + magnitude or volatility).
    
*   Integration of alternative data sources (social media, fundamentals) and late-fusion strategies.
    

11\. Conclusion
---------------

\[Short recap of problem, approach and key findings with final numeric results.\]

12\. Reproducibility & Artifacts
--------------------------------

### 12.1 Environment

*   python == 3.12
    
*   torch == 2.6.0+cu124
    
*   torcheval == 0.0.7
    
*   transformers == 4.57.1
    
*   seaborn == 0.13.2
    
*   numpy == 2.3.4
    
*   gensim == 4.4.0
    
*   matplotlib == 3.10.7
    
*   pandas == 2.3.3
    
*   scikit-learn == 1.7.2
    
*   cuda == 12.4
    

### 12.2 Commands

Bash

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   # Preprocess dataset  python preprocess_finbert.py --config config.py  # Train  python simple_train.py --device cuda --train_epochs 10 --batch_size 8 --freeze True  # Evaluate  python simple_train.py --mode test --checkpoint checkpoints/best_model.pt   `

13\. References
---------------

1.  Zhang, X., & Wang, W. (2018). _Listening to Chaos: A Hierarchical Attention Network for Stock Movement Prediction_. Proceedings of EMNLP.
    
2.  FinBERT pretraining resource: [https://huggingface.co/yiyanghkust/finbert-pretrain](https://huggingface.co/yiyanghkust/finbert-pretrain) (or whichever model you used).
    
3.  Transformers library: Hugging Face. [https://huggingface.co](https://huggingface.co/)
    

Acknowledgements
----------------

\[Professor / lab / dataset providers\]
