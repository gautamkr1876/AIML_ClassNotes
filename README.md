# AIML Class Notes

A collection of class notes, exercises, and projects for AI/ML learning.

Every module ships with **interview-ready revision guides** at its root and (optionally) per-notebook deep-dives inside each lecture folder. Conventions documented in [`.claude/CLAUDE.md`](./.claude/CLAUDE.md).

---

## Modules

### 📘 Module 1: [Data Foundation](./Data%20Foundation/)

Three companion master guides covering the NumPy → Pandas → applied EDA arc:

- 🔖 **[NumPy Master Revision Guide](./Data%20Foundation/Data_Foundation_Revision_Guide.md)** — NumPy fundamentals, broadcasting, vectorization, drill, sourced bank.
- 🐼 **[Pandas Master Revision Guide](./Data%20Foundation/Pandas_Revision_Guide.md)** — Series, DataFrame, iloc/loc, cleanup, sourced bank.
- 📊 **[Amazon + Sachin EDA Revision Guide](./Data%20Foundation/Amazon_Sachin_EDA_Revision_Guide.md)** — joins, groupby, apply, reshape, datetime, visualization, probability.

#### Notebooks

**NumPy track (Food Delivery EDA):**
- 📂 [Food Delivery EDA 1](./Data%20Foundation/Food%20Delivery%20Data%20Exploration%20and%20analysis%201/) — *NumPy foundation: lists vs arrays, why NumPy is fast, shape/dtype, type coercion, indexing, slicing* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/Data%20Foundation/Food%20Delivery%20Data%20Exploration%20and%20analysis%201/Food_Delivery_Data_Exploration_and_analysis_1.ipynb" target="_blank">▶️ Open in Colab</a>
  - 📄 [NumPy & EDA Interview Prep Guide (deep dive)](./Data%20Foundation/Food%20Delivery%20Data%20Exploration%20and%20analysis%201/NumPy_EDA_Interview_Prep_Guide.md)
- 📂 [Food Delivery EDA 2](./Data%20Foundation/Food%20Delivery%20Data%20Exploration%20and%20analysis%202/) — *NumPy intermediate: boolean masking, reshape, axis/aggregations, np.where, sort, matrix multiplication* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/Data%20Foundation/Food%20Delivery%20Data%20Exploration%20and%20analysis%202/Food_Delivery_Data_Exploration_and_analysis_2.ipynb" target="_blank">▶️ Open in Colab</a>
- 📂 [Food Delivery EDA 3](./Data%20Foundation/Food%20Delivery%20Data%20Exploration%20and%20analysis%203/) — *NumPy advanced: broadcasting (4 rules), vectorization, splitting & stacking* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/Data%20Foundation/Food%20Delivery%20Data%20Exploration%20and%20analysis%203/Food_Delivery_Data_Exploration_and_Analysis_3.ipynb" target="_blank">▶️ Open in Colab</a>
- 📂 [Food Delivery EDA 4](./Data%20Foundation/Food%20Delivery%20Data%20Exploration%20and%20analysis%204/) — *Pandas intro: Series, DataFrame, iloc/loc, info/describe, cleanup, value_counts* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/Data%20Foundation/Food%20Delivery%20Data%20Exploration%20and%20analysis%204/Food_Delivery_Data_Exploration_and_analysis_4.ipynb" target="_blank">▶️ Open in Colab</a>

**Applied pandas + EDA + probability (Amazon + Sachin):**
- 📂 [Amazon Sales Data Analysis 1](./Data%20Foundation/G%20-Amazon%20sales%20data%20analysis%201/) — *Joins (concat vs merge, SQL joins), groupby (split-apply-combine), apply, duplicates, sorting* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/Data%20Foundation/G%20-Amazon%20sales%20data%20analysis%201/Amazon_sales_data_analysis_1.ipynb" target="_blank">▶️ Open in Colab</a>
- 📂 [Amazon Sales Data Analysis 2](./Data%20Foundation/G%20-Amazon%20sales%20data%20analysis%202/) — *Missing data, melt/pivot_table, pd.cut, string ops, datetime, univariate viz* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/Data%20Foundation/G%20-Amazon%20sales%20data%20analysis%202/Copy_of_Amazon_sales_data_analysis_2.ipynb" target="_blank">▶️ Open in Colab</a>
- 📂 [Amazon Sales Data Analysis 3](./Data%20Foundation/G-Amazon%20sales%20data%20analysis%203/) — *Bivariate & multivariate viz (scatter/box/violin/heatmap/pairplot), .corr(), hue/subplots* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/Data%20Foundation/G-Amazon%20sales%20data%20analysis%203/Amazon_sales_data_analysis_3.ipynb" target="_blank">▶️ Open in Colab</a>
- 📂 [Analyzing Sachin Tendulkar's ODI Career](./Data%20Foundation/G-Analyzing%20Sachin%20Tendulkar's%20ODI%20Career/) — *Probability fundamentals, set operations, addition/multiplication/complement, conditional, Bayes' Theorem via pandas filtering* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/Data%20Foundation/G-Analyzing%20Sachin%20Tendulkar's%20ODI%20Career/Analyzing_Sachin_Tendulkar's_ODI_Career.ipynb" target="_blank">▶️ Open in Colab</a>

---

### 🧠 Module 5: [ML Coding — Computer Vision](./5.ML%20Coding%20(CV)/)

**🔖 [CV Master Revision Guide](./5.ML%20Coding%20(CV)/CV_Revision_Guide.md)** — full CNN → CV pipeline with mental models, basic→advanced ladders, sourced interview-question bank, 100-question drill.

#### Notebooks

- 📂 [1. Intro to CV & CNN Fundamentals](./5.ML%20Coding%20(CV)/1.Intro%20to%20CV%20and%20CNN%20Fundamentals/) — *Image as tensor, MLP vs CNN, Conv2D, padding/stride, pooling, output-shape formula* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/5.ML%20Coding%20(CV)/1.Intro%20to%20CV%20and%20CNN%20Fundamentals/Intro_to_CV_and_CNN_Fundamentals%20(1).ipynb" target="_blank">▶️ Open in Colab</a>
  - 📄 [CV1 Deep Dive](./5.ML%20Coding%20(CV)/1.Intro%20to%20CV%20and%20CNN%20Fundamentals/CV1_CNN_Fundamentals_Interview_Prep_Guide.md)
- 📂 [2. Tackling Overfitting in CNN](./5.ML%20Coding%20(CV)/2.Tackling%20Overfitting%20in%20CNN/) — *Dropout, BatchNorm, L2, augmentation, EarlyStopping, GlobalAveragePooling. 51% → 78% test acc* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/5.ML%20Coding%20(CV)/2.Tackling%20Overfitting%20in%20CNN/Tackling_Overfitting_in_CNN.ipynb" target="_blank">▶️ Open in Colab</a>
  - 📄 [CV2 Deep Dive](./5.ML%20Coding%20(CV)/2.Tackling%20Overfitting%20in%20CNN/CV2_Tackling_Overfitting_Interview_Prep_Guide.md)
- 📂 [3. Transfer Learning](./5.ML%20Coding%20(CV)/3.Transfer%20learning%201/) — *Pretrained VGG/ResNet, feature extraction, freezing, fine-tuning. 12% → 79% on 737 images* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/5.ML%20Coding%20(CV)/3.Transfer%20learning%201/Transfer_learning_1.ipynb" target="_blank">▶️ Open in Colab</a>
  - 📄 [CV3 Deep Dive](./5.ML%20Coding%20(CV)/3.Transfer%20learning%201/CV3_Transfer_Learning_Interview_Prep_Guide.md)
- 📂 [4. Image Similarity & Embeddings](./5.ML%20Coding%20(CV)/4.Image%20Similarity%20:%20Understanding%20Embeddings/) — *ResNet-50 embeddings, cosine/L2, PCA, Annoy, t-SNE. 1400× faster than brute-force NN* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/5.ML%20Coding%20(CV)/4.Image%20Similarity%20:%20Understanding%20Embeddings/Image_Similarity_using_CNN.ipynb" target="_blank">▶️ Open in Colab</a>
  - 📄 [CV4 Deep Dive](./5.ML%20Coding%20(CV)/4.Image%20Similarity%20:%20Understanding%20Embeddings/CV4_Image_Similarity_Interview_Prep_Guide.md)
- 📂 [5. Object Detection (Two-Stage)](./5.ML%20Coding%20(CV)/5.Object%20localization%20and%20detection%201/) — *Bbox, IoU, NMS, anchor boxes, R-CNN → Fast R-CNN → Faster R-CNN, RPN, ROI pooling* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/5.ML%20Coding%20(CV)/5.Object%20localization%20and%20detection%201/L7_Object_Detection_with_Two_Stage_Methods.ipynb" target="_blank">▶️ Open in Colab</a>
  - 📄 [CV5 Deep Dive](./5.ML%20Coding%20(CV)/5.Object%20localization%20and%20detection%201/CV5_Object_Detection_TwoStage_Interview_Prep_Guide.md)
- 📂 [6. Object Detection (Single-Stage)](./5.ML%20Coding%20(CV)/6.Object%20localization%20and%20detection%202/) — *YOLO, SSD, RetinaNet, Focal Loss, FPN, OpenCV DNN, video inference* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/5.ML%20Coding%20(CV)/6.Object%20localization%20and%20detection%202/OLD_L8_ObjectDetection_with_Single_Stage_Methods.ipynb" target="_blank">▶️ Open in Colab</a>
  - 📄 [CV6 Deep Dive](./5.ML%20Coding%20(CV)/6.Object%20localization%20and%20detection%202/CV6_Object_Detection_SingleStage_Interview_Prep_Guide.md)
- 📂 [7. Object Segmentation](./5.ML%20Coding%20(CV)/7.Object%20segmentation/) — *Semantic vs instance, FCN, U-Net, transposed conv, Dice loss, Mask R-CNN* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/5.ML%20Coding%20(CV)/7.Object%20segmentation/L9_Object_Segmentation.ipynb" target="_blank">▶️ Open in Colab</a>
  - 📄 [CV7 Deep Dive](./5.ML%20Coding%20(CV)/7.Object%20segmentation/CV7_Object_Segmentation_Interview_Prep_Guide.md)
- 📂 [8. Siamese Network](./5.ML%20Coding%20(CV)/8.Siamese%20network/) — *Twin networks, shared weights, contrastive loss, triplet loss + margin, hard-negative mining* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/5.ML%20Coding%20(CV)/8.Siamese%20network/L10_Updated_Siamese_Network.ipynb" target="_blank">▶️ Open in Colab</a>
  - 📄 [CV8 Deep Dive](./5.ML%20Coding%20(CV)/8.Siamese%20network/CV8_Siamese_Network_Interview_Prep_Guide.md)
- 📂 [9. GANs for Image Generation](./5.ML%20Coding%20(CV)/9.GANs%20for%20Image%20Generation/) — *Generator, Discriminator, DCGAN recipe, minimax loss, mode collapse, FID, WGAN/cGAN* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/5.ML%20Coding%20(CV)/9.GANs%20for%20Image%20Generation/L11_Introductory_Lecture_on_GANs.ipynb" target="_blank">▶️ Open in Colab</a>
  - 📄 [CV9 Deep Dive](./5.ML%20Coding%20(CV)/9.GANs%20for%20Image%20Generation/CV9_GANs_Interview_Prep_Guide.md)

---

### 🌱 Module 6: [Basic of GenAI & AI Agents](./6.%20Basic%20of%20GenAI%20%26%20AI%20Agents/)

Companion reference at the module root:

- 📝 **[Top AI Interview Questions & Answers](./6.%20Basic%20of%20GenAI%20%26%20AI%20Agents/Top%20AI%20Interview%20Questions%20%26%20Answers.md)** — production-flavored GenAI Q&A (RAG debugging, prompt design, evals, agents).

#### Notebooks

- 📂 [1. Introduction to AI Engineering](./6.%20Basic%20of%20GenAI%20%26%20AI%20Agents/Introduction%20to%20AI%20Engineering/) — *intro to AI engineering concepts* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/6.%20Basic%20of%20GenAI%20%26%20AI%20Agents/Introduction%20to%20AI%20Engineering/Introduction%20to%20AI%20Engineering.ipynb" target="_blank">▶️ Open in Colab</a>
- 📂 [2. Refresher: Modern Text & Image AI Model Architectures](./6.%20Basic%20of%20GenAI%20%26%20AI%20Agents/2.%20Refresher%3A%20Modern%20Text%20%26%20Image%20AI%20Model%20Architectures/) — *Decoder-only transformers (GPT → Llama), self-attention, multi-head, causal masking, positional encoding (RoPE), Vision Transformers, PyTorch walkthrough* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/6.%20Basic%20of%20GenAI%20%26%20AI%20Agents/2.%20Refresher%3A%20Modern%20Text%20%26%20Image%20AI%20Model%20Architectures/L2_Refresher_Modern_Text_%26_Image_AI_Model_Architectures.ipynb" target="_blank">▶️ Open in Colab</a>
- 📂 [3. Getting Started with Text Generation LLM APIs (OpenAI API)](./6.%20Basic%20of%20GenAI%20%26%20AI%20Agents/3.%20Getting%20started%20with%20Text%20Generation%20LLM%20APIs%20%28OpenAI%20API%29/) — *Env & API setup (OpenAI / Anthropic / HuggingFace), generation vs reasoning models, base/instruct/chat models, chat templates, decoding & generation strategies* — <a href="https://colab.research.google.com/github/gautamkr1876/AIML_ClassNotes/blob/main/6.%20Basic%20of%20GenAI%20%26%20AI%20Agents/3.%20Getting%20started%20with%20Text%20Generation%20LLM%20APIs%20%28OpenAI%20API%29/Getting_Started_with_Text_Generation_LLM_APIs.ipynb" target="_blank">▶️ Open in Colab</a>

> ⚠️ All Module 6 notebooks are >25 MB (lots of embedded screenshots), so GitHub's preview shows *"this file is too big to display"*. Use the **Open in Colab** link to view and run them — Colab renders the full notebook with all images.

---

### 🤖 Module 2: [ML Coding — Supervised Learning](./2.ML%20Coding%20(Supervised%20Learning)/)

*(Notebooks to be added.)*

---

## Working conventions

All revision guides follow the pattern documented in [`.claude/CLAUDE.md`](./.claude/CLAUDE.md):

- 🪜 Mental models per concept
- 🪞 Basic → Intermediate → Advanced ladders
- 🎯 Q&A with inline citations to source repos (`alexeygrigorev/data-science-interviews`, `kojino/120-Data-Science-Interview-Questions`, `rougier/numpy-100`, `ajcr/100-pandas-puzzles`, `chiphuyen/ml-interviews-book`, etc.)
- 🌐 Dedicated **Sourced interview questions** sections so practice never requires leaving the notes
- 🔁 50–100 question revision drills

> **Tone:** terse, scannable, anchored in real interview pressure. Not tutorial prose.
