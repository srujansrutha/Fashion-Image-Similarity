# Fashion-CLIP Similarity System – README

## 1. Overview  
This project fine-tunes OpenAI’s CLIP (“Contrastive Language–Image Pre-training”) model on a private Khaadi-brand catalogue to power image-and-text similarity search.

### What CLIP Does  
CLIP is trained on 400 million public image–caption pairs to align visual and textual embeddings in a single 512-dimension space. After fine-tuning on your labelled Khaadi pairs, the model can:

* turn any product photo into an **embedding**  
* turn any product description into an **embedding**  
* measure cosine similarity between embeddings → “find items that look/mean the same”

### What This Repo Adds  
* **Domain fine-tune** of CLIP on 400 Khaadi items (image + generated description).  
* **FAISS index** for millisecond similarity search.  
* **Training / inference code** split into clean modules.  
* **Example pipeline** that runs end-to-end and prints top-K similar results.

## 2. Dataset Used  

| Source                         | Size | Format |
|--------------------------------|------|--------|
| `khaadi_data.csv`              | 400 rows | ID, Product Name, Product Description, Color, Price, Availability |
| Image folders `images//`   | 1–2 images each | JPEG / PNG |

During preprocessing each row becomes:  

```
text_description = "{Product Name} {Product Description} in {Color} color"
```

Example:  
*“Fabrics 3 Piece Suit Printed Cambric Top Bottoms Dupatta in BLUE color”*

## 3. Project Structure  

```
E:/TrendGully/AI/
├── data/
│   └── raw/
│       ├── khaadi_data.csv
│       └── images/
│           ├── ACA231001/
│           ├── ACA231002/
│           └── …
├── src/models/
│   ├── data_processor.py       # CSV + image validation & text synthesis
│   ├── dataset.py              # PyTorch Dataset → CLIP processor
│   ├── trainer.py              # CLIP fine-tuning loop
│   ├── similarity_engine.py    # Feature extraction + FAISS index
│   └── main.py                 # Orchestrates full pipeline
└── requirements.txt
```

## 4. Key Files Explained  

| File | Purpose |
|------|---------|
| **data_processor.py** | -  Reads CSV-  Builds `text_description`-  Picks best image in each folder |
| **dataset.py** | Converts every (image, text) pair into tensors with padding/truncation so batches stack without error. |
| **trainer.py** | Loads pre-trained `openai/clip-vit-base-patch32`, freezes nothing, trains 5 epochs with CLIP contrastive loss and AdamW 1e-5. |
| **similarity_engine.py** | Extracts NORMALISED embeddings and builds a `faiss.IndexFlatIP` for cosine similarity. |
| **main.py** | Step-by-step:1 Preprocess → 2 Dataset → 3 Train → 4 Build index → 5 Quick test → 6 Save `khaadi_fashion_clip_final.pt`. |

## 5. How to Run  

```bash
# 1. install
pip install -r requirements.txt     # torch, torchvision, transformers, faiss-cpu …

# 2. move to code folder
cd E:/TrendGully/AI/src/models

# 3. launch
python main.py
```

Outputs:  
* `khaadi_clip_epoch_*.pt` – checkpoints per epoch  
* `khaadi_fashion_clip_final.pt` – final weights  
* Printed Top-5 similar items for an image and a text query  
* FAISS index in RAM for experimentation (persist with `faiss.write_index` if desired)

## 6. Important Notes  

* **Batch size vs RAM** – start at 8 on CPU; lower if OOM.  
* **Images missing or corrupted** – `data_processor.py` skips them.  
* **Tokenizer padding bug** – dataset pads/truncates to 77 tokens so DataLoader can stack tensors evenly.  
* **Switch to GPU** – install CUDA build of PyTorch and the script will auto-select `cuda`.

## 7. Extending  

1. **Add more brands** – drop new rows in CSV + image folders; rerun `main.py`.  
2. **Persist vector DB** – replace FAISS with Pinecone or Milvus via `similarity_engine.py`.  
3. **Web demo** – wrap `similarity_engine` in Streamlit for drag-and-drop search.
