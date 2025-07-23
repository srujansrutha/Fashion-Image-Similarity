from data_processor import KhaadiFashionPreprocessor
from dataset import KhaadiFashionDataset
from trainer import CLIPFashionTrainer
from similarity_engine import ProductionSimilarityEngine
from transformers import CLIPProcessor, CLIPModel
import os
import torch

def main():
    CSV="E:/TrendGully/AI/data/raw/khaadi_data.csv"
    IMGDIR="E:/TrendGully/AI/data/raw/images"
    MODEL_PATH = "E:/TrendGully/AI/src/models/clip_fashion_final.pt"

    print(" Starting Khaadi Fashion Pipeline (Using Existing Model)")
    
    # Step 1: Data preprocessing
    pre = KhaadiFashionPreprocessor(CSV, IMGDIR)
    df = pre.load_and_enhance_data()
    data = pre.validate_and_select_images(df)
    print(f" Processed {len(data)} valid fashion items")

    # Step 2: Load existing trained model (SKIP TRAINING)
    print(" Loading existing fine-tuned model...")
    
    # Load CLIP processor
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load the base CLIP model architecture
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Load your fine-tuned weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(" Fine-tuned model loaded successfully")

    # Step 3: Build similarity engine (FIX CONSTRUCTOR)
    print("🔍 Building similarity search engine...")
    
    # Create the similarity engine without save_dir parameter
    sim = ProductionSimilarityEngine(model, proc)  # ← Fixed: Removed save_dir
    
    # Build FAISS index
    sim.build_index(data)

    # Step 4: Test similarity search
    print("🧪 Testing similarity search...")
    q = data[0]['image_path']
    print(f"Query: {q}")
    
    results = sim.find_similar_images(q, k=5)
    print("Top 5 similar items:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['id']} - Score: {r['score']:.3f}")

if __name__=="__main__":
    main()
