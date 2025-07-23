import torch
import numpy as np
import faiss
import pickle
import json
import os
from PIL import Image
from tqdm import tqdm

class ProductionSimilarityEngine:
    def __init__(self, model, processor):  # ← Fixed: Removed save_dir parameter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = processor
        self.model = model.to(self.device)
        self.model.eval()
        self.save_dir = "faiss_index"
        
        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

    def extract_img_feat(self, path):
        """Extract features from image"""
        img = Image.open(path).convert('RGB')
        inp = self.processor(images=[img], return_tensors="pt")
        
        with torch.no_grad():
            feat = self.model.get_image_features(
                pixel_values=inp['pixel_values'].to(self.device)
            )
            return (feat/feat.norm(dim=-1,keepdim=True)).cpu().numpy()

    def build_index(self, data):
        """Build FAISS index for similarity search"""
        print("🏗️ Extracting features and building FAISS index...")
        
        feats, meta = [], []
        for item in tqdm(data, desc="Processing items"):
            f = self.extract_img_feat(item['image_path']).flatten()
            feats.append(f)
            meta.append(item)
        
        # Create FAISS index
        features_array = np.vstack(feats)
        dimension = features_array.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(features_array.astype('float32'))
        self.metadata = meta
        
        # Save the index and metadata
        self.save_index(features_array)
        
        print(f"FAISS index built with {len(meta)} items")

    def save_index(self, features_array):
        """Save FAISS index and metadata to disk"""
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.save_dir, "index.faiss"))
        
        # Save metadata
        with open(os.path.join(self.save_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save raw features
        np.save(os.path.join(self.save_dir, "features.npy"), features_array)
        
        # Save item mapping
        item_mapping = {item['id']: idx for idx, item in enumerate(self.metadata)}
        with open(os.path.join(self.save_dir, "item_mapping.json"), 'w') as f:
            json.dump(item_mapping, f, indent=2)
        
        print(f" FAISS files saved to {self.save_dir}/")

    def load_index(self):
        """Load existing FAISS index from disk"""
        index_path = os.path.join(self.save_dir, "index.faiss")
        metadata_path = os.path.join(self.save_dir, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f" Loaded existing index with {len(self.metadata)} items")
            return True
        else:
            print(" No existing index found")
            return False

    def find_similar_images(self, query_path, k=10):
        """Find similar images using FAISS index"""
        q = self.extract_img_feat(query_path).astype('float32')
        sims, idxs = self.index.search(q, k)
        
        return [{'id':self.metadata[i]['id'],'score':float(s),'path':self.metadata[i]['image_path']}
                for s,i in zip(sims[0],idxs[0])]
