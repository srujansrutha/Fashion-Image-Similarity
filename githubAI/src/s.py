import streamlit as st
import numpy as np
from PIL import Image
import os
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle

# Page configuration
st.set_page_config(
    page_title="Fashion Image Similarity Search",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_similarity_engine():
    """Load the trained model and FAISS index"""
    try:
        # Load CLIP processor
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load base CLIP model
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load your fine-tuned weights
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = "clip_fashion_final.pt"
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            st.success("Fine-tuned model loaded successfully")
        else:
            st.error(f"Model file not found: {model_path}")
            return None, None, None, None, False
        
        # Load FAISS index
        faiss_dir = "faiss_index"
        index_path = os.path.join(faiss_dir, "index.faiss")
        metadata_path = os.path.join(faiss_dir, "metadata.pkl")
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            index = faiss.read_index(index_path)
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            st.success(f"FAISS index loaded with {len(metadata)} items")
            return model, processor, index, metadata, True
        else:
            st.error("FAISS index files not found")
            return None, None, None, None, False
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, False

def extract_image_features(model, processor, image, device):
    """Extract features from PIL Image"""
    inputs = processor(images=[image], return_tensors="pt")
    
    with torch.no_grad():
        features = model.get_image_features(
            pixel_values=inputs['pixel_values'].to(device)
        )
        # Normalize for cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy()

def extract_text_features(model, processor, text, device):
    """Extract features from text description"""
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    
    with torch.no_grad():
        features = model.get_text_features(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device)
        )
        features = features / features.norm(dim=-1, keepdim=True)
    
    return features.cpu().numpy()

def search_similar_items(index, metadata, query_features, top_k=10):
    """Search for similar items using FAISS"""
    similarities, indices = index.search(
        query_features.astype('float32'), top_k
    )
    
    results = []
    for sim, idx in zip(similarities[0], indices[0]):
        if idx != -1:  # Valid result
            item = metadata[idx].copy()
            item['similarity_score'] = float(sim)
            item['similarity_percentage'] = f"{sim * 100:.1f}%"
            results.append(item)
    
    return results

def display_results(results):
    """Display similarity search results with images"""
    if not results:
        st.warning("No similar items found. Try different input.")
        return
    
    st.success(f"Found {len(results)} similar items!")
    
    # Create columns for grid layout
    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(results):
                result = results[i + j]
                
                with col:
                    # Display product image
                    if 'image_path' in result and os.path.exists(result['image_path']):
                        try:
                            img = Image.open(result['image_path'])
                            st.image(img, use_column_width=True)
                        except:
                            st.error("Image not available")
                    else:
                        st.error("Image not found")
                    
                    # Product information
                    st.write(f"**Product ID:** {result.get('id', 'Unknown')}")
                    
                    # Similarity score with color coding
                    similarity = result['similarity_score']
                    percentage = result['similarity_percentage']
                    
                    if similarity > 0.8:
                        st.success(f"Match: {percentage} - Excellent")
                    elif similarity > 0.6:
                        st.info(f"Match: {percentage} - Good")
                    else:
                        st.warning(f"Match: {percentage} - Fair")
                    
                    # Additional details in expandable section
                    with st.expander("Product Details"):
                        if 'text_description' in result:
                            st.write(f"**Description:** {result['text_description']}")
                        
                        st.write(f"**Similarity Score:** {similarity:.4f}")
                        st.write(f"**File Path:** {result.get('image_path', 'N/A')}")

def main():
    # Header
    st.title("Fashion Image Similarity Search")
    
    # Load the similarity engine
    model, processor, index, metadata, loaded = load_similarity_engine()
    
    if not loaded:
        st.error("Failed to load model and FAISS index. Please ensure:")
        st.markdown("""
        - clip_fashion_final.pt exists in the current directory
        - faiss_index/ folder with index files exists
        - All dependencies are installed
        """)
        st.stop()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sidebar configuration
    with st.sidebar:
        st.title("Search Configuration")
        
        search_mode = st.selectbox(
            "Choose search method:",
            ["Upload Image", "Text Description", "Random Browse"]
        )
        
        num_results = st.slider("Number of results:", 1, 20, 10)
        
        st.divider()
        
        # Model information
        st.subheader("Model Information")
        st.info(f"Fine-tuned CLIP model (id = openai/clip-vit-base-patch32)")

    # Main content area
    if search_mode == "Upload Image":
        image_search_interface(model, processor, index, metadata, device, num_results)
    elif search_mode == "Text Description":
        text_search_interface(model, processor, index, metadata, device, num_results)
    else:
        random_browse_interface(metadata, num_results)

def image_search_interface(model, processor, index, metadata, device, num_results):
    """Interface for image-based similarity search"""
    st.subheader("Image Similarity Search")
    st.write("Upload a fashion image to find similar items in our database.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a fashion image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload clear images of clothing items for best results"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
    
    with col2:
        if uploaded_file is not None:
            if st.button("Find Similar Items", type="primary", use_container_width=True):
                with st.spinner("Analyzing image and searching for similar items..."):
                    try:
                        # Find similar items
                        query_features = extract_image_features(model, processor, image, device)
                        results = search_similar_items(index, metadata, query_features, num_results)
                        
                        # Display results
                        display_results(results)
                        
                    except Exception as e:
                        st.error(f"Error during search: {e}")

def text_search_interface(model, processor, index, metadata, device, num_results):
    """Interface for text-based similarity search"""
    st.subheader("Text-to-Image Search")
    st.write("Describe the fashion item you're looking for using natural language.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_query = st.text_area(
            "Enter your fashion description:",
            placeholder="e.g., 'blue embroidered traditional kurta', 'red floral summer dress', 'black leather jacket'",
            height=120,
            help="Be specific about colors, patterns, styles, and garment types"
        )
    
    with col2:
        st.markdown("**Search Tips:**")
        st.markdown("""
        - Include **colors** (blue, red, black)
        - Mention **patterns** (floral, embroidered, printed)  
        - Specify **styles** (traditional, casual, formal)
        - Add **garment types** (kurta, dress, suit)
        """)
    
    if text_query.strip():
        if st.button("Search Fashion Items", type="primary", use_container_width=True):
            with st.spinner("Processing text query and searching database..."):
                try:
                    # Find similar items
                    query_features = extract_text_features(model, processor, text_query, device)
                    results = search_similar_items(index, metadata, query_features, num_results)
                    
                    # Display query info
                    st.info(f"**Query:** '{text_query}'")
                    
                    # Display results
                    display_results(results)
                    
                except Exception as e:
                    st.error(f"Error during search: {e}")

def random_browse_interface(metadata, num_results):
    """Interface for browsing random items"""
    st.subheader("Browse Fashion Collection")
    st.write("Discover random fashion items from our database.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Show Random Items", use_container_width=True):
            try:
                import random
                
                # Select random items
                random_indices = random.sample(
                    range(len(metadata)), 
                    min(num_results, len(metadata))
                )
                
                # Create mock results format
                results = []
                for idx in random_indices:
                    item = metadata[idx].copy()
                    item['similarity_score'] = 1.0  # Perfect match for display
                    item['similarity_percentage'] = "100.0%"
                    results.append(item)
                
                display_results(results)
                
            except Exception as e:
                st.error(f"Error browsing items: {e}")
    
    with col2:
        if st.button("Database Statistics", use_container_width=True):
            st.subheader("Collection Statistics")
            
            total_items = len(metadata)
            st.metric("Total Items", total_items)
            
            st.success(f"Fashion database ready with {total_items} indexed items!")
    
    with col3:
        if st.button("Refresh Page", use_container_width=True):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
