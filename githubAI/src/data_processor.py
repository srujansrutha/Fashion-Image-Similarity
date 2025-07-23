import pandas as pd
from pathlib import Path

class KhaadiFashionPreprocessor:
    def __init__(self, csv_path, images_dir):
        self.csv_path = Path(csv_path)
        self.images_dir = Path(images_dir)

    def load_and_enhance_data(self):
        df = pd.read_csv(self.csv_path)
        df['text_description'] = (
            df['Product Name'].fillna('') + ' ' +
            df['Product Description'].fillna('') + ' in ' +
            df['Color'].fillna('') + ' color'
        ).str.strip()
        df['image_folder'] = df['ID'].astype(str).apply(lambda x: self.images_dir / x)
        df = df[(df['Availability']=='In Stock') & (df['text_description'].str.len()>10)]
        return df.reset_index(drop=True)

    def validate_and_select_images(self, df):
        records = []
        for _, r in df.iterrows():
            folder = r['image_folder']
            if not folder.exists(): continue
            imgs = list(folder.glob('*.jpg')) + list(folder.glob('*.png')) + list(folder.glob('*.jpeg'))
            if not imgs: continue
            records.append({
                'id': r['ID'],
                'text_description': r['text_description'],
                'image_path': str(imgs[0])
            })
        return records
