from torch.utils.data import Dataset
from PIL import Image

class KhaadiFashionDataset(Dataset):
    def __init__(self, data_list, processor, max_length=77):
        self.data = data_list
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['image_path']).convert('RGB')
        if max(img.size) > 512:
            img.thumbnail((512,512), Image.Resampling.LANCZOS)

        inputs = self.processor(
            text=[item['text_description']],
            images=[img],
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'product_id': item['id']
        }
