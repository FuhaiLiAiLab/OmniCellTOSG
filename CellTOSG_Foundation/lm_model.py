import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset

class SentenceDataset(Dataset):
    def __init__(self, sentences: List[str]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class TextEncoder():
    def __init__(self, model_path: str = "microsoft/deberta-v3-small", device: str = "cuda"):
        """
        Args:
            model_path (str, optional): Path to the deberta model. Defaults to 'microsoft/deberta-v3-small'.
            device (str, optional): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the deberta model and tokenizer from the specified model path.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)

    def generate_embeddings(self, sentences: List[str], batch_size: int = 32, text_emb_dim: int = 64) -> torch.Tensor:
        """
        Generate a single-dimensional embedding for each sentence.

        Args:
            sentences (List[str]): List of sentences to embed.
            batch_size (int, optional): Batch size for DataLoader. Defaults to 32.

        Returns:
            List[float]: List of single-dimensional embeddings.
        """
        dataset = SentenceDataset(sentences)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embedding_batches = []
        for batch in tqdm(dataloader, desc="Embedding sentences", unit="batch"):
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            batch_embeddings = torch.mean(outputs.last_hidden_state, dim=1).squeeze()
            # Adaptive pooling to text_emb_dim dimensions (pooling over the hidden dimension)
            projected = torch.nn.functional.adaptive_avg_pool1d(batch_embeddings.unsqueeze(1), output_size=text_emb_dim).squeeze(1)
            embedding_batches.append(projected)
        return torch.cat(embedding_batches, dim=0)

    def save_embeddings(self, embeddings: List[float], output_npy_path: str) -> None:
        """
        Save embeddings to a .npy file and IDs to a CSV file.

        Args:
            embeddings (List[float]): List of single-dimensional embeddings.
            ids (List[str]): List of corresponding IDs.
            output_npy_path (str): Path to save the .npy file.
            output_csv_path (str): Path to save the index CSV file.
        """
        # Save the embeddings to .npy file
        np.save(output_npy_path, np.array(embeddings))
        print(f"Embeddings saved at {output_npy_path} with shape {np.array(embeddings).shape}")