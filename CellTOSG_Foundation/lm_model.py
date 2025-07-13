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
            # Adaptive pooling to seq_emb_dim dimensions (pooling over the hidden dimension)
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


import os
import torch
from tqdm import tqdm
from typing import List
from .dna_gpt.dna_gpt import DNAGPT
from .dna_gpt.tokenizer import KmerTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class RNAGPT_LM:
    def __init__(self, model_path: str ="", model_name: str = "dna_gpt0.1b_h", device: str = "cpu"):
        """
        RNAGPT-based language model.
        Args:
            model_name (str): Name of the RNAGPT model. Defaults to 'dna_gpt0.1b_h'.
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def get_model(self, model_name):
        """
        Initialize the RNAGPT model and tokenizer.
        """
        special_tokens = (
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] +
            ["+", '-', '*', '/', '=', "&", "|", "!"] +
            ['M', 'B'] + ['P'] + ['R', 'I', 'K', 'L', 'O', 'Q', 'S', 'U', 'V'] + ['W', 'Y', 'X', 'Z']
        )
        if model_name in ('dna_gpt0.1b_h'):
            tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=False)
        else:
            tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=True)

        vocab_size = len(tokenizer)
        model = DNAGPT.from_name(model_name, vocab_size)
        return model, tokenizer

    def load_model(self):
        """
        Load the RNAGPT model and tokenizer from the specified model name.
        """
        self.model, self.tokenizer = self.get_model(self.model_name)
        weight_path = os.path.join(self.model_path, f"{self.model_name}.pth")
        self._load_model_weights(weight_path)
        return self

    def _load_model_weights(self, weight_path, dtype=None):
        """
        Load the model weights from a checkpoint file.
        """
        state = torch.load(weight_path, map_location="cpu")
        if 'model' in state.keys():
            self.model.load_state_dict(state['model'], strict=False)
        else:
            self.model.load_state_dict(state, strict=False)
        print(f"loading model weights from {weight_path}")
        self.model.to(device=self.device, dtype=dtype)
        self.model.eval()

    def replace_rna_to_dna(self, sequence):
        # If the sequence is a float (or any non-string), return 'N'.
        if isinstance(sequence, float):
            return 'N'
        # Otherwise, replace 'U' with 'T' in the string.
        return sequence.replace('U', 'T')
    
    def generate_embeddings(self, sequences: List[str], batch_size: int = 16, max_len: int = 256, seq_emb_dim: int = 64) -> np.ndarray:
        """
        Generate tensor embeddings for each RNA sequence. 
        Args:
            sequences (List[str]): List of RNA sequences.
            batch_size (int): Batch size for processing sequences. Defaults to 16.
            max_len (int): Maximum length of the sequences. Defaults to 256.
            seq_emb_dim (int): Dimension to which embeddings are pooled. Defaults to 64.
        Returns:
            np.ndarray: A numpy array of shape (num_sequences, seq_emb_dim)
        """
        print(f"Generating embeddings for {len(sequences)} sequences.")
        device = self.device
        dataset = SentenceDataset(sequences)
        # Override collate_fn to return the raw list of sequences
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

        embeddings = []
        for batch in tqdm(dataloader, desc="Embedding sequences", unit="sequence"):
            # Tokenize and prepare input tensors as a list of tensors.
            batch_tokenized = [
                torch.tensor(
                    self.tokenizer.encode(
                        self.replace_rna_to_dna(seq),
                        max_len=max_len,
                        device=device
                    ),
                    dtype=torch.long
                )
                for seq in batch
            ]
            # Pad sequences in the batch to the same length
            input_ids = torch.nn.utils.rnn.pad_sequence(batch_tokenized, batch_first=True, padding_value=0).to(device)
            max_new_tokens = max_len - input_ids.shape[1]

            # Forward pass to compute embeddings
            with torch.no_grad():
                outputs = self.model(input_ids, max_new_tokens)
                # Compute the sequence embedding (mean pooling along token dimension)
                sequence_embedding = torch.mean(outputs, dim=1).squeeze()
                # Adaptive pooling to the desired dimension (pooling over the hidden dimension)
                projected = torch.nn.functional.adaptive_avg_pool1d(sequence_embedding.unsqueeze(1), output_size=seq_emb_dim).squeeze(1)
                embeddings.append(projected)

        # Concatenate embeddings from all batches along the first dimension
        embeddings_tensor = torch.cat(embeddings, dim=0)
        print(f"Generated embeddings shape: {embeddings_tensor.shape}")

        # Convert to numpy array and return
        return embeddings_tensor.cpu().numpy()


class ProtGPT_LM:
    def __init__(self, model_name: str = "prot_gpt2", device: str = "cpu"):
        """
        ProtGPT2-based language model.
        Args:
            model_name (str): Name of the ProtGPT2 model. Defaults to 'prot_gpt2'.
            device (str): Device to run the model on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def get_model(self, model_name):
        """
        Initialize the ProtGPT2 model and tokenizer.
        """
        model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        return model, tokenizer

    def load_model(self):
        """
        Load the ProtGPT2 model and tokenizer from the specified model name.
        """
        self.model, self.tokenizer = self.get_model(self.model_name)
        return self

    def generate_embeddings(self, sequences: List[str], seq_emb_dim: int = 64) -> torch.Tensor:
        """
        Generate tensor embeddings for each sequence with gradients enabled.
        Args:
            sequences (List[str]): List of DNA sequences.
        Returns:
            torch.Tensor: A tensor of shape (num_sequences, embedding_dim) where
                        num_sequences is the number of sequences and
                        embedding_dim is the dimension of the reduced embeddings.
        """
        print(f"Generating embeddings for {len(sequences)} sequences.")
        device = self.device
        self.model.to(device)
        embeddings = []

        for sequence in tqdm(sequences, desc="Embedding sequences", unit="sequence"):
            try:
                # Tokenize and prepare input tensors
                tokenized = self.tokenizer.encode(sequence)
                input_ids = torch.tensor(tokenized).unsqueeze(0).to(device)

                # Truncate input_ids to max length if needed
                max_length = self.model.config.n_positions  # Max length of the model
                input_ids = input_ids[:, :max_length]

                # Forward pass to compute embeddings
                with torch.no_grad():
                    outputs = self.model(input_ids, labels=input_ids)
                    loss, logits = outputs[:2]

                    # Compute the sequence embedding (mean pooling along token dimension)
                    sequence_embedding = torch.mean(logits, dim=1).squeeze()
                    # Use pooling to reduce the embedding dimension to 1
                    projected = torch.nn.functional.adaptive_avg_pool2d(sequence_embedding.unsqueeze(0).unsqueeze(0), (1, seq_emb_dim)).squeeze()
                    embeddings.append(projected)

            except Exception as e:
                print(f"Error processing sequence: {sequence}. Exception: {e}")
                continue

        # Stack embeddings into a single tensor (shape: [num_sequences, embedding_dim])
        embeddings_tensor = torch.stack(embeddings).reshape(-1, seq_emb_dim)
        print(f"Generated embeddings shape: {embeddings_tensor.shape}")
        return embeddings_tensor.cpu().numpy()