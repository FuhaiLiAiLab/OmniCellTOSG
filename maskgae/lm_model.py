from transformers import AutoTokenizer, AutoModel

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