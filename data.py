import tiktoken
import torch


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.current_index = 0
        text = open(self.file_path, 'r', encoding='utf-8').read()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        print("Loaded {} tokens".format(len(self.tokenizer.encode(text))))
        self.data = self.tokenizer.encode(text)

    def get_next_batch(self, B, T):
        buffer = torch.tensor(
            self.data[self.current_index:self.current_index + B * T + 1])
        labels, targets = buffer[:-1].view(B, T), buffer[1:].view(B, T)
        self.current_index += B * T + 1
        if self.current_index >= len(self.data):
            self.current_index = 0
        return labels, targets
