import tiktoken
import torch


class DataLoader:
    def __init__(self, file_path, B, T, num_workers):
        self.B = B
        self.T = T
        self.file_path = file_path
        self.num_workers = num_workers
        self.current_index = self.B * self.T * num_workers
        text = open(self.file_path, 'r', encoding='utf-8').read()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        print("Loaded {} tokens".format(len(self.tokenizer.encode(text))))
        print("Batches per full pass: {}".format(
            len(self.tokenizer.encode(text)) // (self.B * self.T)))
        self.data = self.tokenizer.encode(text)

    def get_next_batch(self):
        buffer = torch.tensor(
            self.data[self.current_index:self.current_index + self.B * self.T + 1])
        labels, targets = buffer[:-1].view(self.B,
                                           self.T), buffer[1:].view(self.B, self.T)
        self.current_index += self.B * self.T * self.num_workers + 1
        if self.current_index + self.B * self.T + 1 > len(self.data):
            self.current_index = 0
        return labels, targets
