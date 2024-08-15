import tiktoken
from config import GPTConfig
from model import GPT
import torch
from data import DataLoader

num_return_sequences = 5
max_length = 30

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model = GPT(GPTConfig())
model.eval()
model.to(device)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

tokenizer = tiktoken.get_encoding('gpt2')


def test_spongebob():
    tokens = torch.tensor(tokenizer.encode(
        "spongebob squarepants"), dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)
    while (x.size(1) < max_length):
        with torch.no_grad():
            logits, loss = model(x)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(
                probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        print(">", tokenizer.decode(x[i, :max_length].tolist()))


test_spongebob()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
dl = DataLoader('merged_transcripts.txt')


# train
for i in range(3000):
    optimizer.zero_grad()
    values, targets = dl.get_next_batch(B=16, T=64)
    values = values.to(device)
    targets = targets.to(device)
    logits, loss = model(values, targets)
    loss.backward()
    optimizer.step()
    print(loss.item())

# save model
torch.save(model.state_dict(), 'model.pth')

# rerun test
test_spongebob()
