import tiktoken
from config import GPTConfig
from model import GPT
import time
import torch
from data import DataLoader

num_return_sequences = 5
max_length = 30

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
torch.compile(model)
model.to(device)

B, T = 4, 1024
optimizer = torch.optim.AdamW(
    model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
dl = DataLoader('merged_transcripts.txt', B, T)

# train
for i in range(500):
    t0 = time.time()
    optimizer.zero_grad()
    values, targets = dl.get_next_batch()
    values = values.to(device)
    targets = targets.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(values, targets)
    loss.backward()
    # clip gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    tps = (B * T) / (time.time() - t0)

    print(f"loss: {loss.item()} dt: {(time.time() - t0)
          * 1000}ms tps: {tps} norm: {norm.item()}")

# tokenizer = tiktoken.get_encoding('gpt2')


# def test_spongebob():
#     tokens = torch.tensor(tokenizer.encode(
#         """spongebob squarepants
#         """), dtype=torch.long)
#     tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
#     x = tokens.to(device)
#     while (x.size(1) < max_length):
#         with torch.no_grad():
#             logits, loss = model(x)
#             logits = logits[:, -1, :]
#             probs = torch.nn.functional.softmax(logits, dim=-1)
#             topk_probs, topk_indices = torch.topk(
#                 probs, 50, dim=-1)
#             ix = torch.multinomial(topk_probs, 1)
#             xcol = torch.gather(topk_indices, -1, ix)
#             x = torch.cat((x, xcol), dim=1)
#     for i in range(num_return_sequences):
#         print(">", tokenizer.decode(x[i, :max_length].tolist()))


# # test_spongebob()

# save model
torch.save(model.state_dict(), 'model.pth')

# rerun test
# test_spongebob()
