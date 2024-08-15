import tiktoken
from config import GPTConfig
from model import GPT
import time
import torch
from data import DataLoader
import math
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

max_lr = 6e-4
min_lr = max_lr / 10
warmup_steps = 10
max_steps = 50


def get_lr(step):
    # warmup
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    # cosine decay
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * step / max_steps))


    # train
for step in range(1, max_steps + 1):
    t0 = time.time()
    optimizer.zero_grad()
    values, targets = dl.get_next_batch()
    values = values.to(device)
    targets = targets.to(device)
    lr = get_lr(step)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits, loss = model(values, targets)
    loss.backward()
    # clip gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    tps = (B * T) / (time.time() - t0)

    print(f"""loss: {loss.item()} dt: {(time.time() - t0)
          * 1000}ms tps: {tps} norm: {norm.item()} lr: {lr}""")

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
