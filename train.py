import tiktoken
from config import GPTConfig
from model import GPT
import time
import torch
from data import DataLoader
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
torch.compile(model)
model.to(device)

total_batch_size = 524288  # 2^19
B, T = 4, 1024

assert total_batch_size % (
    B * T) == 0, "total_batch_size must be divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"Gradient Accumulation Steps: {grad_accum_steps}")

dl = DataLoader('merged_transcripts.txt', B, T)

max_steps = 50
# around 100ms per step
print(f"Max Steps: {max_steps}",
      f"Estimated Time: {100 * max_steps * grad_accum_steps / 1000} seconds")

max_lr = 6e-4
min_lr = max_lr / 10
warmup_steps = 10
optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=max_lr, device=device)


def get_lr(step):
    # warmup
    if step <= warmup_steps:
        return max_lr * step / warmup_steps
    if step > max_steps:
        return min_lr
    # cosine decay
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * step / max_steps))

    # train
for step in range(1, max_steps + 1):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0
    for mini_batch in range(grad_accum_steps):
        values, targets = map(lambda x: x.to(device), dl.get_next_batch())
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(values, targets)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    # clip gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    tps = (B * T) * grad_accum_steps / (time.time() - t0)

    print(f"| Step | Loss      | Duration (ms)  | TPS     | Grad Norm | Learning Rate |")
    print(f"|------|-----------|----------------|---------|-----------|---------------|")
    print(f"| {step:4d} | {loss_accum:9.4f} | {(time.time() - t0) *
          1000:14.2f} | {tps:7.0f} | {norm.item():9.4f} | {lr:13.6f} |")
    print(f"|------|-----------|----------------|---------|-----------|---------------|")

# tokenizer = tiktoken.get_encoding('gpt2')


def test_spongebob(num_return_sequences=5, max_length=30):
    tokens = torch.tensor(tokenizer.encode(
        """spongebob squarepants"""), dtype=torch.long)
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


# save model
torch.save(model.state_dict(), 'model.pth')

# rerun test
test_spongebob(5, 200)
