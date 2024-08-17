import tiktoken
from config import GPTConfig
from model import GPT
import time
import torch
from data import DataLoader
import math
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist

# set up distributed training
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    # nccl is not working on windows, use gloo when ssh'd on linux box

    # use the following lines to test on windows
    # os.environ["USE_LIBUV"] = '0'
    # init_process_group(backend='gloo')
    # use the following line to test on linux
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ.get('RANK'))
    ddp_world_size = int(os.environ.get('WORLD_SIZE'))
    ddp_local_rank = int(os.environ.get('LOCAL_RANK'))
    device = f'cuda:{ddp_local_rank}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # first process does logging
else:
    master_process = True
    ddp_world_size = 1
    ddp_rank = 0
    ddp_local_rank = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('high')

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module
else:
    raw_model = model
total_batch_size = 524288  # 2^19
B, T = 16, 1024

assert total_batch_size % (
    B * T * ddp_world_size) == 0, "total_batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

dl = DataLoader('merged_transcripts.txt', B, T, ddp_world_size)

max_steps = 1000
if master_process:
    print(f"Gradient Accumulation Steps: {grad_accum_steps}")

max_lr = 6e-4
min_lr = max_lr / 10
warmup_steps = 100

optimizer = raw_model.configure_optimizers(
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
        # dont synchronization until last step
        # this is so ugly
        if ddp:
            if (mini_batch % grad_accum_steps == 0):
                loss.backward()
            else:
                with model.no_sync():
                    loss.backward()
            dist.all_reduce(loss_accum, op=dist.reduce_op.AVG)
        else:
            loss.backward()
    # clip gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0)
    tokens_processed = (B * T) * grad_accum_steps * ddp_world_size
    tps = tokens_processed / dt
    if master_process:
        print("Estimated time remaining (s): ",
              (max_steps - step) * dt)
        print(
            f"| Step | Loss      | Duration (ms)  | TPS     | Grad Norm | Learning Rate |")
        print(
            f"|------|-----------|----------------|---------|-----------|---------------|")
        print(f"| {step:4d} | {loss_accum:9.4f} | {dt*1000:14.2f} | {tps:7.0f} | {norm.item():9.4f} | {lr:13.6f} |")
        print(
            f"|------|-----------|----------------|---------|-----------|---------------|")


def test_spongebob(num_return_sequences=5, max_length=30):
    tokenizer = tiktoken.get_encoding('gpt2')
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
