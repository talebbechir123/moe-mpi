"""MoE GPT with expert parallelism over MPI.

mpirun -np 2 python moe_ep.py --mode {ep,dp,dense} --data FILE --tokenizer tok.json [--val FILE]
"""
import argparse, math, time, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpi4py import MPI

comm = MPI.COMM_WORLD
RANK, WORLD = comm.Get_rank(), comm.Get_size()
TIMERS = {"a2a": 0.0, "allreduce": 0.0}
A2A_BYTES = [0]


def alltoallv(x, send_counts, recv_counts):
    # x: (sum(send_counts), d) -> (sum(recv_counts), d)
    d = x.shape[1]
    send = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    recv = np.empty((int(sum(recv_counts)), d), dtype=np.float32)
    sc = np.asarray(send_counts, dtype=np.int64) * d
    rc = np.asarray(recv_counts, dtype=np.int64) * d
    sd = np.concatenate([[0], np.cumsum(sc)[:-1]])
    rd = np.concatenate([[0], np.cumsum(rc)[:-1]])
    t = time.perf_counter()
    comm.Alltoallv([send, (sc, sd), MPI.FLOAT], [recv, (rc, rd), MPI.FLOAT])
    TIMERS["a2a"] += time.perf_counter() - t
    A2A_BYTES[0] += send.nbytes
    return torch.from_numpy(recv)


class AllToAll(torch.autograd.Function):
    # backward = the reverse exchange

    @staticmethod
    def forward(ctx, x, send_counts, recv_counts):
        ctx.sc, ctx.rc = send_counts, recv_counts
        return alltoallv(x, send_counts, recv_counts)

    @staticmethod
    def backward(ctx, g):
        return alltoallv(g.contiguous(), ctx.rc, ctx.sc), None, None


def allreduce_grads(params, scale):
    # None -> zeros so every rank sends the same buffer size
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    grads = [p.grad for p in params]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads]).numpy()
    out = np.empty_like(flat)
    t = time.perf_counter()
    comm.Allreduce(flat, out, op=MPI.SUM)
    TIMERS["allreduce"] += time.perf_counter() - t
    out = torch.from_numpy(out) * scale
    i = 0
    for g in grads:
        n = g.numel()
        g.copy_(out[i:i + n].view_as(g))
        i += n


class Expert(nn.Module):
    def __init__(self, d, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w1 = nn.Parameter(torch.randn(d, 4 * d, generator=g) * 0.02)
        self.w2 = nn.Parameter(torch.randn(4 * d, d, generator=g) * 0.02)

    def forward(self, x):
        return F.gelu(x @ self.w1) @ self.w2


class MoE(nn.Module):
    def __init__(self, d, n_experts, layer_id, mode, cap_factor):
        super().__init__()
        self.E, self.mode, self.cap_factor = n_experts, mode, cap_factor
        self.router = nn.Linear(d, n_experts, bias=False)
        assert n_experts % WORLD == 0, "experts must be divisible by ranks"
        self.per_rank = n_experts // WORLD
        ids = range(n_experts) if mode == "dp" else \
            range(RANK * self.per_rank, (RANK + 1) * self.per_rank)
        self.ids = list(ids)
        self.experts = nn.ModuleList(
            Expert(d, seed=10_000 + 100 * layer_id + e) for e in self.ids)
        self.stats = None

    def forward(self, x):
        B, T, d = x.shape
        h = x.reshape(-1, d)
        N = h.shape[0]
        probs = F.softmax(self.router(h), dim=-1)            # [N, E]
        gate, eid = probs.max(dim=-1)                         # top-1 routing

        # per-expert capacity
        cap = math.ceil(self.cap_factor * N / self.E)
        onehot = F.one_hot(eid, self.E)
        pos = (torch.cumsum(onehot, 0) - 1).gather(1, eid[:, None]).squeeze(1)
        keep = pos < cap

        # Switch aux loss
        frac = onehot.float().mean(0)
        self.aux = self.E * (frac * probs.mean(0)).sum()

        kept = torch.nonzero(keep).squeeze(1)
        order = kept[torch.argsort(eid[kept], stable=True)]   # grouped by expert
        e_sorted = eid[order]
        tokens = h[order]
        counts_e = torch.bincount(e_sorted, minlength=self.E)  # per expert

        if self.mode == "dp":
            outs, i = [], 0
            for e, c in enumerate(counts_e.tolist()):
                outs.append(self.experts[e](tokens[i:i + c])); i += c
            y_sorted = torch.cat(outs)
        else:
            P, L = WORLD, self.per_rank
            send_e = counts_e.numpy().astype(np.int64)        # [P*L]
            recv_e = np.empty_like(send_e)
            comm.Alltoall(send_e, recv_e)                     # [src][local e]
            send_r = send_e.reshape(P, L).sum(1)
            recv_r = recv_e.reshape(P, L).sum(1)

            recv = AllToAll.apply(tokens, send_r.tolist(), recv_r.tolist())

            # recv layout: [src][local expert]
            cnt = recv_e.reshape(P, L)
            starts = np.concatenate([[0], np.cumsum(cnt.reshape(-1))[:-1]]).reshape(P, L)
            # keep out in the graph when this rank gets no tokens, or backward a2a desyncs
            out = recv * 0.0
            for le in range(L):
                idx = torch.cat([torch.arange(starts[s, le], starts[s, le] + cnt[s, le])
                                 for s in range(P)]).long()
                if idx.numel():
                    out = out.index_put((idx,), self.experts[le](recv[idx]))
            y_sorted = AllToAll.apply(out, recv_r.tolist(), send_r.tolist())

        y = torch.zeros_like(h).index_put((order,), y_sorted * gate[order, None])
        self.stats = (counts_e.numpy(), int(N - keep.sum()))
        return y.view(B, T, d)


class DenseMLP(nn.Module):
    # dense baseline: one MLP the size of one expert

    def __init__(self, d, layer_id):
        super().__init__()
        self.mlp = Expert(d, seed=10_000 + 100 * layer_id)
        self.cap_factor, self.aux, self.stats = 0.0, torch.zeros(()), (np.zeros(1), 0)

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(self, d, heads, n_experts, layer_id, mode, cap):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.moe = DenseMLP(d, layer_id) if mode == "dense" else \
            MoE(d, n_experts, layer_id, mode, cap)

    def forward(self, x, mask):
        a = self.ln1(x)
        x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
        return x + self.moe(self.ln2(x))


class MoEGPT(nn.Module):
    def __init__(self, vocab, d, heads, layers, n_experts, ctx, mode, cap):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads, n_experts, i, mode, cap)
                                    for i in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        T = idx.shape[1]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        x = self.tok(idx) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))


def load_text(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    words = "the cat dog sat ran on a mat park big small happy sun tree".split()
    rng = np.random.default_rng(0)
    return " ".join(rng.choice(words, 200_000)) + "."


@torch.no_grad()
def generate(model, encode, decode, prompt, n, ctx, n_experts, temp=0.8, top_k=50):
    # collective: all ranks call it, rank 0 samples and broadcasts
    model.eval()
    for b in model.blocks:
        b.moe.cap_factor = n_experts
    ids = encode(prompt) or [0]
    for _ in range(n):
        x = torch.tensor([ids[-ctx:]])
        logits = model(x)[0, -1] / temp
        v, i = torch.topk(logits, top_k)
        nxt = int(i[torch.multinomial(F.softmax(v, -1), 1)]) if RANK == 0 else 0
        ids.append(comm.bcast(nxt, root=0))
    model.train()
    return decode(ids)


@torch.no_grad()
def evaluate(model, val, a):
    # fixed held-out batches, no capacity drops
    model.eval()
    caps = [b.moe.cap_factor for b in model.blocks]
    for b in model.blocks:
        b.moe.cap_factor = a.experts
    g = torch.Generator().manual_seed(12345 + RANK)
    tot = 0.0
    for _ in range(a.eval_batches):
        ix = torch.randint(len(val) - a.ctx - 1, (a.batch,), generator=g)
        x = torch.stack([val[i:i + a.ctx] for i in ix])
        y = torch.stack([val[i + 1:i + a.ctx + 1] for i in ix])
        logits = model(x)
        tot += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
    for b, c in zip(model.blocks, caps):
        b.moe.cap_factor = c
    model.train()
    return comm.allreduce(tot / a.eval_batches, op=MPI.SUM) / WORLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ep", "dp", "dense"], default="ep")
    ap.add_argument("--data", default="")
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16, help="per rank")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100, help="linear lr warmup steps")
    ap.add_argument("--cap", type=float, default=1.25, help="capacity factor")
    ap.add_argument("--aux", type=float, default=0.01, help="balance loss weight")
    ap.add_argument("--log", default="")
    ap.add_argument("--tokenizer", default="", help="tok.json from train_tokenizer.py")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--gen", type=int, default=100, help="tokens to generate at the end")
    ap.add_argument("--val", default="", help="held-out text file for validation loss")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=20, help="per rank")
    a = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // WORLD))

    if a.tokenizer:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tokenizer)
        vocab = tk.get_vocab_size()
        cache = (a.data or "synthetic") + ".tok.npy"
        if RANK == 0 and not os.path.exists(cache):   # cached
            lines = load_text(a.data).splitlines(keepends=True)
            ids = []
            for i in range(0, len(lines), 10_000):
                for e in tk.encode_batch(lines[i:i + 10_000]):
                    ids.extend(e.ids)
            np.save(cache, np.asarray(ids, dtype=np.int32))
        comm.Barrier()
        data = torch.from_numpy(np.load(cache).astype(np.int64))
        encode = lambda s: tk.encode(s).ids
        decode = lambda ids: tk.decode(ids)
    else:
        text = load_text(a.data)
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        vocab = len(chars)
        data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda ids: "".join(chars[i] for i in ids)
    if RANK == 0:
        print(f"vocab={vocab} training tokens={len(data):,}")
    val = None
    if a.val:
        vcache = a.val + ".tok.npy"
        if RANK == 0 and not os.path.exists(vcache):
            if a.tokenizer:
                lines, ids = load_text(a.val).splitlines(keepends=True), []
                for i in range(0, len(lines), 10_000):
                    for e in tk.encode_batch(lines[i:i + 10_000]):
                        ids.extend(e.ids)
            else:
                ids = encode(load_text(a.val))
            np.save(vcache, np.asarray(ids, dtype=np.int32))
        comm.Barrier()
        val = torch.from_numpy(np.load(vcache).astype(np.int64))
        if RANK == 0:
            print(f"validation tokens={len(val):,}")

    torch.manual_seed(0)  # same init on every rank
    model = MoEGPT(vocab, a.d, a.heads, a.layers, a.experts, a.ctx, a.mode, a.cap)
    expert_p = [p for n, p in model.named_parameters() if ".experts." in n]
    shared_p = [p for n, p in model.named_parameters() if ".experts." not in n]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    n_exp = sum(p.numel() for p in expert_p)
    n_sh = sum(p.numel() for p in shared_p)
    if RANK == 0:
        print(f"mode={a.mode} ranks={WORLD} experts={a.experts} "
              f"params/rank: shared={n_sh:,} experts={n_exp:,}")

    log, vlog, t0 = [], [], time.perf_counter()
    prev = (0.0, 0.0, 0.0, 0)   # elapsed, a2a, allreduce, step
    for step in range(1, a.steps + 1):
        g = torch.Generator().manual_seed(1000 * step + RANK)
        ix = torch.randint(len(data) - a.ctx - 1, (a.batch,), generator=g)
        x = torch.stack([data[i:i + a.ctx] for i in ix])
        y = torch.stack([data[i + 1:i + a.ctx + 1] for i in ix])

        logits = model(x)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        aux = sum(b.moe.aux for b in model.blocks) / len(model.blocks)
        loss = ce + a.aux * aux
        opt.zero_grad()
        loss.backward()

        # ep: expert grads already summed over ranks
        allreduce_grads(shared_p, 1.0 / WORLD)
        if a.mode == "dp":
            allreduce_grads(expert_p, 1.0 / WORLD)
        else:
            for p in expert_p:
                if p.grad is not None:
                    p.grad /= WORLD
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1.0, step / a.warmup)
        opt.step()

        ce_g = comm.allreduce(ce.item(), op=MPI.SUM) / WORLD
        if step % 10 == 0 or step == 1:
            counts = sum(b.moe.stats[0] for b in model.blocks)
            counts = comm.allreduce(counts, op=MPI.SUM)
            dropped = comm.allreduce(sum(b.moe.stats[1] for b in model.blocks))
            el = time.perf_counter() - t0
            tok_s = step * a.batch * a.ctx * WORLD / el
            a2a = comm.allreduce(TIMERS["a2a"], op=MPI.MAX)
            ar = comm.allreduce(TIMERS["allreduce"], op=MPI.MAX)
            imb = counts.max() / max(counts.mean(), 1e-9)
            dt = el - prev[0]
            tok_s = (step - prev[3]) * a.batch * a.ctx * WORLD / dt
            row = dict(step=step, loss=round(ce_g, 4), tok_s=round(tok_s),
                       a2a_pct=round(100 * (a2a - prev[1]) / dt, 1),
                       allreduce_pct=round(100 * (ar - prev[2]) / dt, 1),
                       imbalance=round(float(imb), 2), dropped=int(dropped))
            prev = (el, a2a, ar, step)
            log.append(row)
            if RANK == 0:
                print(row, flush=True)

        if val is not None and (step % a.eval_every == 0 or step == a.steps):
            t_e, saved = time.perf_counter(), dict(TIMERS)
            vl = evaluate(model, val, a)
            TIMERS.update(saved)
            t0 += time.perf_counter() - t_e  # exclude eval from timing
            vlog.append(dict(step=step, val_loss=round(vl, 4)))
            if RANK == 0:
                print({"step": step, "val_loss": round(vl, 4)}, flush=True)

    # per-rank checkpoint (ep: local experts only)
    torch.save(model.state_dict(), f"ckpt_{a.mode}_rank{RANK}.pt")
    if a.gen > 0:
        text = generate(model, encode, decode, a.prompt, a.gen, a.ctx, a.experts)
        if RANK == 0:
            print("\n--- sample ---\n" + text + "\n--------------")

    if RANK == 0:
        mb = comm.allreduce(A2A_BYTES[0]) / 1e6
        print(f"total all-to-all volume: {mb:.1f} MB")
        if a.log:
            with open(a.log, "w") as f:
                json.dump(dict(args=vars(a), ranks=WORLD, log=log, val=vlog), f, indent=1)
    else:
        comm.allreduce(A2A_BYTES[0])


if __name__ == "__main__":
    main()
