# pointllm/eval/timing_utils.py
import time
import torch

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()

class DeviceTimer:
    """跨 CUDA / MPS / CPU 的统一计时器。"""
    def __init__(self, name="timer"):
        self.name = name
        self.elapsed_ms = []
        self.use_cuda = torch.cuda.is_available()
        self.use_mps = (hasattr(torch, "mps") and torch.backends.mps.is_available())

        # 仅 CUDA 用 Event；MPS/CPU 走 perf_counter
        if self.use_cuda:
            self.start_evt = torch.cuda.Event(enable_timing=True)
            self.end_evt = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self.start_evt.record()
        else:
            _sync()  # MPS/CPU 也先同步一下上一次提交
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.use_cuda:
            self.end_evt.record()
            self.end_evt.synchronize()
            ms = self.start_evt.elapsed_time(self.end_evt)
        else:
            _sync()
            ms = (time.perf_counter() - self.t0) * 1000.0
        self.elapsed_ms.append(ms)

    def stats(self):
        if not self.elapsed_ms:
            return {"n": 0, "mean_ms": 0.0, "std_ms": 0.0}
        arr = self.elapsed_ms
        m = sum(arr) / len(arr)
        v = sum((x - m) ** 2 for x in arr) / len(arr)
        return {"n": len(arr), "mean_ms": m, "std_ms": v ** 0.5}

def wrap_tokenizer(tokenizer):
    """包装 HF tokenizer 以统计 __call__ 的时间。"""
    timer = DeviceTimer("tokenizer")
    orig = tokenizer.__call__
    def timed(*args, **kwargs):
        with timer:
            return orig(*args, **kwargs)
    tokenizer.__call__ = timed
    return timer

def time_module_forward(module, name):
    """给 nn.Module 的 forward 注册前/后钩子以计时。"""
    timer = DeviceTimer(name)
    def _pre(*a, **k):  timer.__enter__()
    def _post(*a, **k): timer.__exit__(None, None, None)
    h1 = module.register_forward_pre_hook(lambda *a, **k: _pre())
    h2 = module.register_forward_hook(lambda *a, **k: _post())
    return timer, (h1, h2)

def find_attr(obj, candidates):
    for nm in candidates:
        if hasattr(obj, nm):
            return getattr(obj, nm), nm
    # fallback: try fuzzy search by substring
    # 先在一层 children 里找
    for name, mod in obj.named_children():
        low = name.lower()
        if any(k in low for k in ["project", "projector", "proj"]):
            return mod, name
    # 再扫 obj.__dict__ 里类似字段
    for name in dir(obj):
        if any(k in name.lower() for k in ["project", "projector", "proj"]):
            try:
                mod = getattr(obj, name)
                if isinstance(mod, torch.nn.Module):
                    return mod, name
            except Exception:
                pass
    raise AttributeError(f"Cannot find any of attributes: {candidates}  "
                         f"(hint: print(model.get_model()) to see submodules)")


def attach_timers(model, tokenizer):
    tok_timer = wrap_tokenizer(tokenizer)

    # encoder 常见命名
    enc, enc_name = find_attr(
        model.get_model(),
        ["point_encoder", "encoder", "point_backbone", "backbone"]
    )
    # projector 常见命名  ← 新增 'mm_projector'
    prj, prj_name = find_attr(
        model.get_model(),
        ["projector", "point_projector", "vision_projector", "mlp_projector", "mm_projector"]
    )

    enc_timer, enc_hooks = time_module_forward(enc, enc_name)
    prj_timer, prj_hooks = time_module_forward(prj, prj_name)
    return {"tokenizer": tok_timer, enc_name: enc_timer, prj_name: prj_timer}, list(enc_hooks)+list(prj_hooks)

