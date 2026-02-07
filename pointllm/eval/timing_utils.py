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
            _sync()
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
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"time_module_forward expects nn.Module, got {type(module)} for '{name}'")

    timer = DeviceTimer(name)

    def _pre(*a, **k):
        timer.__enter__()

    def _post(*a, **k):
        timer.__exit__(None, None, None)

    h1 = module.register_forward_pre_hook(lambda *a, **k: _pre())
    h2 = module.register_forward_hook(lambda *a, **k: _post())
    return timer, (h1, h2)


def _is_valid_module(x):
    return (x is not None) and isinstance(x, torch.nn.Module)


def _list_child_modules(obj, limit=80):
    names = []
    for name, mod in obj.named_children():
        if isinstance(mod, torch.nn.Module):
            names.append(name)
    names = sorted(names)
    if len(names) > limit:
        names = names[:limit] + ["..."]
    return names


def find_attr(obj, candidates):
    """
    Return (module, name) where module is guaranteed to be a non-None nn.Module.
    IMPORTANT: hasattr(obj, nm) is NOT enough because many repos set attributes to None.
    """
    # 1) strict attribute lookup by candidate names
    for nm in candidates:
        if hasattr(obj, nm):
            try:
                v = getattr(obj, nm)
            except Exception:
                continue
            if _is_valid_module(v):
                return v, nm
            # 如果属性存在但为 None / 不是 Module：继续找，不要返回

    # 2) one-level children scan (common patterns)
    for name, mod in obj.named_children():
        low = name.lower()
        if any(k in low for k in ["encoder", "backbone", "point", "project", "projector", "proj"]):
            if _is_valid_module(mod):
                return mod, name

    # 3) dir() fuzzy scan (only accept real Modules)
    for name in dir(obj):
        low = name.lower()
        if any(k in low for k in ["encoder", "backbone", "point", "project", "projector", "proj"]):
            try:
                v = getattr(obj, name)
            except Exception:
                continue
            if _is_valid_module(v):
                return v, name

    # 4) fail with actionable debug info
    raise AttributeError(
        f"Cannot find a non-None nn.Module among candidates={candidates}. "
        f"Available top-level child modules: {_list_child_modules(obj)}. "
        f"(hint: print(model.get_model()) to see real submodule names)"
    )


def attach_timers(model, tokenizer):
    tok_timer = wrap_tokenizer(tokenizer)

    core = model.get_model()

    # encoder 常见命名（注意：很多 repo 有 point_backbone，但也可能叫 point_encoder）
    enc, enc_name = find_attr(
        core,
        ["point_encoder", "encoder", "point_backbone", "backbone"]
    )

    # projector 常见命名（你的 repo 里可能是 point_proj / mm_projector）
    prj, prj_name = find_attr(
        core,
        ["projector", "point_projector", "vision_projector", "mlp_projector", "mm_projector", "point_proj"]
    )

    enc_timer, enc_hooks = time_module_forward(enc, enc_name)
    prj_timer, prj_hooks = time_module_forward(prj, prj_name)

    return {"tokenizer": tok_timer, enc_name: enc_timer, prj_name: prj_timer}, list(enc_hooks) + list(prj_hooks)
