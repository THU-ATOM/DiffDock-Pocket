import numpy as np
import tqdm
import os
import tempfile

"""
    Preprocessing for the SO(2)/torus sampling and score computations, truncated infinite series are computed and then
    cached to memory, therefore the precomputation is only run the first time the repository is run on a machine
"""


def p(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


def grad(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_


X_MIN, X_N = 1e-5, 5000  # relative to pi
SIGMA_MIN, SIGMA_MAX, SIGMA_N = 3e-3, 2, 5000  # relative to pi

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi

current_file_path = os.path.abspath(__file__)
_source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_env_run_dir = os.environ.get('SO3_CACHE_DIR')
if _env_run_dir:
    cache_dir = os.path.abspath(_env_run_dir)
else:
    cache_dir = _source_dir

def _save_with_fallback(path, arr):
    """Try to save `arr` to `path`. On permission error, save to tmp dir instead and
    return the actual path used."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, arr)
        return path
    except PermissionError:
        tmp = tempfile.gettempdir()
        alt = os.path.join(tmp, os.path.basename(path))
        print(f"权限错误: 无法写入 {path}，退回到临时目录 {alt}")
        np.save(alt, arr)
        return alt

_p_path = os.path.join(cache_dir, '.p.npy')
_score_path = os.path.join(cache_dir, '.score.npy')

if os.path.exists(_p_path):
    p_ = np.load(_p_path)
    score_ = np.load(_score_path)
else:
    p_ = p(x, sigma[:, None], N=100)
    _save_with_fallback(_p_path, p_)

    score_ = grad(x, sigma[:, None], N=100) / p_
    _save_with_fallback(_score_path, score_)

def score(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    sign = np.sign(x)
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return -sign * score_[sigma, x]


def p(x, sigma):
    x = (x + np.pi) % (2 * np.pi) - np.pi
    x = np.log(np.abs(x) / np.pi)
    x = (x - np.log(X_MIN)) / (0 - np.log(X_MIN)) * X_N
    x = np.round(np.clip(x, 0, X_N)).astype(int)
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return p_[sigma, x]


def sample(sigma):
    out = sigma * np.random.randn(*sigma.shape)
    out = (out + np.pi) % (2 * np.pi) - np.pi
    return out


score_norm_ = score(
    sample(sigma[None].repeat(10000, 0).flatten()),
    sigma[None].repeat(10000, 0).flatten()
).reshape(10000, -1)
score_norm_ = (score_norm_ ** 2).mean(0)


def score_norm(sigma):
    sigma = np.log(sigma / np.pi)
    sigma = (sigma - np.log(SIGMA_MIN)) / (np.log(SIGMA_MAX) - np.log(SIGMA_MIN)) * SIGMA_N
    sigma = np.round(np.clip(sigma, 0, SIGMA_N)).astype(int)
    return score_norm_[sigma]
