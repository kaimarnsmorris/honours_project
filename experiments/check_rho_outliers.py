"""Are the worst rho-error points GRU-specific (reducible) or shared with Gibbs
(irreducible)? Reuses net_base.pt and the cached Gibbs estimates."""
import numpy as np, torch
import run

A, T, N = 2.0, 100, 400
net = run.AR1Estimator(64, 64, A)
net.load_state_dict(torch.load("extra_out/net_base.pt", map_location="cpu"))
net.eval()

rng = np.random.default_rng(123)                 # identical to gibbs_vs_nn.py
rho = rng.uniform(-1.0, 1.0, N)
sigma = rng.uniform(0.0, A, N)
x = run.simulate_ar1(rho, sigma, T, rng)
with torch.no_grad():
    r_hat, s_hat = net(torch.as_tensor(x, dtype=torch.float32))
r_hat = r_hat.numpy(); s_hat = s_hat.numpy()

d = np.load("extra_out/gibbs_vs_nn_cache.npz")
g_rho = d["g_rho"]

err_nn = (r_hat - rho) ** 2
err_gb = (g_rho - rho) ** 2
order = np.argsort(err_nn)[::-1]
w = order[:10]

print("worst 10 rho-error cases (ranked by GRU):")
print(f"{'rho':>7}{'GRU':>8}{'Gibbs':>8}{'errNN':>9}{'errGibbs':>10}{'sigma':>7}  verdict")
for i in w:
    verdict = "both miss (irreducible)" if err_gb[i] > 0.5 * err_nn[i] else "GRU-specific (reducible)"
    print(f"{rho[i]:7.3f}{r_hat[i]:8.3f}{g_rho[i]:8.3f}{err_nn[i]:9.4f}{err_gb[i]:10.4f}{sigma[i]:7.2f}  {verdict}")

print(f"\nof the worst 10: |rho|>0.9 -> {int(np.sum(np.abs(rho[w])>0.9))} ; "
      f"|rho|>0.8 -> {int(np.sum(np.abs(rho[w])>0.8))}")
print(f"worst-10 mean rho-error:  GRU {err_nn[w].mean():.4f}   Gibbs {err_gb[w].mean():.4f}   "
      f"(ratio {err_nn[w].mean()/max(err_gb[w].mean(),1e-9):.2f}x)")

print("\nbinned mean rho-error (confirms the boundary is easiest on average):")
for lo, hi in [(-1,-0.8),(-0.8,-0.3),(-0.3,0.3),(0.3,0.8),(0.8,1.0)]:
    m = (rho >= lo) & (rho < hi)
    if m.sum():
        print(f"  rho in [{lo:+.1f},{hi:+.1f}): n={m.sum():3d}  "
              f"GRU {err_nn[m].mean():.4f}   Gibbs {err_gb[m].mean():.4f}")
