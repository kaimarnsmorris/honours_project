# Speaker script — *Variance Reduction for Training Neural Bayes Estimators*

Kai Marns-Morris · Honours (BPhil) 2026

Notes: first-person, conversational. `[ ]` are stage cues, not spoken. Aim ~15 min.

---

## 1 · Title slide

Hi, I'm Kai, and I'll be presenting my thesis on **variance reduction for neural Bayes estimators**.

I'd like to start by sincerely thanking my supervisors, Michael Bertolacci and Ed Cripps, for all their help and guidance in supervising me over the past year.

[Move on.]

---

## 2 · Bayesian inference

Most Bayesian statistics starts by fixing a statistical model with some parameters — which I'll call **theta** — and a prior distribution, or initial guess, **p(theta)**.

Once we observe some real data, we update our belief about theta *conditioned* on that data, and this gives us the **posterior distribution**.

You can see it illustrated here: the data sharpens our initial guess into a tighter posterior. I've also marked the **posterior mean** — that's a common thread through my whole thesis, and it brings us to the next slide.

---

## 3 · Point estimation

Very often we don't actually want the entire posterior — we just want a single **point estimate**. Usually that's the posterior mean, though sometimes it's the median or a particular quantile.

The problem is that this is **rarely available in closed form**. So we need some *procedure* that takes in a new dataset **y** and returns an updated estimate of the parameters — I'll call that estimate **theta-hat**.

[Gesture at the y → box → theta-hat diagram.] The question is what goes in the box.

---

## 4 · Method (1): Markov chain Monte Carlo

The traditional "gold standard" answer is **MCMC sampling**. The idea is to draw samples from the posterior, build up a sample histogram, and use that to form our estimate — here, just the average of the samples.

The catch is that we usually can't sample the posterior directly, so instead we simulate a **Markov chain** whose *stationary distribution* is the posterior. That's accurate and general, but it can be **slow** to converge.

And crucially, the whole chain has to be **re-run from scratch for every new dataset**. So if you want to do inference on many datasets, you pay that full cost every single time.

---

## 5 · Method (2): Neural Bayes estimator

A newer approach is the **neural Bayes estimator**. Instead of re-running a sampler each time, we try to *learn* the mapping — straight from an arbitrary dataset to its posterior estimate — using a neural network, leaning on its **universal function-approximator** property.

The natural question is: how do we train such a network? And here's the nice part. Sampling *from* the posterior is hard — but generating **forward** samples of theta and y is generally really easy, because that's exactly how the models are defined.

So in each training step we sample a theta, use it to simulate a dataset y, feed that y through the network, and compare the network's output to the theta we started from. Backpropagating that error trains the network. [Beat.] The exact loss we use turns out to matter a lot, and I'll come back to it.

The big advantage is that once it's trained, any new dataset is a **single forward pass** — the cost is **amortised**. And my thesis is about ways to speed up that *training*, which matters most when the theta–y samples are expensive to generate.

---

## 6 · Does it work?

Let's sanity-check that this actually works on a toy problem. Here the data are normal, and the unknown parameters are the **mean** and the **variance**.

[Point at the plot.] This compares each method's estimate against the true value on held-out test data. The neural estimator (orange) sits right on top of MCMC (blue) — it's genuinely **learned the posterior-mean map**. We take only a slightly worse MSE, and in return the method is amortised, so every future dataset costs a tiny fraction of the time.

---

## 7 · The loss picks which summary you estimate

I promised I'd come back to the loss. The network minimises the **Bayes risk** — the expected loss over theta and y — and the loss function **ℓ** decides *which* posterior summary it ends up targeting.

[Point at the table.] Squared-error loss gives you the posterior **mean**; absolute error gives the **median**; the quantile ("pinball") loss gives you a **quantile**. Throughout this talk we use squared error, so we're always estimating the posterior mean.

Now here's the key observation. This Bayes risk is **not available in closed form**, so in practice we **estimate it by Monte Carlo** — we average the loss over a batch of simulated theta–y pairs.

---

## 8 · The idea

And that's the core idea of the whole thesis. Because we train on a **Monte Carlo estimate** of the true objective, the **gradient** that actually drives training is a **random quantity** — it has variance.

The important point is that this variance is about *how we sample the loss*, not about the inference problem itself. Which means it's the kind of noise **we can reduce** — using classic Monte Carlo variance-reduction techniques. The thesis explores two of them.

[Section transition → "Idea 1: Rao–Blackwellisation".]

---

## 9 · The trick  *(Rao–Blackwellisation)*

The first idea uses **conditional conjugacy**. Split the parameters into two blocks, theta-one and theta-two. In many models, if you *condition* on theta-two, the posterior of theta-one comes out in **closed form**.

Remember the network trains on (y, theta) pairs where the target theta is a **noisy draw**. The trick is to replace the conjugate block, theta-one, with its **exact conditional mean** — a cleaner label to train against.

This is classic **Rao–Blackwell**: replacing a sampled quantity by its conditional mean can only ever *reduce* variance. A concrete example is Bayesian linear regression — condition on the noise variance, and the coefficients are Gaussian, so their conditional mean is closed-form.

---

## 10 · Bayesian linear regression: a cleaner target

Let's make that concrete. [Point at the model.] Given the noise variance sigma-squared, both the coefficient mean and therefore the target's conditional mean are available in **closed form**.

If you write out the training loss and condition on sigma-squared and y, it splits into two pieces: a part the network can actually fit — the distance to the **clean target** — plus a term that's just **constant** in the network's parameters.

So instead of chasing a noisy sampled beta, we plug in the exact closed-form conditional mean. Rao–Blackwell then tells us the loss estimate and its **gradient** will both be **less noisy** — same mean, never larger variance.

---

## 11 · Does it pay off?

Does it actually help? Here's Bayesian linear regression with ten coefficients. [Point.]

The green line is the Bayes-optimal floor from an exact Gibbs sampler — the best any estimator could do. The Rao–Blackwellised training (in colour) reaches a **lower test error** at the **small and moderate batch sizes** — exactly where the training gradient is noisiest and a cleaner signal helps most.

As the batch grows the advantage shrinks, and by a batch of 256 the two curves meet — because with enough samples the ordinary gradient is already clean.

---

## 12 · In general

And nothing here was special to linear regression. Take *any* model, split off a **conditionally conjugate** block, and replace its sampled target with the exact conditional mean.

[Point at the diagram.] The Rao–Blackwellised gradient is just the ordinary one **conditioned** on theta-two and y — so by the law of total variance it can only shrink, **never grow**. That holds for any model and any architecture. Both routes estimate the *same* loss L; the reweighted one is simply **less noisy**.

[Section transition → "Idea 2: importance sampling".]

---

## 13 · A more general lever

The second idea is **importance sampling**, and unlike Rao–Blackwell it needs **no conjugacy** at all.

The idea: instead of drawing the parameters from the prior p, draw them from a **proposal** q, and then **reweight** by the ratio p-over-q so the risk is left completely unchanged. [Point at the identity.] Because it's unbiased for *any* proposal q, the trained estimator still targets the original prior — we're free to spend the simulation budget wherever q puts its mass.

The catch is there's **no guarantee**. A badly chosen proposal can actually *increase* the variance and collapse the effective sample size.

---

## 14 · A test case: AR(1)

We tested this on a first-order autoregressive model, with parameters rho and sigma, using a recurrent network and checking it against the exact Gibbs posterior.

The intuition was that the **extreme edges** — tiny sigma, or rho near plus-or-minus one — *look* hardest to estimate, so those are the natural place to spend extra draws. [Point to the right plot.] So we built a proposal that **oversamples** exactly those edges.

But look at the left panel [point]. The network already tracks the Gibbs reference across the whole prior. And the two edge cases we singled out are telling: the tiny-sigma one (the star) is genuinely **unidentifiable** — it's a flat, signal-free series — while the rho-near-minus-one case (the cross) is actually estimated **perfectly**. So the edges only *look* hard.

---

## 15 · Importance sampling: the result

And the result is negative. [Left plot.] Every proposal we tried **inflated** the training-gradient variance — the exact opposite of what we wanted, and the very quantity Rao–Blackwell shrinks. [Right plot.] Correspondingly, the test error dropped in **no region** — no proposal beat the plain uniform baseline.

Why does it fail here? Two reasons. The estimator is already sitting near the **Bayes floor**, so there's very little reducible error for *any* proposal to recover. And the regions we flooded only *looked* hard — those near-non-stationary edges are actually the **easiest** places to pin down rho, because the big swings carry the most information.

So the lesson is the contrast: Rao–Blackwell is unconditional, while importance sampling only helps when the proposal genuinely matches where the signal is.

---

## 16 · Conclusion

To wrap up: training a neural Bayes estimator is **itself a Monte Carlo procedure**, so its training variance is something we can attack directly.

**Rao–Blackwellisation** needs a conjugate block, but in return gives a *provably* lower-variance gradient for any model or architecture — and on linear regression it trained to a lower error, most of all where the batch was small.

**Importance sampling** needs nothing special and is far more general, but it's unguaranteed and problem-dependent — and on our AR(1) test it didn't help.

And there's plenty of **future work** — many more variance-reduction techniques left to try.

---

## 17 · Thank you

Thank you.

[The chair will run questions — don't invite them yourself.]
