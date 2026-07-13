# Speaker script — *Variance Reduction for Training Neural Bayes Estimators*

Kai Marns-Morris · Honours (BPhil) 2026

Notes: first-person, conversational. `[ ]` are stage cues, not spoken. Aim ~15 min.

---

## 1 · Title slide

Hi, I'm Kai, and I'll be presenting my thesis on **variance reduction for neural Bayes estimators**.

I'd like to start by sincerely thanking my supervisors, Michael Bertolacci and Ed Cripps, for all their help and guidance in supervising me over the past year.

[Move on.]

---

## Overview  *(roadmap slide — right after the title)*

Before the details, here's the whole talk in one slide. We'll build **neural Bayes estimators** — networks that map a dataset straight to its posterior estimate, giving fast, *amortised* inference.

The catch — and the theme of the thesis — is that *training* one is itself a **Monte Carlo** problem, so the gradient that drives training is **noisy**. I'll treat that as a **variance-reduction** problem: cut the noise, and train better.

I try two classic techniques. **Rao–Blackwellisation** — provably lower-variance, but it needs conjugacy. And **importance sampling** — completely general, but with no guarantee it helps. Let's see how each does.

---

## 2 · Bayesian inference

Most Bayesian statistics starts by fixing a statistical model with some parameters — which I'll call **theta** — and a prior distribution, or initial guess, **p(theta)**.

Once we observe some real data, we update our belief about theta *conditioned* on that data, and this gives us the **posterior distribution**.

You can see it illustrated here: the data sharpens our initial guess into a tighter posterior. I've also marked the **posterior mean** — that's a common thread through my whole thesis, and it brings us to the next slide.

---

## 3 · Point estimation

Very often we don't actually want the entire posterior — we just want a single **point estimate**. Usually that's the posterior mean, though sometimes the median or a particular quantile.

The problem is that this is **rarely available in closed form**. So we need some *procedure* that takes in a new dataset **y** and returns an estimate of the parameters — I'll call that estimate **theta-hat**.

[Gesture at the y → box → theta-hat diagram.] The question is what goes in the box.

---

## 4 · The Bayes risk

But first — which point estimate should we actually return? Decision theory gives a clean answer: pick a **loss** function, and return the estimate that minimises the **Bayes risk** — the expected loss averaged over theta and y.

[Point at the table.] And the loss you choose decides *which* posterior summary is optimal: squared-error loss gives the posterior **mean**, absolute error gives the **median**, and the quantile — or "pinball" — loss gives a **quantile**. Throughout this talk we use squared error, so we're always after the posterior mean.

Now, normally you'd use this criterion the other way round — to *choose* which estimator to use. The twist in this thesis is that we'll instead use it as a **computational trick**. I'll come back to exactly how.

---

## 5 · Method (1): Markov chain Monte Carlo

The traditional "gold standard" way to get that posterior mean is **MCMC sampling**: draw samples from the posterior, build up a histogram, and average them to form the estimate.

The catch is we usually can't sample the posterior directly, so instead we simulate a **Markov chain** whose *stationary distribution* is the posterior. That's accurate and general, but it can be **slow** to converge.

And crucially, the whole chain has to be **re-run from scratch for every new dataset**. So if you want to do inference on many datasets, you pay that full cost every single time.

---

## 6 · Method (2): Neural Bayes estimator

A newer approach is the **neural Bayes estimator**. Instead of re-running a sampler each time, we *learn* the mapping — straight from a dataset to its posterior estimate — with a neural network, leaning on its **universal function-approximator** property.

So how do we train it? Here's the nice part. Sampling *from* the posterior is hard, but generating **forward** samples of theta and y is easy — that's exactly how the model is defined. So we simulate a theta, simulate a dataset y from it, push y through the network, and nudge the weights so the output matches the theta we started from — minimising that squared-error loss over many such pairs.

Once it's trained, any new dataset is a **single forward pass** — the cost is **amortised**. And my thesis is about speeding up that *training*, which matters most when the theta–y samples are expensive to generate.

---

## 7 · The idea: training is Monte Carlo

Let's come back to that Bayes risk. Our **goal** is to find the network weights gamma that minimise it — capital-L of gamma, the expected loss over theta and y.

But it has no closed form. So each training step estimates it by **Monte Carlo** on a minibatch — averaging the loss over a batch of simulated theta–y pairs — and takes a **stochastic-gradient** step downhill.

And here's the key **insight** — really the whole thesis in one line. That minibatch gradient is a *random estimate*: it carries variance. And that variance comes from *how we sample the loss*, not from the inference problem itself. So it's noise **we can reduce**, using classical Monte Carlo variance-reduction techniques. The thesis explores two of them.

[Section transition → "Idea 1: Rao–Blackwellisation".]

---

## 8 · The trick  *(Rao–Blackwellisation)*

The first technique is **Rao–Blackwellisation**. The principle is simple and completely general: if you replace a randomly-sampled quantity by its **conditional mean**, you keep the same expectation but can only ever **reduce** the variance.

Here's how we use it. The network's training target theta is a **noisy draw**. Split it into two blocks; condition on the second block, theta-two, and swap the noisy target for its conditional mean given theta-two — a **cleaner label**, at no cost to what we're estimating.

The one thing this needs is that the conditional mean be computable in **closed form** — which happens exactly when that block is **conditionally conjugate**. The concrete example is Bayesian linear regression: condition on the noise variance, and the coefficients are Gaussian, so their conditional mean is closed-form.

---

## 9 · Bayesian linear regression: a cleaner target

Let's make that concrete. [Point at the model.] Given the noise variance sigma-squared, both the coefficient mean and therefore the target's conditional mean are available in **closed form**.

If you write out the training loss and condition on sigma-squared and y, it splits into two pieces: a part the network can actually fit — the distance to the **clean target** — plus a term that's just **constant** in the network's parameters.

So instead of chasing a noisy sampled beta, we plug in the exact closed-form conditional mean. Rao–Blackwell then tells us the loss estimate and its **gradient** will both be **less noisy** — same mean, never larger variance.

---

## 10 · Does it pay off?

Does it actually help? Here's Bayesian linear regression with ten coefficients. [Point.]

The green line is the Bayes-optimal floor from an exact Gibbs sampler — the best any estimator could do. The Rao–Blackwellised training (in colour) reaches a **lower test error** at the **small and moderate batch sizes** — exactly where the training gradient is noisiest and a cleaner signal helps most.

As the batch grows the advantage shrinks, and by a batch of 256 the two curves meet — because with enough samples the ordinary gradient is already clean.

---

## 11 · In general

And nothing here was special to linear regression. Take *any* model, split off a **conditionally conjugate** block, and replace its sampled target with the exact conditional mean.

[Point at the diagram.] The Rao–Blackwellised gradient is just the ordinary one **conditioned** on theta-two and y — so by the law of total variance it can only shrink, **never grow**. That holds for any model and any architecture. Both routes estimate the *same* loss L; the reweighted one is simply **less noisy**.

[Section transition → "Idea 2: importance sampling".]

---

## 12 · A more general lever

The second idea is **importance sampling**, and unlike Rao–Blackwell it needs **no conjugacy** at all.

The idea: instead of drawing the parameters from the prior p, draw them from a **proposal** q, and then **reweight** by the ratio p-over-q so the risk is left completely unchanged. [Point at the identity.] Because it's unbiased for *any* proposal q, the trained estimator still targets the original prior — we're free to spend the simulation budget wherever q puts its mass.

The catch is there's **no guarantee**. A badly chosen proposal can actually *increase* the variance and collapse the effective sample size.

---

## 13 · A test case: AR(1)

We tested this on a first-order autoregressive model, with parameters rho and sigma, using a recurrent network and checking it against the exact Gibbs posterior.

The intuition was that the **extreme edges** — tiny sigma, or rho near plus-or-minus one — *look* hardest to estimate, so those are the natural place to spend extra draws. [Point to the right plot.] So we built a proposal that **oversamples** exactly those edges.

But look at the left panel [point]. The network already tracks the Gibbs reference across the whole prior. And the highlighted case (the star) is telling: it's a tiny-sigma series — essentially a flat, signal-free line — so rho simply **can't be identified** from it. It's not that the estimator fails; the data carry no information. So these edges only *look* hard.

---

## 14 · Importance sampling: the result

And the result is negative. [Left plot.] Every proposal we tried **inflated** the training-gradient variance — the exact opposite of what we wanted, and the very quantity Rao–Blackwell shrinks. [Right plot.] Correspondingly, the test error dropped in **no region** — no proposal beat the plain uniform baseline.

Why does it fail here? Two reasons. The estimator is already sitting near the **Bayes floor**, so there's very little reducible error for *any* proposal to recover. And the regions we flooded only *looked* hard — those near-non-stationary edges are actually the **easiest** places to pin down rho, because the big swings carry the most information.

So the lesson is the contrast: Rao–Blackwell is unconditional, while importance sampling only helps when the proposal genuinely matches where the signal is.

---

## 15 · Conclusion

To wrap up: training a neural Bayes estimator is **itself a Monte Carlo procedure**, so its training variance is something we can attack directly.

**Rao–Blackwellisation** needs a conjugate block, but in return gives a *provably* lower-variance gradient for any model or architecture — and on linear regression it trained to a lower error, most of all where the batch was small.

**Importance sampling** needs nothing special and is far more general, but it's unguaranteed and problem-dependent — and on our AR(1) test it didn't help.

And there's plenty of **future work** — many more variance-reduction techniques left to try.

---

## 16 · Thank you

Thank you.

[The chair will run questions — don't invite them yourself.]

---

*(The "Does it work?" normal-model demo slide is currently disabled in the deck. If you re-enable it, it slots in right after Method (2): "Let's sanity-check on a toy normal model — the neural estimator sits right on top of MCMC, at a fraction of the cost.")*
