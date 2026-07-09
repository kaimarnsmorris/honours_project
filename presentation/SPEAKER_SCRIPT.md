# Speaker Script — Variance Reduction for Training Neural Bayes Estimators

**Format:** talk only (~13:30 target), Q&A separate, hard cap **15:00**.
Talking points are prompts, not a word-for-word read — say them in your own voice.

## Timing budget

| Frame | Slide | Target (cumulative) |
|-------|-------------------------------|--------------|
| 1  | Title                          | 0:20 |
| 2  | The problem, in one breath     | 1:00 |
| 3  | Expensive — and repeated       | 2:00 |
| 4  | NBEs: pay once                 | 3:30 |
| 5  | The pivot                      | 5:00 |
| 6  | A trick for killing noise      | 5:50 |
| 7  | The RB loss                    | 7:10 |
| 8  | **The guarantee (peak)**       | 8:20 |
| 9  | Does it pay off?               | 10:20 |
| 10 | IS contrast **[CUTTABLE]**     | 11:50 |
| 11 | What separates them **[CUTTABLE]** | 12:50 |
| 12 | Takeaway                       | 13:40 |
| 13 | Thanks / Q&A                   | 13:50 |

**Checkpoints:** you should be starting **the guarantee (frame 8) at ~7:10** and **finishing frame 9 by ~10:20**. If you reach frame 9 later than ~11:00, take the **CUT POINT** (skip 10–11). Without frames 10–11 the Takeaway lands at **~11:10** — a comfortable margin.

## Delivery reminders

- Speak *to the figures*, not to the bullet text. The bullets are anchors for the room; the words are yours.
- The emotional peak is **frame 8 (the guarantee)** — slow down, pause after "never worse, usually better."
- Frame 9: physically point at orange-below-blue and at the green floor before you give the three numbers.
- Know your one-line answer to "what's a neural Bayes estimator" and "what's Rao–Blackwellisation" cold — those are the likely first questions.

---

## Frame 1 — Title (→ 0:20)
- Good [morning/afternoon]. My thesis is on **variance reduction for training neural Bayes estimators** — making a modern, neural approach to Bayesian inference cheaper to train.
- I'll keep the machinery light and focus on one clean idea and what it buys us.

## Frame 2 — The problem, in one breath (→ 1:00)
- Here's the whole talk in one breath.
- Normally, Bayesian inference means running a sampler — MCMC — from scratch for **every** new dataset. Expensive, and you pay it again every time.
- Neural Bayes estimators avoid that: train a network **once**, and then inference for a new dataset is a single forward pass.
- My starting point: training that network is *itself* a Monte Carlo computation — so the classical tricks for reducing Monte Carlo noise should apply to the training.
- **[full version]** I look at two of them: one comes with a guarantee, the other is more general but doesn't.
  **[short version, if IS is cut]** I focus on one that gives a *provable* guarantee — that's today's story.

## Frame 3 — Bayesian inference is expensive, and repeated (→ 2:00)
- Let me set that up. In Bayesian inference, for each dataset $y$ we want the posterior over the parameters.
- The workhorse is MCMC. It's accurate, but it has **no memory** — a new dataset means starting the sampler again from scratch.
- So with many datasets — which is common — you pay the same expensive computation over and over. That repetition is the thing we want to kill.

## Frame 4 — Neural Bayes estimators: pay once (→ 3:30)
- The neural Bayes estimator idea is to pay that cost **once**.
- Under squared-error loss, the best possible estimator is just the **posterior mean** — and that's a fixed function of the data. It doesn't change from dataset to dataset.
- So we approximate that function with a neural network. *(point at the box)* Data comes in on the left, the parameter estimate comes out on the right.
- We train it **once**, on simulated data where we know the true parameters: feed the network simulated datasets and nudge its weights so its output matches the parameter that generated the data. After that, any new dataset is a single forward pass. That box is the only place we spend the expensive compute.

## Frame 5 — The observation this thesis starts from (→ 5:00)
- Here's the observation the whole thesis turns on.
- Training the network means minimising the **Bayes risk** — the expected squared error between the true parameter and the network's estimate, averaged over the model. *(gesture at the equation)*
- In practice that expectation is a **Monte Carlo average** over simulated draws. So the loss — and the gradient we actually descend — is a **random quantity**. It has variance.
- And that variance comes from *how we sample the loss*, not from the inference problem. Which means we can reduce it without changing the problem at all. That's the opening.

## Frame 6 — A trick for killing Monte Carlo noise (→ 5:50)
- So how do you reduce Monte Carlo noise? Here's the key idea — no special background needed.
- Suppose you're estimating an average by sampling, but part of it you can work out **exactly**. Then don't sample that part — substitute its exact value, its conditional mean.
- Averaging out a random draw can only **remove** variance; it can never add any. Statisticians call this **Rao–Blackwellisation**.

## Frame 7 — The Rao–Blackwellised loss (→ 7:10)
- We apply exactly that to the training loss. Split the parameters into two blocks — call them $\tau$ and $\beta$.
- Suppose $\beta$ has a tractable conditional posterior given the rest — here it's conjugate, so we have it in closed form.
- Then instead of *sampling* $\beta$ as the target, we substitute its **conditional posterior mean**. We've integrated $\beta$ out of the loss analytically instead of by sampling.
- The leftover variance becomes a constant — it doesn't even affect the gradient.

## Frame 8 — The guarantee (peak) (→ 8:20)
- And now the payoff — this is the heart of it. *(slow down)*
- That new gradient is a **conditional average** of the original one. The law of total variance then says its variance can only go **down** — the gap is positive semi-definite.
- So this can **never** increase the gradient variance. Not for one model — for *any* model, *any* architecture. And for a linear network the same reduction carries all the way through to the fitted weights themselves.
- In other words: **for free, provably, never worse and usually better.** *(pause)*

## Frame 9 — Does it pay off? (→ 10:20)
- Does that translate into a real gain? I tested it on **Bayesian linear regression**, integrating out the coefficients. A Gibbs sampler gives the best achievable loss — the **Bayes floor**, in green.
- *(point)* Blue is the standard estimator, orange is the Rao–Blackwellised one. Orange sits **below** blue everywhere — and its band is **tighter**, which is the variance reduction showing up directly.
- Three numbers: **lower loss in every cell** of the grid; at the baseline it **closes about half — 52% — of the gap to the floor**; and it **matches the standard estimator at a quarter of the batch size**. So you can train smaller for the same accuracy.

> **>>> CUT POINT — if you reached this slide later than ~11:00, skip frames 10–11.**
> Bridge line into the Takeaway: *"Rao–Blackwellisation isn't the only lever — there's a more general one, importance sampling, which I'm happy to get into in questions. But the guarantee is the heart of the story, so let me close."*

## Frame 10 — A contrasting lever: importance sampling **[CUTTABLE]** (→ 11:50)
- Briefly, the contrast — because it sharpens what makes Rao–Blackwell special.
- Importance sampling is a second lever. It needs **no conjugacy**: you reshape the distribution you simulate from and reweight back to the same objective. Much more general. The price: **no guarantee**.
- On an autoregressive model with a recurrent network, it simply didn't help. *(left)* The estimator already sits right on the Gibbs reference across the whole prior — almost no error left to recover. *(right)* And push the proposal hard, the effective sample size collapses and you just add noise.

## Frame 11 — What separates the two techniques **[CUTTABLE]** (→ 12:50)
- So here's the clean comparison.
- Rao–Blackwell needs a conjugate block, but where you have one it gives an **unconditional guarantee**, and it carries through to the fitted weights.
- Importance sampling needs nothing special, but has **no guarantee** and doesn't carry to the weights.
- Where conjugacy exists, Rao–Blackwell is the safer *and* the stronger lever.

## Frame 12 — Takeaway (→ 13:40)
- To wrap up. Because training a neural Bayes estimator is a Monte Carlo computation, the whole Monte Carlo variance-reduction toolbox applies to **training**.
- Rao–Blackwellisation in particular is a cheap, reliable lever: a **provable** cut in gradient variance, and real movement toward the Bayes floor.
- And the punchline: **conjugacy — the classical trick that made Gibbs sampling possible — turns out to be just as useful for training the neural estimators meant to replace it.**

## Frame 13 — Thanks / Q&A (→ 13:50)
- Thank you — I'm happy to take questions.
- *(Backup slides available: the variance proof, the full grid, convergence-vs-samples, the importance-sampling detail, and the model/architecture specifics.)*
