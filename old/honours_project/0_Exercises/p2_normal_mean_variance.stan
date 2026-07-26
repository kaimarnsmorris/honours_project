data {
  int<lower=0> N;
  vector[N] x;
  real mu_0;
  real<lower=0> sigma_0;
  real<lower=0> A;
}
parameters {
  real mu;
  real<lower=0, upper=A> sigma;
}
model {
  // Prior
  sigma ~ uniform(0, A);
  mu ~ normal(mu_0, sigma_0);
  // Likelihood
  x ~ normal(mu, sigma);
}
