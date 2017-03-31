
data {
  int<lower=0> N; // number of data points in dataset
  int<lower=1> P; // number of known covariates
  int<lower=1> G; // number of observed genes

  vector[P] x[N]; // Covariates
  real y[N];      // Expression values
  int<lower=1, upper=G> gene[N]; // Gene identifiers

}
parameters {
  // variance parameter
  real<lower=0> sigma;

  // Partial regression weights
  matrix[G, P] w;
  vector<lower=0>[P] alpha;
}
model {
  // priors
  for (g in 1:G){
    w[g] ~ normal(0, sigma * alpha);
  }
  sigma ~ lognormal(0, 1);
  alpha ~ inv_gamma(1, 1);

  // likelihood
  for (n in 1:N){
    x[n] ~ normal(w[gene[n]] * x[n], sigma);
  }
}
