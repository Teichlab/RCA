data {
    int<lower=1> K;
    int<lower=1> N;
    int<lower=1> G;
    int y[N, G];
}
parameters {
    simplex[K] theta;
    real<lower=0> lambda[K, G];
}
model {
    real ps[K];
    real per_gene_ps[G];
    for (n in 1:N) {
        for (k in 1:K) {
            for (g in 1:G) {
                per_gene_ps[g] = poisson_lpmf(y[n, g] | lambda[k, g]);
            }
            ps[k] = log(theta[k]) + log_sum_exp(per_gene_ps);
        }
        target += log_sum_exp(ps);
    }
}
generated quantities {
    vector[N] log_p_y;
    real per_gene_ps[G];
    for (n in 1:N){
        for (g in 1:G) {
            per_gene_ps[g] = poisson_lpmf(y[n] | lambda[1, g]);
        }
        log_p_y[n] = log_sum_exp(per_gene_ps);
    }
}
