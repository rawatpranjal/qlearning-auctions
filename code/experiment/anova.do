generate decay_ = decay>0.99925
generate alpha_ = alpha>0.05
generate gamma_ = gamma>0.05
generate num_actions_ = num_actions > 8
pwcorr n alpha_ gamma_ egreedy asynchronous design feedback num_actions decay_, star(0.05)
anova bid2val n alpha_ gamma_ egreedy asynchronous design feedback num_actions decay_
anova bid2val n##gamma_##asynchronous##design
margins design##n##gamma_##asynchronous
anovaplot bid2val n##gamma_##asynchronous##design, scatter(ms(i))
