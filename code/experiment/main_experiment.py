########################################################################

print('loading packages...')
import numpy as np
np.set_printoptions(precision = 4, suppress = True)
import random
import pandas as pd

########################################################################

print('writing functions...')
def initialize_qtable(N, num_actions, feedback):
    if feedback == 1:
        Q = np.random.uniform(0,1,(N, num_actions,num_actions))
    else:
        Q = np.random.uniform(0,1,(N, num_actions))   
        #Q = np.zeros((N, num_actions))   
    return Q

def exploratory_strategy(Q, N, egreedy, num_actions,past_win,winning_bid,eps,beta):
    # Input: Master Q table, agent index, past winning action
    # Output: actions taken by all agents
    actions = np.ones(N,dtype=int)
    for agent_idx in range(N):
        if egreedy == 1:
            if past_win == 1:
                if np.random.uniform()>eps:
                    actions[agent_idx] = np.argmax(Q[agent_idx,winning_bid])
                else:
                    actions[agent_idx] = np.random.choice(range(num_actions))
            else:
                if np.random.uniform()>eps:
                    actions[agent_idx] = np.argmax(Q[agent_idx])
                else:
                    actions[agent_idx] = np.random.choice(range(num_actions))
        else: 
            if past_win == 1:
                max_Q = np.max(Q[agent_idx,winning_bid])
                logits = (Q[agent_idx,winning_bid] - max_Q) / beta
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits)
                actions[agent_idx] = np.random.choice(range(len(probs)), p=probs)
            else:
                max_Q = np.max(Q[agent_idx])
                logits = (Q[agent_idx] - max_Q) / beta
                exp_logits = np.exp(logits)
                probs = exp_logits / np.sum(exp_logits)
                actions[agent_idx] = np.random.choice(range(len(probs)), p=probs)    
    return actions


def actions2rewards(actions, valuations, design, action2bid, method=0):
    # compute winners and winning payments
    bids = action2bid[actions]
    unique_values_sorted = np.sort(np.unique(bids))[::-1]
    first_highest_value = unique_values_sorted[0]
    if len(unique_values_sorted) > 1:
        second_highest_value = unique_values_sorted[1]
    else:
        second_highest_value = first_highest_value
    winners_payment = design * first_highest_value + (1-design) * second_highest_value
    winners_idx = np.where(bids == first_highest_value)[0]
    # compute reward vector
    if method==0:
        winner_idx = np.random.choice(winners_idx)
        rewards = bids * 0.0
        rewards[winner_idx] = valuations[winner_idx] - winners_payment
    else:
        winning_bid_idx = np.where(bids==first_highest_value,1,0)
        no_of_winners = np.sum(winning_bid_idx)
        rewards = (1/no_of_winners) * (valuations - winners_payment) * winning_bid_idx 
    return rewards


def counterfactual_reward(agent_idx,actions,valuations,design,num_actions,action2bid):
    action_copy = actions.copy()
    cf_rewards = np.zeros(num_actions)
    for agent_action in range(num_actions):
        action_copy[agent_idx] = agent_action
        cf_rewards[agent_action] = actions2rewards(action_copy,valuations,design,action2bid,method=1)[agent_idx]
    return np.array(cf_rewards)


def update_qvalues(rewards, actions, Q, feedback, asynchronous, winning_bid, alpha, gamma, valuations,design,action2bid):
    N = Q.shape[0]
    num_actions = Q.shape[1]
    for agent_idx in range(N):
        action = actions[agent_idx]
        reward = rewards[agent_idx]
        if asynchronous == 1:
            if feedback == 1:
                Q[agent_idx,winning_bid,action] =  (1-alpha)*Q[agent_idx,winning_bid,action] + alpha * (reward + gamma * np.max(Q[agent_idx,winning_bid]))
            else:
                Q[agent_idx,action] =  (1-alpha)*Q[agent_idx,action] + alpha * (reward + gamma * np.max(Q[agent_idx]))
        else: 
            reward_vec = counterfactual_reward(agent_idx,actions,valuations,design,num_actions,action2bid)
            if feedback == 1:
                Q[agent_idx,winning_bid,:] =  (1-alpha)*Q[agent_idx,winning_bid,:] + alpha * (reward_vec + gamma * np.max(Q[agent_idx,winning_bid]))
            else:
                Q[agent_idx] =  (1-alpha)*Q[agent_idx] + alpha * (reward_vec + gamma * np.max(Q[agent_idx]))
    return Q


def experiment(N,alpha,gamma,egreedy,asynchronous,design,feedback,num_actions,decay,verbose=2):
    Q = initialize_qtable(N, num_actions,feedback)
    common_valuation = 1
    valuations = np.ones(N)*common_valuation
    action2bid = np.linspace(0,common_valuation,num_actions)
    initial_eps = 1.0
    eps_decay = decay
    min_eps = 0.01
    eps = initial_eps
    initial_beta = 1.0
    beta_decay = decay
    min_beta = 0.01
    beta = initial_beta
    winning_bid_history = []
    num_episodes = 250000
    winning_action = 0
    action_history = []
    
    print('Bidders:',N)
    print('Learning rate:',alpha)
    print('Discount factor:',gamma)
    print('eGreedy',egreedy)
    print('Asynchronous',asynchronous)
    print('First price:',design)
    print('Feedback:',feedback)
    print('Bid space:',action2bid)
    print('Q-table shape:',Q.shape)
    print('Epsilon: initial, final and decay rate', initial_eps, min_eps, eps_decay)
    print('Beta: initial, final and decay rate', initial_beta, min_beta, beta_decay)
    print('Num episodes:', num_episodes)

    # training loop
    for episode in range(num_episodes):
        
        # act from exploratory strategy
        actions = exploratory_strategy(Q, N, egreedy, num_actions,feedback,winning_action,eps,beta) 

        # collect rewards
        rewards = actions2rewards(actions,valuations,design,action2bid)
        
        # update Q-values
        Q = update_qvalues(rewards, actions, Q, feedback, asynchronous, winning_action, alpha, gamma, valuations,design,action2bid)

        # update exploration params
        eps = max(min_eps, eps * eps_decay)
        beta = max(min_beta, beta * beta_decay)

        # winning bid and action
        bids = actions*common_valuation/(num_actions-1)
        winning_bid = np.max(bids)
        winning_action = np.max(actions)

        # store history
        winning_bid_history.append(winning_bid)
        action_history.append(action2bid[actions])
        
        if (verbose==1) & (episode%10000==0) & (episode>0):
            print(episode,np.round(eps,2),np.round(beta,2),np.round(np.mean(winning_bid_history[-1000:]),2),np.round(np.std(winning_bid_history[-1000:]),2))
                
        if (verbose==2) & (episode%10000==0) & (episode>0):
            #action_history.append(actions)
            print('\nEpisode:',episode,np.round(eps,3))
            print('Bids:',action2bid[actions],', Rewards:',rewards)
            #print(winning_bid_history[-20:])
            print('Agent 0:',Q[0,:])
            #print('Agent 0 Cf reward', counterfactual_reward(0,actions,valuations,design,num_actions,action2bid))
            print('Agent 1:',Q[1,:])
            #print('Agent 1 Cf reward', counterfactual_reward(1,actions,valuations,design,num_actions,action2bid))
            print('Stdev of last 10000 bids:',np.std(winning_bid_history[-10000:]))
            print('Avg of last 10000 bids:',np.mean(winning_bid_history[-10000:]))
            
        if (episode%10000==0) & (eps==min_eps or beta==min_beta) & (np.std(winning_bid_history[-1000:])<0.001):
            break

    Y = np.mean(winning_bid_history[-1000:])
    Y_std = np.std(winning_bid_history[-1000:])
    Y_min = np.min(winning_bid_history[-1000:])
    Y_max = np.max(winning_bid_history[-1000:])
    
    if verbose==2:
        return (Y,episode,Y_std,Y_min,Y_max,N,alpha,gamma,egreedy,asynchronous,design,feedback,num_actions,decay,action_history,winning_bid_history)
    else:
        return (Y,episode,Y_std,Y_min,Y_max,N,alpha,gamma,egreedy,asynchronous,design,feedback,num_actions,decay,Q)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

########################################################################

print('running test case...')
params = {'N': 4, 'alpha': 0.1, 'gamma': 0.99, 'egreedy': 0, 'design': 0, 
          'asynchronous': 1, 'feedback': 1, 'num_actions': 6, 'decay':0.9999, 'verbose':2}
(Y,episode,Y_std,Y_min,Y_max,N,alpha,gamma,egreedy,asynchronous,design,feedback,num_actions,decay,action_history,winning_bid_history) = experiment(**params)

########################################################################

print('running main experiment...')
parameters = {'N':[2,4,6,8,10],
              'alpha':[0.01,0.05,0.1,0.5],
              'gamma':[0.0,0.25, 0.5, 0.75, 0.99],
              'egreedy':[0,1],
              'design':[0.0, 0.25, 0.5, 0.75,1.0],
              'asynchronous':[0,1],
              'feedback':[0,1],
              'num_actions':[6,21,31,51],
              'decay':[0.999,0.9995,0.9999,0.99995,0.99999],
              'verbose':[1]
             }

K = 1 # Number of times to sample
data = []
for _ in range(K):
    sampled_params = {}
    for key, value in parameters.items():
        if isinstance(value, list):
            sampled_params[key] = random.choice(value)
        else:
            sampled_params[key] = value
    
    print(sampled_params)
    (Y,episode,Y_std,Y_min,Y_max,N,alpha,gamma,egreedy,asynchronous,design,feedback,num_actions,decay,Q) = experiment(**sampled_params)
    data.append((Y,episode,Y_std,Y_min,Y_max,N,alpha,gamma,egreedy,asynchronous,design,feedback,num_actions,decay))
    print((_,Y,episode,Y_std))
    print("-" * 20)

########################################################################

print('storing as pickle...')
df = pd.DataFrame(data, columns = ['bid2val','episode','Y_std','Y_min','Y_max','N','alpha','gamma','egreedy','asynchronous','design','feedback','num_actions','decay'])
df.to_pickle('data_experiment_main.pkl')


