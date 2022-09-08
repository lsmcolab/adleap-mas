###
# Imports
###
import sys
import pandas as pd
import os
sys.path.append(os.getcwd())

from src.envs.TradeStockEnv import TradeStockEnv, TSAgent
from src.utils import tradestocktools as tst

###
# Setting the environment
###
tickers = ['GOOG', 'GOOGL']
start = "2017-01-01"
end = "2022-01-01"

prices = tst.get_prices(tickers,start,end)
model, beta = tst.get_model(prices, lag_order = 10, coint_rank = 1)
spread = tst.get_spread(prices, beta)

split_size = int(len(prices) * 0.8)
train, test = spread[:split_size], spread[split_size:]
priceA, priceB = prices[prices.columns[0]], prices[prices.columns[1]]
mean = tst.cross_check(test, level = 0)

cols = ['step','stock A', 'stock B', 'numA','numB','Balance' ,'Returns']
results = [[0,0,0,0,0,0,0]]
results_df = pd.DataFrame(results, columns = cols, dtype = float)

agent = TSAgent(index=0,atype='pomcp')
components = {\
    'agent':agent,
    'train':train,
    'test':test,
    'priceA': priceA,
    'priceB':priceB,
    'mean':mean,
    'beta':beta
}

env = TradeStockEnv(components, display=True)
state = env.reset()

###
# ADLEAP-MAS MAIN ROUTINE
### 
state = env.reset()
agent = env.get_adhoc_agent()

done, max_episode = False, 200
while env.episode < max_episode and not done:
    # 1. Importing agent method
    method = env.import_method(agent.type)

    # 2. Reasoning about next action and target
    agent.next_action, _ = method(state, agent)

    # 3. Taking a step in the environment
    R = ((env.state['stockA']* env.state['numstockA'])+
           (env.state['stockB']* env.state['numstockB']))+env.state['balance']
    #print('> From:',env.state,'| R:',R)
    #print('step',env.episode,'action',env.action_dict[agent.next_action])
    
    temp = [[env.episode,env.state['stockA'],env.state['stockB'],
             env.state['numstockA'],env.state['numstockB'],
             env.state['balance'],R]]
    print(temp)
    test = pd.DataFrame(temp, columns = cols, dtype = float)
    results_df = pd.concat([results_df,test], ignore_index = True, axis = 0)

    next_state, reward, done, _ = env.step(action=agent.next_action)
    state = next_state

    R = ((env.state['stockA']* env.state['numstockA'])+
           (env.state['stockB']* env.state['numstockB']))+env.state['balance']
    #print('> To:',env.state,'| R:',R)
    print('Episode:',env.episode,'| Return:',R)
env.close()
###
# THE END - That's all folks :)
###