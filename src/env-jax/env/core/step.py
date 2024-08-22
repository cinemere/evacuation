
def agent_step(action, agent):
    print(f"running `agent_step` with {action=} {agent=}")
        
    new_agent = agent
    terminated_agent = False # depends on wall collision
    reward_agent = 0 # depends on reward params
    
    return agent, terminated_agent, reward_agent

def pedestrians_step(pedestrians, agent, now):
    print(f"running `pedestrians_step` with {pedestrians=} {agent=} {now=}")

    new_pedestrians = pedestrians
    terminated_pedestrians = False
    reward_pedestrians = 0
    intrinsic_reward = 0
    
    return new_pedestrians, terminated_pedestrians, reward_pedestrians, intrinsic_reward
