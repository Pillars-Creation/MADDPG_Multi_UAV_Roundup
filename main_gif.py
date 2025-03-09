import numpy as np
from maddpg import MADDPG
from sim_env import UAVEnv
from buffer import MultiAgentReplayBuffer
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
from PIL import Image

warnings.filterwarnings('ignore')


def obs_list_to_state_vector(obs):
    state = np.hstack([np.ravel(o) for o in obs])
    return state


def save_image(env_render, filename):
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')  # Use 'RGBA' mode since the buffer includes transparency
    image = image.convert('RGB')  # Convert to 'RGB' if you don't need transparency
    image.save(filename)


if __name__ == '__main__':
    env = UAVEnv()
    n_agents = env.num_agents
    actor_dims = []
    for agent_id in env.observation_space.keys():
        actor_dims.append(env.observation_space[agent_id].shape[0])
    critic_dims = sum(actor_dims)

    # action space is a list of arrays, assume each agent has same action space
    n_actions = 2
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           fc1=128, fc2=128,
                           alpha=0.0001, beta=0.003, scenario='UAV_Round_up',
                           chkpt_dir='tmp/maddpg/')

    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims,
                                    n_actions, n_agents, batch_size=256)

    PRINT_INTERVAL = 50
    N_GAMES = 5000
    MAX_STEPS = 100
    total_steps = 0
    score_history = []
    target_score_history = []
    evaluate = False  # 默认为训练模式
    best_score = -30
    eval_counter = 0  # 用于记录训练次数的计数器

    plt.ion()  # 开启交互模式
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = None
    step_text = ax.text(0.02, 1.05, '', transform=ax.transAxes, color='red', alpha=0.8)

    if evaluate:
        maddpg_agents.load_checkpoint()
        print('----  evaluating  ----')
    else:
        print('----training start----')

    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        score_target = 0
        dones = [False] * n_agents
        episode_step = 0
        while not any(dones):
            if evaluate or (eval_counter % PRINT_INTERVAL == 0 and not evaluate):
                env_render = env.render(i)
                if im is None:
                    im = ax.imshow(env_render)
                else:
                    im.set_data(env_render)
                
                # 添加 target 标记
                ax.text(0.95, 0.95, 'target', transform=ax.transAxes, color='red', ha='right', va='top')
                
                step_text.set_text(f'Steps: {episode_step}')
                plt.draw()
                plt.pause(0.01)

            actions = maddpg_agents.choose_action(obs, total_steps, evaluate or (eval_counter % PRINT_INTERVAL == 0 and not evaluate))
            obs_, rewards, dones = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                dones = [True] * n_agents

            memory.store_transition(obs, state, actions, rewards, obs_, state_, dones)

            if total_steps % 10 == 0 and not evaluate:
                maddpg_agents.learn(memory, total_steps)

            obs = obs_
            score += sum(rewards[0:2])
            score_target += rewards[-1]
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        target_score_history.append(score_target)
        avg_score = np.mean(score_history[-100:])
        avg_target_score = np.mean(target_score_history[-100:])
        if not evaluate:
            if i % PRINT_INTERVAL == 0 and i > 0 and avg_score > best_score:
                print('New best score', avg_score, '>', best_score,'saving models...')
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score), '; average target score {:.1f}'.format(avg_target_score))

        eval_counter += 1  # 每次训练结束后计数器加 1

    # save data
    file_name ='score_history.csv'
    if not os.path.exists(file_name):
        pd.DataFrame([score_history]).to_csv(file_name, header=False, index=False)
    else:
        with open(file_name, 'a') as f:
            pd.DataFrame([score_history]).to_csv(f, header=False, index=False)

    if evaluate or (eval_counter % PRINT_INTERVAL == 0 and not evaluate):
        plt.ioff()  # 关闭交互模式
        plt.close()
