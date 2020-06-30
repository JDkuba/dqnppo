def play(agent, no_games, env, batch_size, render=False, rolling_average=10):
    mean_rewards = []
    last_ten_rewards = 0
    time = 0

    for i in range(1, no_games + 1):
        reward_agg = 0

        done = False
        state = env.reset()
        while not done:
            time += 1
            if render:
                env.render()
            action, meta_info = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_experience(state, action, reward, next_state, done, meta_info)
            agent.learn(batch_size, time)

            reward_agg += reward
            state = next_state

        last_ten_rewards += reward_agg
        if i % rolling_average == 0:
            mean_rewards.append((i, last_ten_rewards / rolling_average))
            last_ten_rewards = 0
            print(f"Episode {i} reward: {mean_rewards[-1]}")

    return mean_rewards
