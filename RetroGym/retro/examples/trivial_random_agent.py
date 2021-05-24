import retro

def main():
    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step()
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
