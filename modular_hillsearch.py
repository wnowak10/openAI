import gym
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, parameters):
    observation = env.reset()
    totalreward = 0
    counter = 0
    for _ in xrange(200):
        # env.render()
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        counter += 1
        if done:
            break
    return totalreward

def train(submit):
    env = gym.make('CartPole-v0')
    if submit:
        env.monitor.start('hillsearch', force=True)

    episodes_per_update = 5
    # noise_scaling = 0.1
    parameters = np.mod(np.random.rand(4),1)
    bestreward = 0
    counter = 0

    for _ in xrange(200):
        counter += 1
        if counter<3:
            noise_scaling = .9
        elif counter >2 and counter < 5: # as counter goes on, slightly decrease step size? does this make any difference? 
            noise_scaling = .8
        else:
            noise_scaling = .7
        # newparams = np.mod((parameters + (np.random.rand(4)*2-1)*noise_scaling),2) # with mod
        # newparams = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling # no mod
        newparams = np.mod((parameters + (np.random.rand(4))*noise_scaling),1) # with mod

        # print newparams
        # reward = 0
        # for _ in xrange(episodes_per_update):
        #     run = run_episode(env,newparams)
        #     reward += run
        reward = run_episode(env,newparams)
        # print "reward %d best %d" % (reward, bestreward)
        if reward > bestreward:
            # print "update"
            bestreward = reward
            parameters = newparams
            if reward == 200:
                break

    if submit:
        for _ in xrange(100):
            run_episode(env,parameters)
        env.monitor.close()
    return counter


train(True)

# results = []
# for _ in xrange(1000):
#     results.append(train(submit=False))

# print np.median(results)
# print np.mean(results)


# plt.hist(results,50,normed=1, facecolor='g', alpha=0.75)
# plt.xlabel('Episodes required to reach 200')
# plt.ylabel('Frequency')
# plt.title('Histogram of Random Search')
# plt.show()