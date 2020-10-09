# Policy Iteration
import gym
import operator as op
from classes import PolicyIterationTrainer

seed = 1
# Create the environment
env = gym.make('FrozenLake8x8-v0')
env.reset()

default_pi_config = dict(
    max_iteration=1000,
    evaluate_interval=1,
    gamma=1.0,
    eps=1e-10
)


def policy_iteration(train_config=None):
    config = default_pi_config.copy()
    if train_config is not None:
        config.update(train_config)

    trainer = PolicyIterationTrainer(gamma=config['gamma'], eps=config['eps'])

    old_policy_result = {
        obs: 0 for obs in range(trainer.obs_dim)
    }
    should_stop = False
    for i in range(config['max_iteration']):
        # train the agent
        trainer.train()
        new_policy_result = {
            obs: trainer.policy(obs) for obs in range(trainer.obs_dim)
        }
        #  compare the new policy with old policy
        should_stop = op.eq(old_policy_result, new_policy_result)

        if should_stop:
            print("We found policy is not changed anymore at "
                  "itertaion {}. Current mean episode reward "
                  "is {}. Stop training.".format(i, trainer.evaluate()))
            break
        old_policy_result = new_policy_result

        # evaluate the result
        if i % config['evaluate_interval'] == 0:
            print(
                "[INFO]\tIn {} iteration, current mean episode reward is {}."
                "".format(i, trainer.evaluate()))

            if i > 20:
                print("You sure your codes is OK? It shouldn't take so many "
                      "({}) iterations to train a policy iteration "
                      "agent.".format(i))

    assert trainer.evaluate() > 0.8, \
        "We expect to get the mean episode reward greater than 0.8. " \
        "But you get: {}. Please check your codes.".format(trainer.evaluate())

    return trainer


pi_agent = policy_iteration()
print("Your policy iteration agent achieve {} mean episode reward. The optimal score "
      "should be almost {}.".format(pi_agent.evaluate(), 0.86))
pi_agent.render()
pi_agent.print_table()
