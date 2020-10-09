# Value iteration
import gym
import operator as op
from classes import PolicyIterationTrainer, ValueIterationTrainer


seed = 1
# Create the environment
env = gym.make('FrozenLake8x8-v0')
env.reset()

# Managing configurations of experiments
default_vi_config = dict(
    max_iteration=10000,
    evaluate_interval=100,  # don't need to update policy each iteration
    gamma=1.0,
    eps=1e-10
)


def value_iteration(train_config=None):
    config = default_vi_config.copy()
    if train_config is not None:
        config.update(train_config)
    trainer = ValueIterationTrainer(gamma=config['gamma'])

    old_policy_result = {
        obs: 0 for obs in range(trainer.obs_dim)
    }

    for i in range(config['max_iteration']):
        # train the agent
        trainer.train()
        new_policy_result = {
            obs: trainer.policy(obs) for obs in range(trainer.obs_dim)
        }
        should_stop = op.eq(old_policy_result, new_policy_result)

        old_policy_result = new_policy_result

        # evaluate the result
        if i % config['evaluate_interval'] == 0:
            print("[INFO]\tIn {} iteration, current "
                  "mean episode reward is {}.".format(
                i, trainer.evaluate()
            ))
            trainer.update
            if should_stop:
                print("We found policy is not changed anymore at "
                      "itertaion {}. Current mean episode reward "
                      "is {}. Stop training.".format(i, trainer.evaluate()))
                break
            if i > 3000:
                print("You sure your codes is OK? It shouldn't take so many "
                      "({}) iterations to train a policy iteration "
                      "agent.".format(
                    i))

    assert trainer.evaluate() > 0.8, \
        "We expect to get the mean episode reward greater than 0.8. " \
        "But you get: {}. Please check your codes.".format(trainer.evaluate())

    return trainer


vi_agent = value_iteration()
print("Your value iteration agent achieve {} mean episode reward. The optimal score "
      "should be almost {}.".format(vi_agent.evaluate(), 0.86))
vi_agent.print_table()