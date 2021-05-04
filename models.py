"""
Description:
    This code is basically adapted from FinRL (https://github.com/AI4Finance-LLC/FinRL).
    In order to use MLP+LSTM policy in othis project, I switched from stable_baseline3 to
    stable_baseline. Also did some modification to fit our environemnt.
"""
import pandas as pd
import numpy as np
import time
import gym
import pdb

import config

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3
from stable_baselines import SAC
from stable_baselines.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Used to access our parameters from the config file
# MLP policy map is ours, kwarg unpacking and noise is FinRL
MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO2}
POLICIES = {"MlpPolicy": MlpPolicy, "MlpLstmPolicy": MlpLstmPolicy}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}
POLICY_KWARGS = {x: config.__dict__[f"{x}_PARAMS"] for x in POLICIES.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

class DRLAgent:
    def __init__(self, env):
        self.env = env

    # Some from https://github.com/AI4Finance-LLC/FinRL (FinRL) but
    # modified to suit our needs
    @staticmethod
    def DRL_prediction(model, environment):
        
        test_env, test_obs = environment.get_sb_env()

        # Run through testing once, return the assets history overtime from the method_name
        for i in range(environment.max_t):
            action, _states = model.predict(test_obs)
            test_obs, reward, done, _ = test_env.step(action)

            if done: 
                assets_history = test_env.env_method(method_name="save_assets_history")
                break

        return assets_history

    # Mostly from https://github.com/AI4Finance-LLC/FinRL (FinRL)
    # Just extracts policy and model arguments, initializes the arguments,
    # adds in action noise for stable baselines, and returns
    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
    ):
        self.model_name = model_name
        self.policy = policy

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS[policy]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
        )

        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        return model

    # Train and load model (save trained to directory, load from directory)
    def train_model(self, model, save_path=None, total_timesteps=5000):
        model = model.learn(total_timesteps=total_timesteps)
        if save_path != None:
            model.save(save_path)
        return model

    def load_model(self, model_name, model_path):
        model = MODELS[model_name].load(model_path)
        return model
