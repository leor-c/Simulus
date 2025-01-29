from functools import partial 
from pathlib import Path

import hydra
import loguru
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import gym

from agent import AgentLS
from envs import SingleProcessEnv, POPWMEnv4Play
from game import AgentEnv, EpisodeReplayEnv, Game, AgentTokenEnv
from models.actor_critic import ActorCriticLS
from models.world_model import POPRetNetWorldModel


def play_image_envs(cfg: DictConfig):
    device = torch.device(cfg.common.device)
    assert cfg.mode in ('episode_replay', 'agent_in_env', 'agent_in_world_model', 'play_in_world_model')

    env_fn = partial(instantiate, config=cfg.env.test)
    test_env = SingleProcessEnv(env_fn)

    if cfg.mode.startswith('agent_in_'):
        h, w, _ = test_env.env.unwrapped.observation_space.shape
    else:
        h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]

    if cfg.mode == 'episode_replay':
        env = EpisodeReplayEnv(replay_keymap_name=cfg.env.keymap, episode_dir=Path('media/episodes'))
        keymap = 'episode_replay'

    else:
        tokenizer = instantiate(cfg.tokenizer)
        world_model = POPRetNetWorldModel(obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions,
                                          config=instantiate(cfg.world_model.retnet), **cfg.world_model)
        actor_critic = ActorCriticLS(**cfg.actor_critic, act_vocab_size=test_env.num_actions,
                                     token_embed_dim=cfg.tokenizer.embed_dim,
                                     tokens_per_obs=cfg.world_model.retnet.tokens_per_block - 1,
                                     context_len=cfg.world_model.context_length)

        agent = AgentLS(tokenizer, world_model, actor_critic).to(device)
        agent.load(Path('checkpoints/last.pt'), device)

        if cfg.mode == 'play_in_world_model':
            env = POPWMEnv4Play(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device,
                                env=env_fn())
            keymap = cfg.env.keymap

        elif cfg.mode == 'agent_in_env':
            env = AgentEnv(agent, test_env, cfg.env.keymap, do_reconstruction=cfg.reconstruction)
            keymap = 'empty'
            if cfg.reconstruction:
                size[1] *= 3

        elif cfg.mode == 'agent_in_world_model':
            wm_env = POPWMEnv4Play(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device,
                                   env=env_fn())
            env = AgentEnv(agent, wm_env, cfg.env.keymap, do_reconstruction=False)
            keymap = 'empty'

    game = Game(env, keymap_name=keymap, size=size, fps=cfg.fps, verbose=bool(cfg.header),
                record_mode=bool(cfg.save_mode))
    game.run()


def play_token_envs(cfg: DictConfig):
    device = torch.device(cfg.common.device)

    # enable image rendering:

    env_fn = partial(instantiate, config=cfg.env.test)
    test_env = SingleProcessEnv(env_fn)

    h, w = test_env.env.pixel_obs_shape
    loguru.logger.info(f"Obs shape: ({h}, {w})")
    # if cfg.mode.startswith('agent_in_'):
    #     h, w, _ = test_env.env.unwrapped.observation_space.shape
    # else:
    #     h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]

    if cfg.mode == 'episode_replay':
        env = EpisodeReplayEnv(replay_keymap_name=cfg.env.keymap, episode_dir=Path('media/episodes'))
        keymap = 'episode_replay'

    else:
        tokenizer = None
        if isinstance(test_env.env.observation_space, gym.spaces.MultiDiscrete):
            obs_vocab_size = test_env.env.observation_space.nvec[0]
        elif isinstance(test_env.env.observation_space, gym.spaces.Box):
            obs_vocab_size = test_env.env.observation_space.high.max() + 1
        else:
            assert False, f"unsupported obs space type '{test_env.env.observation_space}'"

        world_model = POPRetNetWorldModel(obs_vocab_size=obs_vocab_size, act_vocab_size=test_env.num_actions,
                                          config=instantiate(cfg.world_model.retnet), **cfg.world_model)
        actor_critic = ActorCriticLS(**cfg.actor_critic, act_vocab_size=test_env.num_actions,
                                     token_embed_dim=cfg.tokenizer.embed_dim,
                                     tokens_per_obs=cfg.world_model.retnet.tokens_per_block - 1,
                                     context_len=cfg.world_model.context_length)

        agent = AgentLS(tokenizer, world_model, actor_critic).to(device)
        agent.load(Path('checkpoints/last.pt'), device, load_tokenizer=False)

        if cfg.mode == 'play_in_world_model':
            env = POPWMEnv4Play(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device,
                                env=env_fn())
            keymap = cfg.env.keymap

        elif cfg.mode == 'agent_in_env':
            env = AgentTokenEnv(agent, test_env, cfg.env.keymap, do_reconstruction=cfg.reconstruction)
            keymap = 'empty'
            if cfg.reconstruction:
                size[1] *= 3

        elif cfg.mode == 'agent_in_world_model':
            wm_env = POPWMEnv4Play(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device,
                                   env=env_fn())
            env = AgentTokenEnv(agent, wm_env, cfg.env.keymap, do_reconstruction=False)
            keymap = 'empty'

    game = Game(env, keymap_name=keymap, size=size, fps=cfg.fps, verbose=bool(cfg.header),
                record_mode=bool(cfg.save_mode))
    game.run()


from main import config_name
@hydra.main(config_path="../config", config_name=config_name, version_base=None)
def main(cfg: DictConfig):
    device = torch.device(cfg.common.device)
    assert cfg.mode in ('episode_replay', 'agent_in_env', 'agent_in_world_model', 'play_in_world_model')

    if cfg.env.obs_modality == 'image':
        play_image_envs(cfg)
    elif cfg.env.obs_modality == 'token':
        play_token_envs(cfg)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
