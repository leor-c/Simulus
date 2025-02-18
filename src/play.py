from functools import partial
from pathlib import Path

import click
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import numpy as np
from scipy.ndimage import rotate

from main import build_agent
from envs import SingleProcessEnv, POPWMEnv4Play
from game import AgentEnv, EpisodeReplayEnv, Game
from utils.preprocessing import get_obs_processor


def play_atari(cfg: DictConfig, mode, reconstruction_mode, header_info, fps, model_path: Path):
    save_mode = 0

    device = torch.device(cfg.common.device)
    assert mode in ('episode_replay', 'agent_in_env', 'agent_in_world_model', 'play_in_world_model')

    env_fn = partial(instantiate, config=cfg.env.test)
    test_env = SingleProcessEnv(env_fn)

    if mode.startswith('agent_in_'):
        h, w, _ = test_env.env.unwrapped.observation_space.shape
    else:
        h, w = 64, 64
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]

    if mode == 'episode_replay':
        env = EpisodeReplayEnv(replay_keymap_name=cfg.env.keymap, episode_dir=Path('media/episodes'))
        keymap = 'episode_replay'

    else:
        agent = build_agent(test_env, cfg, device)
        if model_path is not None:
            agent.load(model_path, device)

        if mode == 'play_in_world_model':
            env = POPWMEnv4Play(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device,
                                env=env_fn())
            keymap = cfg.env.keymap

        elif mode == 'agent_in_env':
            env = AgentEnv(agent, test_env, cfg.env.keymap, do_reconstruction=reconstruction_mode)
            keymap = 'empty'
            if reconstruction_mode:
                size[1] *= 3

        elif mode == 'agent_in_world_model':
            wm_env = POPWMEnv4Play(tokenizer=agent.tokenizer, world_model=agent.world_model, device=device,
                                   env=env_fn())
            env = AgentEnv(agent, wm_env, cfg.env.keymap, do_reconstruction=False)
            keymap = 'empty'

    game = Game(env, keymap_name=keymap, size=size, fps=fps, verbose=bool(header_info),
                record_mode=bool(save_mode))
    game.run()


def play_craftax(cfg: DictConfig, fps: int, actions_info: bool, model_path: Path):
    device = torch.device(cfg.common.device)
    from envs.wrappers.craftax import make_craftax
    from craftax.craftax.play_craftax import BLOCK_PIXEL_SIZE_HUMAN, CraftaxRenderer, Action
    env = SingleProcessEnv(make_craftax)
    gymnax_env = env.env.env.env
    jax_env = gymnax_env._env

    recording = False
    frames_buffer = []
    record_dir = Path('media') / 'recordings'

    obs, info = env.reset()
    env_state = gymnax_env.env_state

    pixel_render_size = 64 // BLOCK_PIXEL_SIZE_HUMAN

    renderer = CraftaxRenderer(env, None, pixel_render_size=pixel_render_size)
    print(f'screen size: {renderer.screen_size}')
    renderer.render(env_state)

    agent = build_agent(env, cfg, device=device)
    if model_path is not None:
        agent.load(model_path, device, load_tokenizer=False, load_world_model=True, load_actor_critic=True)
    agent.reset_actor_critic(1, None, None)

    import pygame
    pygame.key.set_repeat(0, 0)
    clock = pygame.time.Clock()

    print(f"Press 'R' to start/stop recording a video.")

    obs_processors = {m: get_obs_processor(m) for m in env.modalities}

    def obs_to_torch(obs):
        assert isinstance(obs, dict)
        assert set(obs.keys()) == set([m.name for m in env.modalities]), f"{set(obs.keys())} != {env.modalities}"
        torch_obs = {m: obs_processors[m].to_torch(obs[m.name], device=device) for m in env.modalities}
        processed_obs = {m: obs_processors[m](v) for m, v in torch_obs.items()}
        return processed_obs

    while not renderer.is_quit_requested():
        is_record_key_pressed = any([(e.type == pygame.KEYDOWN and e.unicode == 'R') for e in renderer.pygame_events])
        if is_record_key_pressed:
            if not recording:
                recording = True
                print('Started recording.')
            else:
                print('Stopped recording.')
                Game.save_recording(record_dir, np.stack(frames_buffer))
                recording = False
                frames_buffer = []

        if recording:
            frame = pygame.display.get_surface()
            frame = np.fliplr(rotate(pygame.surfarray.array3d(frame), angle=-90))
            frames_buffer.append(frame)

        if env_state.is_sleeping or env_state.is_resting:
            action = [Action.NOOP.value]
        else:
            action = agent.act(obs_to_torch(obs))
            action = action.detach().cpu().numpy()
            if actions_info:
                print(f"Action: {action[0]} ({Action(action[0])})")

        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            env_state = gymnax_env.env_state
            renderer.render(env_state)

            if terminated or truncated:
                agent.reset_actor_critic(1, None, None)

        renderer.update()
        clock.tick(fps)

    print(type(jax_env))


def get_config(benchmark: str):
    from hydra import compose, initialize
    initialize(version_base=None, config_path="../config", job_name="play")
    overrides = [f"benchmark={benchmark}", 'hydra.run.dir=.', 'hydra.output_subdir=null']
    cfg = compose(config_name="base", overrides=overrides)
    return cfg


@click.command()
@click.option('-m', '--mode',
              type=click.Choice(['episode_replay', 'agent_in_env', 'agent_in_world_model', 'agent_in_world_model']),
              default='agent_in_env')
@click.option('-r', '--reconstruction-mode', is_flag=True, show_default=True, default=False, help='Reconstruction mode. Shows the original observation (left), downscaled observation (center), and reconstructed obs (right) - how the agent sees the world.')
@click.option('-h', '--header-info', is_flag=True, show_default=True, default=False, help='Show cumulative return, controller actions, and step info.')
@click.option('--fps', type=click.IntRange(min=1, max=240), default=15, help='frames per second')
@click.option('-p', '--model-path', type=click.Path(exists=True))
def atari(mode, reconstruction_mode, header_info, fps, model_path):
    cfg = get_config(benchmark='atari')
    play_atari(cfg, mode, reconstruction_mode, header_info, fps, model_path)


@click.command()
@click.option('-m', '--mode', type=click.Choice(['agent_in_env']), default='agent_in_env')
@click.option('--fps', type=click.IntRange(min=1, max=240), default=15, help='frames per second')
@click.option('-i', '--actions-info', is_flag=True, show_default=True, default=False, help='Print the actions of the agent.')
@click.option('-p', '--model-path', type=click.Path(exists=True))
def craftax(mode, fps, actions_info, model_path):
    cfg = get_config(benchmark='craftax')
    play_craftax(cfg, fps, actions_info, model_path)


@click.group()
def main():
    pass


main.add_command(atari)
main.add_command(craftax)


if __name__ == "__main__":
    main()
