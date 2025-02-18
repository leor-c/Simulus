from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import InterpolationMode, resize

from agent import Agent
from envs import SingleProcessEnv, POPWMEnv4Play
from game.keymap import get_keymap_and_action_names
from utils.preprocessing import get_obs_processor

from utils import ObsModality


class AgentEnv:
    def __init__(self, agent: Agent, env: SingleProcessEnv, keymap_name: str, do_reconstruction: bool) -> None:
        assert isinstance(env, SingleProcessEnv) or isinstance(env, POPWMEnv4Play)
        self.agent = agent
        self.env = env
        _, self.action_names = get_keymap_and_action_names(keymap_name, env.env)
        self.do_reconstruction = do_reconstruction
        self.obs = None
        self._t = None
        self._return = None
        self.obs_processors = {m: get_obs_processor(m) for m in env.modalities}

    @torch.no_grad()
    def _to_tensor(self, obs: dict[str, np.ndarray]):
        assert isinstance(obs, dict)
        assert set(obs.keys()) == set([m.name for m in self.env.modalities]), f"{set(obs.keys())} != {self.env.modalities}"
        device = self.agent.device
        torch_obs = {m: self.obs_processors[m].to_torch(obs[m.name], device=device) for m in self.env.modalities}
        processed_obs = {m: self.obs_processors[m](v) for m, v in torch_obs.items()}
        return processed_obs

    @torch.no_grad()
    def _to_array(self, obs: torch.FloatTensor):
        assert obs.ndim == 4 and obs.size(0) == 1
        return obs[0].mul(255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    def reset(self):
        obs, info = self.env.reset()
        self.obs = self._to_tensor(obs) if isinstance(self.env, SingleProcessEnv) else obs
        self.agent.actor_critic.reset(1)
        self._t = 0
        self._return = 0
        return obs

    def step(self, *args, **kwargs):
        with torch.no_grad():
            act = self.agent.act(self.obs, should_sample=True).cpu().numpy()
        obs, reward, terminated, truncated, info = self.env.step(act)
        self.obs = self._to_tensor(obs) if isinstance(self.env, SingleProcessEnv) else obs
        self._t += 1
        self._return += reward[0]
        info = {
            'timestep': self._t,
            'action': self.action_names[act[0]],
            'return': self._return,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> Image.Image:
        assert self.obs[ObsModality.image].size() == (1, 3, 64, 64)
        original_obs = self.env.env.unwrapped.original_obs if isinstance(self.env, SingleProcessEnv) else self._to_array(self.obs)
        if self.do_reconstruction:
            rec = torch.clamp(self.agent.tokenizer.encode_decode(self.obs, should_preprocess=True, should_postprocess=True)[ObsModality.image], 0, 1)
            try:
                rec = self._to_array(resize(rec, original_obs.shape[:2], interpolation=InterpolationMode.NEAREST_EXACT))
                resized_obs = self._to_array(
                    resize(self.obs[ObsModality.image], original_obs.shape[:2], interpolation=InterpolationMode.NEAREST_EXACT))
            except AttributeError:
                rec = self._to_array(resize(rec, original_obs.shape[:2], interpolation=InterpolationMode.NEAREST))
                resized_obs = self._to_array(
                    resize(self.obs[ObsModality.image], original_obs.shape[:2], interpolation=InterpolationMode.NEAREST))
            arr = np.concatenate((original_obs, resized_obs, rec), axis=1)
        else:
            arr = original_obs
        return Image.fromarray(arr)
