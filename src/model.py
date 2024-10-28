import torch
import torch.nn as nn
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.torch_utils import one_hot as torch_one_hot


class TorchFrameStackingModel(TorchModelV2, nn.Module):
    """A simple FC model that takes the last n observations as input."""

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, num_frames=3
    ):
        nn.Module.__init__(self)
        super(TorchFrameStackingModel, self).__init__(
            obs_space, action_space, None, model_config, name
        )

        self.num_frames = num_frames
        self.num_outputs = num_outputs

        # Construct actual (very simple) FC model.
        assert len(obs_space.shape) == 1
        in_size = self.num_frames * (obs_space.shape[0] + action_space.n)
        self.layer1 = SlimFC(in_size=in_size, out_size=256, activation_fn="tanh")
        self.layer2 = SlimFC(in_size=256, out_size=256, activation_fn="tanh")
        self.out = SlimFC(
            in_size=256, out_size=self.num_outputs, activation_fn="linear"
        )
        self.values = SlimFC(in_size=256, out_size=1, activation_fn="linear")

        self._last_value = None

        self.view_requirements["prev_n_obs"] = ViewRequirement(
            data_col="obs", shift="-{}:0".format(num_frames - 1), space=obs_space
        )
        self.view_requirements["prev_n_actions"] = ViewRequirement(
            data_col="actions",
            shift="-{}:-1".format(self.num_frames),
            space=self.action_space,
        )

    def forward(self, input_dict, states, seq_lens):
        obs = input_dict["prev_n_obs"]
        obs = torch.reshape(obs, [-1, self.obs_space.shape[0] * self.num_frames])
        actions = torch_one_hot(input_dict["prev_n_actions"], self.action_space)
        actions = torch.reshape(actions, [-1, self.num_frames * actions.shape[-1]])
        input_ = torch.cat([obs, actions], dim=-1)
        features = self.layer1(input_)
        features = self.layer2(features)
        out = self.out(features)
        self._last_value = self.values(features)
        return out, []

    def value_function(self):
        return torch.squeeze(self._last_value, -1)
