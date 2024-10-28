import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.policy.policy import Policy
import gymnasium.spaces as spaces


class FixedActionPolicy(Policy):
    """Policy that chooses a fixed action"""

    def __init__(self, action, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.action = action

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        # return action batch, RNN states, extra values to include in batch
        return [self.action for _ in obs_batch], [], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        return {}

    @override(Policy)
    def get_weights(self):
        return {}

    @override(Policy)
    def set_weights(self, weights):
        pass


class HighestBitratePolicy(FixedActionPolicy):
    """Policy that chooses the highest bitrate"""

    def __init__(self, observation_space, action_space, config):
        # if isinstance(action_space, spaces.Tuple):
        #     action_space = action_space[0]
        #     print("WARNING: Ignoring weight action space")
        action = action_space.start + action_space.n - 1
        FixedActionPolicy.__init__(
            self, action, observation_space, action_space, config
        )


class LowestBitratePolicy(FixedActionPolicy):
    """Policy that chooses the lowest bitrate"""

    def __init__(self, observation_space, action_space, config):
        # if isinstance(action_space, spaces.Tuple):
        #     action_space = action_space[0]
        #     print("WARNING: Ignoring weight action space")
        action = action_space.start
        FixedActionPolicy.__init__(
            self, action, observation_space, action_space, config
        )


def get_flattened_index_len(obs_space, key):
    idx = 0
    for k in obs_space:
        o = obs_space[k]
        if isinstance(o, spaces.Box):
            o_len = len(o)
        else:
            raise NotImplementedError()

        if k == key:
            return idx, o_len

        idx += o_len

class GreedyKPolicy(Policy):
    """Policy that chooses a greedy action based on the bandwidth of the last k segments"""

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.k = config["k"]
        self.bitrate_index = 3
        self.download_time_index = 4
        self.next_bitrates_index, self.next_bitrates_len = 9, action_space.n

    def get_initial_state(self):
        # first element counts valid elements,
        # rest is 2xk for bitrates and download times
        return [np.zeros(self.k * 2 + 1, dtype=np.float32)]

    def is_recurrent(self) -> bool:
        return True

    def num_state_tensors(self) -> int:
        return 1

    def state_get_bitrates(self, state):
        num = int(state[0])
        return state[1:1+num]

    def state_get_download_times(self, state):
        num = int(state[0])
        return state[1+self.k:1+self.k+num]

    def state_add_bitrate_and_download_time(self, state, bitrate, time):
        # increment counter
        state[0] = min(state[0] + 1, self.k)
        # shift bitrates by 1 to right and add new one
        start = 1
        end = 1 + self.k
        state[start+1:end] = state[start:end-1]
        state[start] = bitrate
        # same for times
        start = 1 + self.k
        end = 1 + self.k + self.k
        state[start+1:end] = state[start:end-1]
        state[start] = time
        return state

    @override(Policy)
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        next_actions = []
        next_states = []
        # print("obs batch", obs_batch)
        # print("state batches", state_batches)
        for obs, state in zip(obs_batch, state_batches[0]):
            # print("state:", state)
            # add observed bitrates and download times to state
            new_bitrate = obs[self.bitrate_index]
            new_time = obs[self.download_time_index]
            if new_time > 0:
                # only add rates if we actually downloaded something
                state = self.state_add_bitrate_and_download_time(
                    state, new_bitrate, new_time
                )

            # compute average bitrate
            download_times = self.state_get_download_times(state)
            bitrates = self.state_get_bitrates(state)
            # print("Download times", download_times)
            # print("Bitrates", bitrates)
            segment_size = bitrates  # because segment length is 1 second
            if download_times.sum() == 0:
                avg_bitrate = 0
            else:
                avg_bitrate = (download_times / download_times.sum() * segment_size / download_times).sum()

            # select max possible bitrate
            next_bitrates = obs[self.next_bitrates_index:self.next_bitrates_index+self.next_bitrates_len]
            action = 0
            while action + 1 < len(next_bitrates) and next_bitrates[action + 1] <= avg_bitrate:
                action += 1

            # print("Next", next_bitrates, "avail", avg_bitrate, "selected", action)
            next_actions.append(action)
            next_states.append(state)

        return next_actions, [np.stack(next_states)], {}

    @override(Policy)
    def learn_on_batch(self, samples):
        return {}

    @override(Policy)
    def get_weights(self):
        return {}

    @override(Policy)
    def set_weights(self, weights):
        pass
