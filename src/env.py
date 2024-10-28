from enum import Enum
import numpy as np
import gymnasium.spaces as spaces

import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from typing import Tuple, Optional
from ray.rllib.utils.typing import MultiAgentDict
from ray.rllib.utils import check_env
from collections import OrderedDict
from minerva import minerva_weights
from mpd import MediaQuality

from quality import to_megabits
from session import StreamingSession
from traces.load import load_traces


class BandwidthSharing(Enum):
    # all clients get the same share of the bandwidth (like in plain tcp)
    Equal = "equal"
    # all clients get a share proportional to their selected bitrate
    Proportional = "proportional"
    # the weight is determined by minerva, giving all clients a bandwidth
    # share that leads to equal interpolated quality
    Minerva = "minerva"
    # clients select a weight >= 0 as an action to determine their
    # share of the total bandwidth
    # Action = "action"


class StreamingEnv(MultiAgentEnv):
    def __init__(self, env_config) -> None:
        super().__init__()
        self.sessions: list[StreamingSession] = []
        self.session_mqs: list[MediaQuality] = env_config["media_qualities"]
        # TODO URGENT: UNIT CONSISTENCY, simply use mbit and seconds everywhere..
        self.session_mqs_dicts_mbit = [to_megabits(s.get_dict()) for s in self.session_mqs]

        assert len(self.session_mqs) > 0
        # we assume that all sessions start at the beginning
        self.max_active_sessions = len(self.session_mqs)

        self.client_resolutions = None
        self.all_bandwidths, self.all_bw_names = \
            StreamingEnv.load_bandwidths_from_config(env_config)

        # environment time tolerance (for event-based downloading)
        # Warning: very low values may result in numerical instability
        self.time_tolerance = env_config.get("time_tolerance", 0.0001)

        self.bw_sharing = BandwidthSharing(env_config["bw_sharing"])
        # Reward = quality_fairness_coeff * qoe + (1 - quality_fairness_coeff) * fairness
        self.quality_fairness_coeff = env_config["quality_fairness_coeff"]
        assert 0 <= self.quality_fairness_coeff <= 1, f"Invalid quality-fairness coefficient of {self.quality_fairness_coeff}"

        self.time_interval = env_config["time_interval"]
        self.time_s_idx = 0
        self.time_ms = 0
        self.sim_step = 0
        self.next_bandwidth_ms = self.time_interval

        self.bandwidths = self.all_bandwidths[0]
        self.current_bw_total = self.bandwidths[0]
        self._done = np.zeros(self.max_active_sessions, dtype=bool)
        # agents that got done in this (the last) step
        self._step_done = np.zeros(self.max_active_sessions, dtype=bool)
        # agents that requested an action in this (the last) step
        self._step_action_request = np.zeros(self.max_active_sessions, dtype=bool)
        self._truncated = np.zeros(self.max_active_sessions, dtype=bool)
        self._rewards = np.zeros(self.max_active_sessions)
        self._tmp_minerva_weights = np.ones(self.max_active_sessions)
        # stores the (unmasked) client weights
        # self._client_weights = np.ones(self.max_active_sessions)

        self._next_expected_event = np.zeros(self.max_active_sessions, dtype=float)

        self.bw_shares = None

        # Multi Agent names and corresponding indicies

        # TODO: Include heuristics as policies in rllib
        self._names = [f"{i}" for i in range(self.max_active_sessions)]
        self._ixs = {name: i for i, name in enumerate(self._names)}
        self._agent_ids = self._names

        # create dummy sessions & clients to create observation space
        self.buffer_size = env_config["buffer_size"]
        self._create_sessions()
        self.remote_rng = False if "remote_rng" not in env_config.keys() else env_config["remote_rng"]
        assert len(self.sessions) > 0

        # Observations are dictionaries with the agents and the target's location.
        self.observation_space = spaces.Dict()
        for ix, _agent_id in enumerate(self._agent_ids):
            n_actions = len(self.session_mqs[ix].bitrates)
            self.observation_space[_agent_id] = spaces.Dict(
                # WARNING: If you change anything here, you also have to change the
                # indexes used by GreedyKPolicy
                OrderedDict({
                    "qoe": spaces.Box(low=0, high=1, shape=(1,)),
                    "qoe_ema": spaces.Box(low=0, high=1, shape=(1,)),
                    "quality": spaces.Box(low=0, high=1, shape=(1,)),
                    "bitrate": spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "download_time": spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "init_time": spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "rebuffer_time": spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "buffer": spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "remaining": spaces.Box(low=0, high=np.inf, shape=(1,)),
                    "next_bitrates": spaces.Box(low=0, high=np.inf, shape=(n_actions,)),
                    "next_qualities": spaces.Box(low=0, high=1, shape=(n_actions,))
                })
            )

        # self.observation_space =  spaces.Dict({_agent_id: self._agent_obs_space for _agent_id in self._agent_ids})
        self.action_space = spaces.Dict({
            _agent_id: (
                # spaces.Tuple((spaces.Discrete(len(self.session_mqs[ix].bitrates)), spaces.Box(1, float('inf'))))
                # if self.bw_sharing == BandwidthSharing.Action else
                spaces.Discrete(len(self.session_mqs[ix].bitrates))
            )
            for ix, _agent_id in enumerate(self._agent_ids)
        })
        # uncomment for debugging
        # check_env(self)

    @staticmethod
    def load_bandwidths_from_config(env_config):
        if "bandwidths" in env_config and env_config["bandwidths"] is not None:
            # specify bandwidths directly
            all_bandwidths = env_config["bandwidths"]
            all_bw_names = env_config["bw_names"]
            # to avoid confusion due to duplicate configs
            assert "traces" not in env_config or env_config["traces"] is None
        else:
            # only specify traces file/dir
            all_bandwidths, all_bw_names = load_traces(
                env_config["traces"],
                env_config.get("traces_include", None),
                env_config.get("traces_exclude", None)
            )

        return all_bandwidths, all_bw_names

    def set_bandwidths(self, bandwidths, bw_names):
        self.all_bandwidths = np.array(bandwidths, dtype=object) # list containing the bandwidth for each time step
        self.all_bw_names = bw_names

    def set_media_qualities(self, mqs):
        """
        Set media qualities of the sessions

        :param mqs: list of media quality objects
        """
        self.session_mqs = mqs
        assert len(mqs) == 0 or len(mqs) == self.max_active_sessions

    def _create_sessions(self):
        """Creates the sessions for the current epoch"""
        self.sessions = []
        for i in range(self.max_active_sessions):
            self.sessions.append(
                StreamingSession(
                    self.session_mqs[i],
                    self.buffer_size,
                    self.time_tolerance
                )
            )

    def truncated_dict(self, ma_dict) -> dict[str, bool]:
        "return truncated dict for all agents contained in the MultiAgentDict"
        truncated = {agent: self._truncated[self._ixs[agent]] for agent in ma_dict.keys()}
        truncated["__all__"] = self._truncated.sum() == self.max_active_sessions
        return truncated

    def _get_rewards_dict(self):
        return {self._names[ix]: v for ix, v in enumerate(self._rewards)}

    def _get_obs_dict(self, agent_mask):
        return {self._names[ix]: s.get_state() for ix, s in enumerate(self.sessions) if agent_mask[ix]}

    def _get_dones_dict(self):
        dones = {self._names[ix]: v for ix, v in enumerate(self._done)}
        # Ususally truncated should not be incorporated into dones
        # But according to https://docs.ray.io/en/latest/_modules/ray/rllib/env/multi_agent_env.html
        # dones['__all__']=True is Currently in truncation scenarios
        all_done = (self._done.sum() + self._truncated.sum()) == self.max_active_sessions
        dones['__all__'] = all_done
        # if all_done:
        #     print(f"All done after sim step {self.sim_step}, time = {self.time_ms}, trunc: {self._truncated.sum()}")
        return dones

    def _get_truncated(self):
        truncs = {self._names[ix]: v for ix, v in enumerate(self._truncated)}
        truncs["__all__"] = bool(self._truncated.sum() == self.max_active_sessions)
        return truncs

    def _get_infos_dict(self, agent_mask):
        infos = {}
        for ix, s in enumerate(self.sessions):
            if agent_mask[ix]:
                info = s.get_info()
                # add information that's only available in the environment

                # note that this is only the current bandwidth at the end of the
                # step. Within the step, the bandwidth of a client may change.
                info["bw"] = self.bw_shares[ix]
                info["bw_total"] = self.current_bw_total
                info["fairness"] = self.get_fairness()
                info["sim_time"] = self.time_ms
                info["trace"] = self.bandwidths_name
                info["sim_step"] = self.sim_step
                infos[self._names[ix]] = info

        return infos

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        super().reset(seed=seed, options=options)

        # randomly select num active sessions & reset dones
        self.num_active_sessions = self.max_active_sessions
        self._done.fill(0)
        self._step_done.fill(0)
        self._step_action_request.fill(0)
        self._truncated.fill(0)
        self._rewards.fill(0)
        self._tmp_minerva_weights.fill(1)
        self.time_s_idx = 0
        self.time_ms = 0
        self.sim_step = 0
        self.next_bandwidth_ms = self.time_interval

        # create sessions & clients
        self._create_sessions()

        if self.remote_rng:
            sampler = ray.get_actor("trace_sampler")
            bw_ix = ray.get(sampler.sample.remote(self.all_bandwidths.shape[0]))
        else:
            bw_ix = np.random.randint(0, self.all_bandwidths.shape[0])

        self.bandwidths = self.all_bandwidths[bw_ix]
        self.bandwidths_name = self.all_bw_names[bw_ix]
        self.current_bw_total = self.bandwidths[0]

        self._update_bw_shares()

        # in the initial step, we request actions from all agents
        return (
            self._get_obs_dict(np.ones(self.max_active_sessions)),
            self._get_infos_dict(np.ones(self.max_active_sessions))
        )

    @staticmethod
    def calculate_bw_distribution(bw_total, weights):
        bw_distribution = []
        total_demand = sum(weights)
        if total_demand == 0:
            return np.zeros(len(weights))
        for bw_demand in weights:
            bw_percent = bw_demand / total_demand
            bw_distribution.append(bw_total * bw_percent)
        return np.array(bw_distribution)

    def _update_bw_shares(self):
        bw_total = self.bandwidths[self.time_s_idx]
        self.current_bw_total = bw_total
        bw_demand = [session.bw_demand for session in self.sessions]

        if self.bw_sharing == BandwidthSharing.Proportional:
            weights = bw_demand
        elif self.bw_sharing == BandwidthSharing.Equal:
            weights = [1 if x > 0 else 0 for x in bw_demand]
        elif self.bw_sharing == BandwidthSharing.Minerva:
            # mask utilities of inactive clients
            filtered_utilities = []
            filtered_init_weights = []
            for i, x in enumerate(bw_demand):
                if x > 0:
                    filtered_utilities.append(self.session_mqs_dicts_mbit[i])
                    filtered_init_weights.append(self._tmp_minerva_weights[i])

            # fill in weights of active clients
            # inactive clients have weight 0
            weights = np.zeros(len(bw_demand))
            if len(filtered_utilities) > 0:
                # TODO: WARNING: Adapt unit of bw_total as well
                filtered_weights = minerva_weights(filtered_utilities, bw_total / 1_000_000, init_weights=filtered_init_weights, stop_epsilon=1e-2)
                k = 0
                for i, x in enumerate(bw_demand):
                    if x > 0:
                        weights[i] = filtered_weights[k]
                        self._tmp_minerva_weights[i] = filtered_weights[k]
                        k += 1

        # elif self.bw_sharing == BandwidthSharing.Action:
        #     weights = [max(self._client_weights[i], 1) if x > 0 else 0 for i, x in enumerate(bw_demand)]

        self.bw_shares = StreamingEnv.calculate_bw_distribution(
            bw_total, weights
        )
        # print(bw_total, self.bw_shares)

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        # pass new bit rates to sessions
        for agent, action in action_dict.items():
            ix = self._ixs[agent]
            # if isinstance(action, tuple):
            #     # update weights of acting agents
            #     action, self._client_weights[ix] = action
            self.sessions[ix].set_next_segment_quality(action)

        self._rewards.fill(0)
        self._step_done.fill(0)
        self._step_action_request.fill(0)
        next_dones = self._done.copy()

        # step through sessions until at least one session requires an action
        # or until all sessions are done
        while not self._step_action_request.any() and not next_dones.all():
            self.sim_step += 1
            self._update_bw_shares()

            # Get estimates for the time of the next event
            for i, session in enumerate(self.sessions):
                next_event_time = session.estimate_next_event_time(self.bw_shares[i])
                self._next_expected_event[i] = self.time_ms + next_event_time

            # time that will elapse in this step
            dt = np.clip(
                self._next_expected_event.min() - self.time_ms,
                a_min=0,
                a_max=self.next_bandwidth_ms - self.time_ms
            )
            # dt = 1
            # dt = 0.01
            # print(f"ENV {self.time_ms}: Next expected steps at {self._next_expected_event} (next bw at {self.next_bandwidth_ms}) => dt={dt}")

            # update each session
            for i, session in enumerate(self.sessions):
                self._step_action_request[i], next_dones[i] = \
                    session.simulate(elapsed_time_ms=dt, download_rate=self.bw_shares[i])

            self.time_ms += dt

            # get reward of done clients at the current (!) simulation time
            if self._done.sum() != next_dones.sum():
                for i, session in enumerate(self.sessions):
                    if self._done[i] or not next_dones[i]:
                        continue

                    self._step_done[i] = True
                    self._rewards[i] = self.get_reward(session)
                    self._done[i] = True

            if self.time_ms >= self.next_bandwidth_ms - self.time_tolerance:
                self.time_s_idx += 1
                self.next_bandwidth_ms += self.time_interval

                if self.time_s_idx >= len(self.bandwidths):
                    # truncate all agents that are not done
                    self._truncated = np.array([not d for d in self._done], dtype=bool)
                    break

        # we have simulated until some point in time in which an agent
        # requests an action (or all agents are done)
        # => it has finished downloading a segment and we can assign
        #    the corresponding reward
        for i, session in enumerate(self.sessions):
            if not self._step_action_request[i]:
                continue

            self._rewards[i] = self.get_reward(session)

        # agents that require obs because they should
        # perform a new step or are done / truncated
        requires_obs = np.logical_or(
            np.logical_or(
                self._step_action_request,
                self._step_done
            ),
            self._truncated
        )
        new_obs = self._get_obs_dict(requires_obs)
        info = self._get_infos_dict(requires_obs)

        dones = self._get_dones_dict()
        truncateds = self._get_truncated()
        rewards = self._get_rewards_dict()

        return new_obs, rewards, dones, truncateds, info

    def get_reward(self, session: StreamingSession):
        qoe = session.get_qoe()
        fairness = self.get_fairness()
        return (
            self.quality_fairness_coeff * qoe
            + (1 - self.quality_fairness_coeff) * fairness  # * session.get_rebuffer_factor()
        )

    def get_fairness(self):
        vs = []
        for i, s in enumerate(self.sessions):
            # consider the QoE of streams that are running
            # or that just finished
            if (s.done and not self._step_done[i]) or s.t == 0:
                continue

            vs.append(s.current_qoe_ema_uncorrected)

        if len(vs) == 0:
            return 1
        return 1 - 2 * np.array(vs).std()
