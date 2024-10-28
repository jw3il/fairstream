from collections import OrderedDict
import math
import numpy as np

from mpd import MediaQuality

def to_np_float(a):
    if isinstance(a, list) or isinstance(a, np.ndarray):
        return np.array(a, dtype=np.float32)
    return np.array([a], dtype=np.float32)

class StreamingSession:
    # Switching parameter from minerva, but in normalized qoe scale (/100)
    # see section 8.1.3 in https://dl.acm.org/doi/10.1145/3341302.3342077
    PARAM_SWITCHING = 2.5 / 100
    # TODO: Change weights
    PARAM_INIT_STALL = 1
    PARAM_REBUFFER = 10

    PARAM_EMA_SMOOTHING = 0.8

    def __init__(
        self,
        media_quality: MediaQuality,
        buffer_capacity: int,
        time_tolerance: float
    ):
        """
        Representation of a streaming session from a single client
        :param media_quality: Assosiated Media information including qualities, corresponding bitrates, segment information
        :param buffer_capacity: Capacity of the buffer (in # segments)
        """
        self.media_quality = media_quality
        self.buffer_capacity = buffer_capacity
        self.time_tolerance = time_tolerance
        
        self.other_sessions = []
        # the duration of the video after that the client will leave the network, in milliseconds
        self.segments_left = self.media_quality.get_num_segments()
        self.media_duration = self.media_quality.get_duration()
        self.segment_duration = self.media_quality.get_segment_duration()
        self.buffer_capacity_ms = self.segment_duration * 1000 * self.buffer_capacity

        self.bitrates = self.media_quality.get_bitrates()
        self.qualities = self.media_quality.get_qualities()

        # will be set upon taking an action
        self.current_action = 0
        self.current_quality = 0.
        self.previous_action = 0
        self.previous_quality = 0.
        self.bitrate = 0
        self.bw_demand = 0
        self.total_time = 0

        # download rate is set externally
        self.download_rate = 0
        # total time spend for re-buffering in milliseconds
        self.rebuffer_time_acc = 0
        # bits of the current segment
        self.segment_bits = 0
        # remaining bits of the segment that have to be downloaded
        self.segment_bits_remaining = 0

        # remaining ms in the buffer
        self.buffer_ms = 0
        self.segment_download_time = 0
        self.segment_stall_duration = 0

        self.agent = None

        # total time spent downloading
        self.total_time_ms = 0
        # time at which we started downloading the last segment
        self.last_segment_start_ms = 0

        self.quality_switch = False
        self.done = False
        self.first_step_done = False
        self.avg_pq_differences = []

        # whether the session is currently downloading
        self.downloading = False
        self.t = 0

        # exponential moving average of the qoe
        self.current_qoe_ema_uncorrected = 0
        self.current_qoe_ema_corrected = 0

        assert self.PARAM_SWITCHING >= 0
        assert self.PARAM_REBUFFER >= 0
        assert self.PARAM_INIT_STALL >= 0
        assert 0 <= self.PARAM_EMA_SMOOTHING <= 1

    def attach_agent(self, agent):
        self.agent = agent

    def qoe_ema_iteration(self):
        qoe = self.get_qoe()
        kappa = self.PARAM_EMA_SMOOTHING
        self.current_qoe_ema_uncorrected = (
            (kappa * self.current_qoe_ema_uncorrected + (1 - kappa) * qoe)
        )
        self.current_qoe_ema_corrected = self.current_qoe_ema_uncorrected / (1 - np.power(kappa, self.t))
        assert self.current_qoe_ema_corrected <= 1

    def get_segment_download_time_slice(self) -> slice:
        """
        Get the time interval since the start of the current segment download.

        :returns: slice with time interval
        """
        return slice(self.last_segment_start_ms, self.total_time_ms)

    def get_rebuffer_factor(self):
        if self.t == 0:
            rebuffer_factor = self.PARAM_INIT_STALL
        else:
            rebuffer_factor = self.PARAM_REBUFFER

        return np.exp(-rebuffer_factor * self.rebuffer_time_acc / 1_000)

    def get_qoe(self):
        if self.t == 0:
            return 0

        smoothness = abs(self.current_quality - self.previous_quality)
        quality = (
            (
                self.current_quality
                - self.PARAM_SWITCHING * smoothness
                + self.PARAM_SWITCHING
            )
            / (1 + self.PARAM_SWITCHING)
        )

        return quality * self.get_rebuffer_factor()

    def set_next_segment_quality(self, ix):
        """
        Set the quality of the next segment

        :param ix: quality index
        """
        assert self.segment_bits_remaining == 0, (
            "Switching qualities while downloading is not allowed! "
            f"There are {self.segment_bits_remaining} bits remaining."
        )
        # initial quality switch does not count
        if self.t > 0:
            self.quality_switch = self.bitrate != self.bitrates[ix]
            self.previous_quality = self.current_quality
        else:
            # set the initial quality
            self.previous_quality = self.qualities[ix]

        self.previous_action = self.current_action
        self.current_action = ix
        self.current_quality = self.qualities[ix]
        # override previous values if this was the first step in the
        # environment to not punish high initial qualities
        if not self.first_step_done:
            self.previous_action = self.current_action
            self.previous_quality = self.current_quality
        self.bitrate = self.bitrates[ix]
        self._update_demand()
        self.segment_bits = self.segment_duration * self.bitrate
        self.segment_bits_remaining = self.segment_bits
        self.last_segment_start_ms = self.total_time_ms
        self.segment_download_time = 0
        self.segment_stall_duration = 0
        self.downloading = True

    def _buffer_is_full(self):
        return self._time_until_free_buffer() > 0

    def _time_until_free_buffer(self):
        diff = (self.buffer_ms + self.segment_duration * 1000) - self.buffer_capacity_ms
        return max(diff, 0)

    def _update_demand(self):
        if self.done or self._buffer_is_full():
            self.bw_demand = 0
        else:
            self.bw_demand = self.bitrate

    def estimate_next_event_time(self, download_rate):
        if self.done or download_rate <= 0:
            return float('inf')

        if self._buffer_is_full():
            return self._time_until_free_buffer()

        if self.segment_bits_remaining == 0:
            return 0
        else:
            # we are downloading with rate > 0 but are not done yet
            # * 1000 to convert to ms
            return (self.segment_bits_remaining / download_rate) * 1_000

    def simulate(self, elapsed_time_ms, download_rate):
        # video already ended
        if self.done:
            return False, self.done

        self.total_time_ms += elapsed_time_ms

        # only streaming because we can't download
        if self._buffer_is_full():
            assert not self.downloading

            self.buffer_ms -= elapsed_time_ms
            self.rebuffer_time_acc = 0
            self._update_demand()
            # get a new action as soon as the buffer is not full anymore
            action_required = not self._buffer_is_full()
            return action_required, self.done

        # only streaming because the bandwidth is 0 (downloading at 0 speed)
        if download_rate == 0:
            if self.buffer_ms < elapsed_time_ms:
                # rebuffering starts when the buffer is empty
                self.buffer_ms = 0
                self.rebuffer_time_acc += elapsed_time_ms - self.buffer_ms
            else:
                # buffer has enough time left, stream video
                self.buffer_ms -= elapsed_time_ms

            self._update_demand()
            return False, self.done

        # download and streaming
        download_time_ms = min((self.segment_bits_remaining / (download_rate*0.001)), elapsed_time_ms)
        self.segment_download_time += download_time_ms

        # if the buffer level is lower than the download time, simulate streaming until the buffer is empty
        if self.buffer_ms < download_time_ms:
            self.rebuffer_time_acc += download_time_ms - self.buffer_ms
            self.buffer_ms = 0
        else:
            self.rebuffer_time_acc = 0
            self.buffer_ms -= elapsed_time_ms

        # calculate how much is left (better numerical stability as using the actual download time)
        self.segment_bits_remaining -= elapsed_time_ms * 0.001 * download_rate
        if self.segment_bits_remaining <= self.time_tolerance:
            self.segment_bits_remaining = 0

        action_required = False
        # segment has been downloaded
        # -> increase buffer and request new action (if not done)
        if self.segment_bits_remaining == 0:
            if self.downloading:
                # we are done downloading the segment
                self.segments_left -= 1
                self.t += 1
                self.qoe_ema_iteration()
                self.done = self.segments_left == 0
                self.buffer_ms += self.segment_duration * 1000
                self.first_step_done = True
                self.downloading = False

            # check if we can directly continue downloading
            # after adding this segment to the buffer
            action_required = not self.done and not self._buffer_is_full()

        self._update_demand()
        return action_required, self.done

    def _segment_download_rate(self):
        if self.segment_download_time == 0:
            return 0

        downloaded_bits = self.segment_bits - self.segment_bits_remaining
        return np.float32(downloaded_bits / (self.segment_download_time * 0.001))

    def _get_bitrate_in_mbit(self):
        return self.bitrate / 1_000_000

    def _get_t_init(self):
        if self.t == 1:
            return self.rebuffer_time_acc
        return 0

    def _get_t_reb(self):
        if self.t == 1:
            return 0
        return self.rebuffer_time_acc

    def get_state(self):
        state = {
            "qoe": to_np_float(self.get_qoe()),
            "qoe_ema": to_np_float(self.current_qoe_ema_corrected),
            "quality": to_np_float(self.current_quality),
            "bitrate": to_np_float(self._get_bitrate_in_mbit()),
            "download_time": to_np_float(self.segment_download_time / 1_000),
            "init_time": to_np_float(self._get_t_init() / 1_000),
            "rebuffer_time": to_np_float(self._get_t_reb() / 1_000),
            "buffer": to_np_float(self.buffer_ms / 1_000),
            "remaining": to_np_float(self.segments_left),
            "next_bitrates": np.divide(to_np_float(self.bitrates), 1_000_000, dtype=np.float32),
            "next_qualities": to_np_float(self.qualities)
        }
        return OrderedDict(state)

    def get_info(self):
        info = {
            "quality_diff": abs(self.current_quality - self.previous_quality),
            "quality_switch": self.quality_switch,
            "qoe": self.get_qoe(),
            "qoe_ema_corrected": self.current_qoe_ema_corrected,
            "qoe_ema_uncorrected": self.current_qoe_ema_uncorrected,
            "t": self.t,
            "total_duration": self.total_time_ms,
            "quality": self.current_quality,
            "bitrate": self._get_bitrate_in_mbit(),
            "download_rate":  self._segment_download_rate(),
            "download_time": self.segment_download_time / 1_000,
            "init_time": self._get_t_init() / 1_000,
            "rebuffer_time": self._get_t_reb() / 1_000,
            "buffer": self.buffer_ms / 1_000,
        }
        return OrderedDict(info)
