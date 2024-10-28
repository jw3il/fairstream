from typing import Dict, Optional
from xml.dom import minidom
import abc
import numpy as np
import math

DEFAULT_REPS=[
                (512, 288, 449480),
                (704, 396, 843768),			
                (896, 504, 1416688),
                (1280, 720, 2656696),
                (1920, 1080, 4741120),
                (3840, 2160, 7498176)
            ]


class MediaQuality(metaclass=abc.ABCMeta):
    """
    Interface for adaptive streaming.

    Assumes that all segments have the same bitrates and qualities.
    """
    def __init__(self):
        super().__init__()
        self.quality_dict = None
    
    @abc.abstractmethod
    def get_segment_duration(self) -> float:
        """
        Get the duration (in sec) of a segment.

        :return: duration
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_segments(self) -> Optional[int]:
        """
        Get the total number of segments. Can be None for an endless stream.

        :return: number of segments
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_bitrates(self) -> np.ndarray:
        """
        Get the available bitrates (in bit/sec) for a segment.

        :return: array of bitrates
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_qualities(self) -> np.ndarray:
        """
        Get the expected perceptual qualities in [0, 1] corresponding to the available bitrates.
        The order of these qualities is the same as the one of the bitrates.

        :return: array with perceptual qualities in [0, 1]
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_duration(self) -> float:
        """
        Get the total media duration

        :return: float with media length in seconds or np.inf for endless streams
        """
        raise NotImplementedError

    def get_dict(self):
        """
        Get a dictionary mapping bitrates to qualities.

        :return: bitrate-quality dict
        """
        if self.quality_dict is None:
            self.quality_dict = dict(zip(self.get_bitrates(), self.get_qualities()))
        return self.quality_dict


class SimpleDictMediaQuality(MediaQuality):
    def __init__(self, bitrate2quality, segment_duration: float, num_segments: int) -> None:
        assert isinstance(bitrate2quality, dict)
        super().__init__()

        self.segment_duration = segment_duration
        self.num_segments = num_segments
        b2q = list(bitrate2quality.items())
        b2q.sort(key=lambda tup: tup[0])
        self.bitrates = np.array([x[0] for x in b2q], dtype=float)
        self.qualities = np.array([x[1] for x in b2q], dtype=float)

    def get_segment_duration(self):
        return self.segment_duration

    def get_num_segments(self):
        return self.num_segments

    def get_bitrates(self):
        return self.bitrates

    def get_qualities(self):
        return self.qualities

    def get_duration(self):
        if self.num_segments is None:
            return np.inf
        return self.segment_duration * self.num_segments


class VideoStreamingQuality(MediaQuality):
    def __init__(self, segment_duration: float, num_segments: int, screen_size=None,representations=None) -> None:
        super().__init__()

        self.segment_duration = segment_duration
        self.num_segments = num_segments

        self.screen_size = screen_size
        self.mpd = MPD(representations=representations)

        self.bitrates = np.array([x[2] for x in self.mpd.representations], dtype=float)
        self.qualities = np.array(self._calculate_nppds(), dtype=float)

    def get_segment_duration(self):
        return self.segment_duration

    def get_num_segments(self):
        return self.num_segments

    def get_bitrates(self):
        return self.bitrates

    def get_qualities(self):
        return self.qualities

    def get_duration(self):
        if self.num_segments is None:
            return np.inf
        return self.segment_duration * self.num_segments

    def _calculate_nppds(self):
        pq = []
        for representation in self.mpd.representations:
            pq.append(round(min(math.sqrt(representation[0] ** 2 + representation[1] ** 2)
                                 / math.sqrt(self.screen_size[0] ** 2 + self.screen_size[1] ** 2), 1.0), 4))
        return pq


class HDQuality(VideoStreamingQuality):
    """Quality class of HD Streaming Device"""
    def __init__(self, segment_duration: float, num_segments: int, representations=None) -> None:
        super().__init__(segment_duration, num_segments,
                         screen_size=(1920,1080),
                         representations=representations)


class FourKQuality(VideoStreamingQuality):
    """Quality class of 4K Streaming Device"""
    def __init__(self, segment_duration: float, num_segments: int, representations=None) -> None:
        super().__init__(segment_duration, num_segments, 
                         screen_size=(3840,2160),
                         representations=representations)


class MPD():
    """ Media Presentation Description class
    A simplified Media Presentation Description containing 
    - possible representations in the for of [(width, height, bitrate),...]
    """
    def __init__(self, representations=None) -> None:
        if representations is None:
            self.representations = DEFAULT_REPS
        else:
            self.representations=representations
        self.representations.sort(key=lambda tup: tup[2])
    
    def __str__(self) -> str:
        mpd_str = f'MPD Representations: {self.representations}'
        return mpd_str
    

    def load_from_mpd(self, video_mpd_path):
        """load mpd information from real mpd file

        :param video_mpd_path: path to mpd file
        """
        dom = minidom.parse(video_mpd_path)
        self.representations = []
        elements = dom.getElementsByTagName('Representation')
        for rep in elements:
            try:
                height = int(rep.attributes['height'].value)
                width = int(rep.attributes['width'].value)
                bandwidth = int(rep.attributes['bandwidth'].value)
                self.representations.append((width, height, bandwidth))
            except KeyError:
                continue

        self.representations.sort(key=lambda tup: tup[2])