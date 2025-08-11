from __future__ import annotations
from MINE.Analysis import SessionAnalytics
from MINE.Log import Log
from numpy.typing import NDArray

import scipy.signal as signal
import pandas as pd
import numpy as np
import time

class PPG_Event:
    def __init__(self):
        self.systolic_peak_value: float = 0
        self.dicrotic_notch_value: float | None = None
        self.diastolic_peak_value: float | None = None
        self.diastolic_trough_value: float | None = None

        self.dicrotic_notch_timestamp_offset: float | None = None
        self.diastolic_peak_timestamp_offset: float | None = None
        self.diastolic_trough_timestamp_offset: float | None = None
        pass

class StreamProcesses:

    #region [ Stream Generation ]
    @staticmethod
    def butterworth_filter(session: SessionAnalytics, input_stream: str, output_stream: str):
        """
        :param session: The session object to read and write to.
        :param input_stream: The name of the input stream.
        :param output_stream: The name of the output stream.
        :return:
        """

        input_stream_data: pd.DataFrame = session.stream_data_dictionary[input_stream]

        # region [ Copy Inputs ]
        timestamps = input_stream_data["Timestamp"].copy()
        values = input_stream_data["Value"].copy()
        # endregion

        first_timestamp: float = timestamps.iloc[0]
        ten_seconds_in = first_timestamp + 10

        # region [ Set Initial 10 Seconds to 0 ]
        ten_second_index: int = timestamps[timestamps > ten_seconds_in].index[0]
        values.iloc[0: ten_second_index - 1] = 0
        # endregion

        # region [ Centre Values Around 0]
        first_value: float = values.iloc[ten_second_index]
        values.iloc[ten_second_index - 1:] = values.iloc[ten_second_index - 1:] - first_value
        # endregion

        # region [ Frequency Band Filter ]
        butterworth_filter_data = signal.butter(5, [1, 12], btype='band', fs=25, output='ba')
        values = signal.filtfilt(butterworth_filter_data[0], butterworth_filter_data[1], values)
        # endregion

        output_stream_data: pd.DataFrame = pd.DataFrame({
            "Value": values,
            "Timestamp": timestamps
        })

        session.stream_data_dictionary[output_stream] = output_stream_data

    @staticmethod
    def detect_ppg_peaks(session: SessionAnalytics, input_stream: str, output_stream: str):
        """
        :param session: The session object to read and write to.
        :param ppg_stream: The name of the input stream.
        """

        input_stream_data: pd.DataFrame = session.stream_data_dictionary[input_stream]

        timestamps: np.ndarray[float] = input_stream_data['Timestamp'].to_numpy()
        values: np.ndarray[float] = input_stream_data['Value'].to_numpy()

        systolic_peak_indexes: np.ndarray[int] = signal.find_peaks(values, distance= 25 * 0.5, height=0.2)[0]
        systolic_peak_timestamps: np.ndarray[float] = timestamps[systolic_peak_indexes]

        systolic_peak_event_annotations: list[PPG_Event] = [PPG_Event() for _ in range(systolic_peak_indexes.shape[0])]

        for i in range(systolic_peak_indexes.shape[0] - 1):
            ppg_event = systolic_peak_event_annotations[i]
            ppg_event_timestamp = systolic_peak_timestamps[i]

            systolic_peak: int = systolic_peak_indexes[i]
            next_systolic_peak = systolic_peak_indexes[i + 1]

            ppg_event.systolic_peak_value = values[systolic_peak]

            event_segment = values[systolic_peak: next_systolic_peak]

            #region [ Calculate Diastolic Trough ]
            local_valleys: np.ndarray[int] = signal.find_peaks(-event_segment, prominence=0.05)[0]
            diastolic_trough = systolic_peak + local_valleys[0] if len(local_valleys) > 0 else None

            ppg_event.diastolic_peak_value = values[diastolic_trough] if diastolic_trough else None
            ppg_event.diastolic_trough_timestamp_offset = timestamps[diastolic_trough] - ppg_event_timestamp if diastolic_trough else None
            #endregion

            if diastolic_trough:
                tail_segment = values[diastolic_trough: next_systolic_peak]

                #region [ Calculate Diastolic Peak ]
                tail_peaks: np.ndarray[int] = signal.find_peaks(tail_segment, prominence=0.02)[0]
                diastolic_peak = diastolic_trough + tail_peaks[0] if len(tail_peaks) > 0 else None

                ppg_event.diastolic_peak_value = values[diastolic_peak] if diastolic_peak else None
                ppg_event.diastolic_peak_timestamp_offset = timestamps[diastolic_peak] - ppg_event_timestamp if diastolic_peak else None
                #endregion

                #region [ Calculate Dicrotic Notch ]
                tail_valleys: np.ndarray[int] = signal.find_peaks(-tail_segment, prominence=0.02)[0]
                dicrotic_notch = diastolic_trough + tail_valleys[0] if len(tail_valleys) > 0 else None

                ppg_event.dicrotic_notch_value = values[dicrotic_notch] if dicrotic_notch else None
                ppg_event.dicrotic_notch_timestamp_offset = timestamps[dicrotic_notch] - ppg_event_timestamp if dicrotic_notch else None
                #endregion

        output_stream_data: pd.DataFrame = pd.DataFrame({
            "Value": systolic_peak_event_annotations,
            "Timestamp": systolic_peak_timestamps
        })

        output_stream_data["Value"] = output_stream_data["Value"].astype(object)
        session.stream_data_dictionary[output_stream] = output_stream_data

    @staticmethod
    def generate_peak_interval_duration_stream(session: SessionAnalytics, input_stream: str, output_stream: str):
        ppg_annotated_peaks_stream: pd.DataFrame = session.stream_data_dictionary[input_stream]

        timestamps = ppg_annotated_peaks_stream["Timestamp"].to_numpy()

        peak_displacements = np.diff(timestamps)
        mid_point_timestamps =  (timestamps[:-1] + timestamps[1:]) / 2

        session.stream_data_dictionary[output_stream] = pd.DataFrame({
            "Value": peak_displacements,
            "Timestamp": mid_point_timestamps
        })

    @staticmethod
    def generate_interval_differences_stream(session: SessionAnalytics, input_stream: str, output_stream: str):
        ppg_peak_interval_stream: pd.DataFrame = session.stream_data_dictionary[input_stream]

        values = ppg_peak_interval_stream["Value"].to_numpy()
        timestamps = ppg_peak_interval_stream["Timestamp"].to_numpy()

        interval_differences = np.diff(values)
        mid_point_timestamps =  (timestamps[:-1] + timestamps[1:]) / 2

        session.stream_data_dictionary[output_stream] = pd.DataFrame({
            "Value": interval_differences,
            "Timestamp": mid_point_timestamps
        })

    @staticmethod
    def generate_rmssd_stream(session: SessionAnalytics, input_stream: str, output_stream: str, window_size: float = 20, step_size: float = 5):
        ppg_interval_differences_stream: pd.DataFrame = session.stream_data_dictionary[input_stream]

        input_values = ppg_interval_differences_stream["Value"].to_numpy()
        intput_timestamps = ppg_interval_differences_stream["Timestamp"].to_numpy()

        output_values = []
        output_timestamps = []

        start_time = intput_timestamps[0]
        end_time = intput_timestamps[-1]

        sample_point = start_time + (window_size / 2)
        while sample_point < (end_time - (window_size / 2)):
            sample_mask = (intput_timestamps >= sample_point - (window_size / 2)) & (intput_timestamps <= sample_point + (window_size / 2))
            filtered_values = input_values[sample_mask]

            converted_samples = filtered_values * 1000
            squared_differences= converted_samples ** 2

            squared_means = np.mean(squared_differences)
            rmssd = np.sqrt(squared_means)

            output_values = np.append(output_values, rmssd)
            output_timestamps = np.append(output_timestamps, sample_point)

            sample_point = sample_point + step_size

        session.stream_data_dictionary[output_stream] = pd.DataFrame({
            "Value": output_values,
            "Timestamp": output_timestamps
        })


    @staticmethod
    def generate_sdnn_stream(session: SessionAnalytics, input_stream: str, output_stream: str, window_size: float = 20, step_size: float = 5):
        ppg_interval_differences_stream: pd.DataFrame = session.stream_data_dictionary[input_stream]

        input_values = ppg_interval_differences_stream["Value"].to_numpy()
        intput_timestamps = ppg_interval_differences_stream["Timestamp"].to_numpy()

        output_values = []
        output_timestamps = []

        start_time = intput_timestamps[0]
        end_time = intput_timestamps[-1]

        sample_point = start_time + (window_size / 2)
        while sample_point < (end_time - (window_size / 2)):
            sample_mask = (intput_timestamps >= sample_point - (window_size / 2)) & (intput_timestamps <= sample_point + (window_size / 2))
            filtered_values = input_values[sample_mask]

            converted_samples = filtered_values * 1000
            mean = np.mean(converted_samples)

            differences_from_the_mean = filtered_values - mean
            squared_differences_from_the_mean = np.square(differences_from_the_mean)

            mean_squared_difference = np.mean(squared_differences_from_the_mean)
            sdnn = np.sqrt(mean_squared_difference)

            output_values = np.append(output_values, sdnn)
            output_timestamps = np.append(output_timestamps, sample_point)

            sample_point = sample_point + step_size

        session.stream_data_dictionary[output_stream] = pd.DataFrame({
            "Value": output_values,
            "Timestamp": output_timestamps
        })
    #endregion

    #region [ Sampling ]
    @staticmethod
    def get_sdnn_sample_from_annotated_ppg(session: SessionAnalytics, input_stream: str, start_time: float, end_time: float) -> float:
        annotated_ppg_stream: pd.DataFrame = session.stream_data_dictionary[input_stream]
        annotated_ppg_stream_in_range = annotated_ppg_stream.loc[annotated_ppg_stream["Timestamp"].between(start_time, end_time)]

        ppg_peak_timestamps: NDArray[float] = annotated_ppg_stream_in_range["Timestamp"].to_numpy()

        beat_differences: NDArray[float] = np.diff(ppg_peak_timestamps)
        beat_differences_in_milliseconds: NDArray[float] = beat_differences * 1000

        standard_deviation: float = float(np.std(beat_differences_in_milliseconds))
        return standard_deviation


    @staticmethod
    def get_rmssd_sample_from_annotated_ppg(session: SessionAnalytics, input_stream: str, start_time: float, end_time: float) -> float:
        annotated_ppg_stream: pd.DataFrame = session.stream_data_dictionary[input_stream]
        annotated_ppg_stream_in_range = annotated_ppg_stream.loc[annotated_ppg_stream["Timestamp"].between(start_time, end_time)]

        ppg_peak_timestamps: NDArray[float] = annotated_ppg_stream_in_range["Timestamp"].to_numpy()

        beat_differences: NDArray[float] = np.diff(ppg_peak_timestamps)
        sequential_beat_differences: NDArray[float] = np.diff(beat_differences)
        sequential_beat_differences_in_milliseconds: NDArray[float] = sequential_beat_differences * 1000

        squared_sequential_differences = sequential_beat_differences_in_milliseconds ** 2
        mean: float = np.mean(squared_sequential_differences)
        root_of_mean: float = np.sqrt(mean)

        return root_of_mean
    #endregion



    @staticmethod
    def reset_stream_timestamps(session: SessionAnalytics, stream: str):
        input_stream_data: pd.DataFrame = session.stream_data_dictionary[stream]
        first_timestamp: float = input_stream_data["Timestamp"].iloc[0]
        input_stream_data["Timestamp"] = input_stream_data["Timestamp"] - first_timestamp

    @staticmethod
    def create_vectors_from_component_streams(session: SessionAnalytics, x_stream: str, y_stream: str, z_stream: str, output_stream: str):
        def closest_indices(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
            indices = np.searchsorted(reference, target)
            indices = np.clip(indices, 1, len(reference) - 1)
            prev = reference[indices - 1]
            next_ = reference[indices]
            closer = np.where(np.abs(target - prev) < np.abs(target - next_), indices - 1, indices)
            return closer

        x: pd.DataFrame = session.stream_data_dictionary[x_stream]
        y: pd.DataFrame = session.stream_data_dictionary[y_stream]
        z: pd.DataFrame = session.stream_data_dictionary[z_stream]

        x_timestamps: np.ndarray = x['Timestamp'].to_numpy()
        y_timestamps: np.ndarray = y['Timestamp'].to_numpy()
        z_timestamps: np.ndarray = z['Timestamp'].to_numpy()

        x_values: np.ndarray = np.concatenate(x['Value'].to_numpy())
        y_values: np.ndarray = np.concatenate(y['Value'].to_numpy())
        z_values: np.ndarray = np.concatenate(z['Value'].to_numpy())

        idx_y: np.ndarray = closest_indices(y_timestamps, x_timestamps)
        idx_z: np.ndarray = closest_indices(z_timestamps, x_timestamps)

        aligned_y_values: np.ndarray = y_values[idx_y]
        aligned_z_values: np.ndarray = z_values[idx_z]

        vectors = [np.array([x, y, z]) for x, y, z in zip(x_values, aligned_y_values, aligned_z_values)]

        output_stream_data: pd.DataFrame = pd.DataFrame({
            "Value": vectors,
            "Timestamp": x_timestamps
        })

        output_stream_data["Value"] = output_stream_data["Value"].astype(object)

        session.stream_data_dictionary[output_stream] = output_stream_data

    @staticmethod
    def calculate_magnitudes_from_vector_stream(session: SessionAnalytics, input_stream: str, output_stream: str):
        vector_stream = session.stream_data_dictionary[input_stream]

        vectors = vector_stream["Value"].to_numpy()
        timestamps: np.ndarray = vector_stream['Timestamp'].to_numpy()

        magnitudes = np.array([np.linalg.norm(vec) for vec in vectors])
        output_stream_data: pd.DataFrame = pd.DataFrame({
            "Value": magnitudes,
            "Timestamp": timestamps
        })

        session.stream_data_dictionary[output_stream] = output_stream_data
