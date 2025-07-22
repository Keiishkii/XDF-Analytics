from __future__ import annotations
from MINE.Analysis import SessionAnalytics

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