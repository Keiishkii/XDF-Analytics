#region [ Imports ]
from __future__ import annotations
from typing import Any
from enum import Enum
from IPython.core.display_functions import display
from typing import TYPE_CHECKING
from MINE.Log import Log

if TYPE_CHECKING: from MINE.StreamFilter import IStreamFilter, TimestampStreamFilter
if TYPE_CHECKING: from MINE.SessionFilters import ISessionFilter, ContainsStreamSessionFilter

import re
import pyxdf
import pandas as pd
import os
import contextlib
import io

#endregion [ Imports ]

#region [ Enum ][ Export Method ]
class ExportMethod(Enum):
    CSV = 1
    XLSX = 2
#endregion

#region [ Class ][ SessionAnalytics ]
class SessionAnalytics:
    #region [ Initialisation ]
    def __init__(self):
        self.file_name: str | None = None
        self.stream_count: int = 0
        self.stream_information: pd.DataFrame | None = None
        self.stream_data_dictionary: dict[str, pd.DataFrame] = dict[str, pd.DataFrame]()
        pass

    @classmethod
    def create_from_path(cls, path: str, name: str = None) -> SessionAnalytics:
        """
        :param path: The path to the xdf file to be imported.
        :param name: A descriptive name used to identify the xdf files data. Does not affect importing the data.
        :return: An MINE object containing the organised data and header of the xdf file.
        """

        xdf_data, xdf_header = pyxdf.load_xdf(path)
        return cls.create_from_raw_data(xdf_data, xdf_header, name)

    @classmethod
    def create_from_raw_data(cls, xdf_data, xdf_header, name: str = None) -> SessionAnalytics:
        """
        :param xdf_data: The raw data from the xdf file.
        :param xdf_header: The raw header from the xdf file.
        :param name: A descriptive name used to identify the xdf files data. Does not affect importing the data.
        :return: An MINE object containing the organised data and header of the xdf file.
        """

        def generate_content_dataframe(stream_data: dict) -> pd.DataFrame:
            return pd.DataFrame([
                {"Value": value, "Timestamp": timestamp}
                for value, timestamp in zip(stream_data["time_series"], stream_data["time_stamps"])
            ])
        def get_unique_stream_name(base_name: str, existing_names: dict[str, pd.DataFrame]) -> str:
            if base_name not in existing_names: return base_name

            suffix_index = 1
            while f"{base_name} (Duplicate) {suffix_index}" in existing_names: suffix_index += 1

            Log.warning(f"Duplicate stream name found '{stream_name}', added to dataframe dictionary as '{stream_name} (Duplicate) {suffix_index}'")
            return f"{base_name} (Duplicate) {suffix_index}"

        analytics = cls()
        analytics.file_name = name

        analytics.stream_data_dictionary = {}
        analytics.stream_information = pd.DataFrame(columns=[
            "Stream ID",
            "Stream Name",
            "Type",
            "Channels",
            "Channel Count",
            "Channel Format",
            "Nominal Sample Rate",
            "Effective Sample Rate",
            "Samples"
        ])

        for stream in xdf_data:
            stream_info = stream["info"]
            stream_name = stream_info["name"][0]

            stream_desc = stream_info["desc"]
            stream_channel_list = (stream_desc[0]["channels"][0]["channel"]
                      if (stream_desc is not None 
                          and isinstance(stream_desc, list)
                          and len(stream_desc) > 0
                          and isinstance(stream_desc[0], dict)
                          and "channels" in stream_desc[0]
                          and stream_desc[0]["channels"] is not None
                          and len(stream_desc[0]["channels"]) > 0
                          and "channel" in stream_desc[0]["channels"][0])
                      else None)

            channel_labels = [channel["label"][0] for channel in stream_channel_list] if stream_channel_list is not None else None

            stream_content_dataframe = generate_content_dataframe(stream)

            if stream_content_dataframe.empty:
                Log.warning(f"Stream '{stream_name}' is empty. Skipping.")
                continue

            unique_stream_name = get_unique_stream_name(stream_name, analytics.stream_data_dictionary)

            analytics.stream_data_dictionary[unique_stream_name] = stream_content_dataframe
            analytics.stream_information.loc[len(analytics.stream_information)] = [
                stream_info["stream_id"],
                unique_stream_name,
                stream_info["type"][0],
                channel_labels,
                stream_info["channel_count"][0],
                stream_info["channel_format"][0],
                stream_info["nominal_srate"][0],
                stream_info["effective_srate"],
                len(stream_content_dataframe)]

        analytics.file_name = name
        analytics.stream_count = len(xdf_data)
        analytics.stream_data_dictionary = dict(sorted(analytics.stream_data_dictionary.items()))
        analytics.stream_information.sort_values(by=["Stream Name"], inplace=True, ignore_index=True)

        return analytics
    #endregion

    #region [ Modify ]
    def localise_timestamps(self, local_zero: float | None = None) -> None:
        if local_zero is None:
            lowest_timestamp = float("inf")
            for stream_name, stream_dataframe in self.stream_data_dictionary.items():
                lowest_timestamp = min(lowest_timestamp, stream_dataframe["Timestamp"].min())

            for stream_name, stream_dataframe in self.stream_data_dictionary.items():
                stream_dataframe["Timestamp"] -= lowest_timestamp
        else:
            for stream_name, stream_dataframe in self.stream_data_dictionary.items():
                stream_dataframe["Timestamp"] -= local_zero

    #endregion

    #region [ Markers ]
    def get_paired_markers(self, marker_stream: str, starting_marker_suffix: str, ending_marker_suffix: str) -> pd.DataFrame:
        """
        :param marker_stream: The name of the marker stream to be used for marker pairing. This stream should contain markers with the suffix specified in starting_marker_suffix and ending_marker_suffix, respectively.
        :param starting_marker_suffix: The marker suffix that is used to identify a starting marker.
        :param ending_marker_suffix: The marker suffix that is used to identify an ending marker.
        :return: Returns a dataframe containing the paired start and end markers, as well as their timestamps and duration.
        """

        def find_end_marker(marker_dataframe: pd.DataFrame, base_marker: str, index: int) -> tuple[str, float] | None:
            for i in range(index + 1, len(marker_dataframe)):
                ending_marker = marker_dataframe.iloc[i]["Value"][0]

                if not ending_marker.endswith(ending_marker_suffix): continue
                if base_marker != ending_marker[:-len(ending_marker_suffix)]: continue

                return str(ending_marker), float(marker_dataframe.at[marker_dataframe.index[i], "Timestamp"])
            return None

        marker_dataframe = self.stream_data_dictionary[marker_stream]
        marker_pairs = pd.DataFrame(columns=[
            "Marker",
            "Start Marker",
            "End Marker",
            "Start Timestamp",
            "End Timestamp",
            "Duration"
        ])

        for i in range(0, len(marker_dataframe)):
            starting_marker: str = marker_dataframe.at[i, "Value"][0]
            if not starting_marker.endswith(starting_marker_suffix): continue

            base_marker: str = starting_marker[:-len(starting_marker_suffix)]
            ending_marker, ending_timestamp = find_end_marker(marker_dataframe, base_marker, i)

            if (ending_marker is None) or (ending_timestamp is None): continue

            starting_timestamp = marker_dataframe.iloc[i]["Timestamp"]
            duration = ending_timestamp - starting_timestamp

            marker_pairs.loc[len(marker_pairs)] = [
                base_marker,
                starting_marker,
                ending_marker,
                starting_timestamp,
                ending_timestamp,
                duration
            ]

        return marker_pairs
    #endregion

    #region [ Filter ]
    def get_filtered_subset(self, dataframe_filter: IStreamFilter | list[IStreamFilter]) -> SessionAnalytics:
        """
        :param dataframe_filter: The filter or filters to be applied to the dataframes.
        :return: Returns a new Analytics object containing the now filtered dataframes.
        """

        filtered_analytics = self

        dataframe_filter_list = dataframe_filter if isinstance(dataframe_filter, list) else [dataframe_filter]
        for current_filter in dataframe_filter_list:
            filtered_analytics = current_filter.evaluate(filtered_analytics)

        return filtered_analytics
    #endregion

    #region [ Display ]
    def display_network_information(self) -> None:
        display(self.stream_information)

    def display_stream(self, stream: str) -> None:
        display(self.stream_data_dictionary[stream])
    #endregion

    #region [ Exports ]
    def export_network_information(self, directory: str = "exports", export_method: ExportMethod = ExportMethod.CSV) -> None:
        filename: str = "Network Information"
        os.makedirs(directory, exist_ok=True)

        match export_method:
            case ExportMethod.CSV:
                self.stream_information.to_csv(f"{directory}/{filename}.csv", index=False)
            case ExportMethod.XLSX:
                self.stream_information.to_excel(f"{directory}/{filename}.xlsx", index=False)

    def export_stream(self, stream: str, directory: str = "exports", export_method: ExportMethod = ExportMethod.CSV) -> None:
        filename: str = f"Stream Data - {re.sub(r"^\w\s.-", '_', stream).strip()}"
        os.makedirs(directory, exist_ok=True)

        if stream not in self.stream_data_dictionary:
            Log.error(f"Stream '{stream}' not found in dataframes. Could not export.")
            return

        stream_dataframe = self.stream_data_dictionary[stream]

        match export_method:
            case ExportMethod.CSV:
                stream_dataframe.to_csv(f"{directory}/{filename}.csv", index=False)
            case ExportMethod.XLSX:
                stream_dataframe.to_excel(f"{directory}/{filename}.xlsx", index=False)
    #endregion
#endregion

#region [ Class ][ ExperimentAnalytics ]
class ExperimentAnalytics:
    #region [ Initialisation ]
    def __init__(self):
        self.analytics_dataframe: pd.DataFrame | None = None
        self.file_count: int = 0

    @classmethod
    def create_from_paths(cls, participant_dataframe: pd.DataFrame) -> ExperimentAnalytics:
        """
        :param paths: A data frame representing the participant data, with columns "Participant_ID" and "File_Path" containing the participant ID and path to the xdf file, respectively..
        :return: A Multi-file MINE object containing the organised information for each of the xdf files.
        """

        session_analytics = cls()
        session_analytics.analytics_dataframe = pd.DataFrame(columns=[
            "Session_ID",
            "Analysis_Object",
            "Streams",
            "Stream_Count",
            "Valid"
        ])

        for _, row in participant_dataframe.iterrows():
            participant_id = row["Participant_ID"]
            path = row["File_Path"]

            Log.message(f"Processing: {participant_id}")
            analysis_object = SessionAnalytics.create_from_path(path, f"{participant_id}")
            session_analytics.analytics_dataframe.loc[len(session_analytics.analytics_dataframe)] = [
                analysis_object.file_name,
                analysis_object,
                analysis_object.stream_data_dictionary.keys(),
                analysis_object.stream_count,
                True
            ]

        session_analytics.file_count = len(session_analytics.analytics_dataframe)
        return session_analytics
    #endregion

    #region [ Filter ]
    def get_filtered_subset(self, dataframe_filter: ISessionFilter | list[ISessionFilter]) -> ExperimentAnalytics:
        """
        :param dataframe_filter: The filter or filters to be applied to the dataframes.
        :return: Returns a new Analytics object containing the now filtered dataframes.
        """

        filtered_session_analytics = self

        dataframe_filter_list = dataframe_filter if isinstance(dataframe_filter, list) else [dataframe_filter]
        for current_filter in dataframe_filter_list:
            filtered_session_analytics = current_filter.evaluate(filtered_session_analytics)

        filtered_session_analytics.file_count = len(filtered_session_analytics.analytics_dataframe)
        return filtered_session_analytics
    #endregion
#endregion






#region [ Filtering Dictionaries ]
def get_subset_between_timestamps(dataframe_dictionary: dict, start: float, end: float) -> dict[str, pd.DataFrame]:
    """
    :param dataframe_dictionary: The original dictionary of which you wish to extract a subset.
    :param start: The starting timestamp in Unix time.
    :param end: The ending timestamp in Unix time.
    :return: A dictionary of dataframes, where each dataframe represents a stream in the xdf file.
        Samples outside the start and end timestamps are discarded.
    """

    def is_empty(dataframe: pd.DataFrame) -> bool:
        return any(col not in dataframe.columns for col in ["Value", "Timestamp"]) or len(dataframe) == 0

    subset_dictionary = {}

    for key, value in dataframe_dictionary.items():
        subset_dictionary[key] = pd.DataFrame(columns = ["Value", "Timestamp"]) if is_empty(value) else pd.DataFrame([
            {"Value": value, "Timestamp": timestamp}
            for value, timestamp in zip(value["Value"], value["Timestamp"])
            if start <= timestamp <= end
        ])

    return subset_dictionary

def get_subset_between_stream_values(dataframe_dictionary: dict, stream_name: Any, starting_value: Any, ending_value: Any) -> dict[str, pd.DataFrame] | None:
    """
    :param dataframe_dictionary: The original dictionary of which you wish to extract a subset.
    :param stream_name: The stream used to perform the value lookup.
    :param starting_value: The value signifying the starting time of the new subset.
    :param ending_value: The value signifying the ending time of the new subset.
    :return: A dictionary of dataframes, where each dataframe represents a stream in the xdf file,
        Samples outside the given start and end times are ignored are discarded.
    """

    stream_dataframe = dataframe_dictionary.get(stream_name)
    if stream_dataframe is None:
        Log.error(f"Stream {stream_name} not found in dataframe dictionary.")
        return None

    start_timestamp = get_timestamp_from_value(stream_dataframe, starting_value)
    if start_timestamp is None:
        Log.error(f"Starting value {starting_value} not found in stream {stream_name}.")
        return None

    end_timestamp = get_timestamp_from_value(stream_dataframe, ending_value)
    if end_timestamp is None:
        Log.error(f"Ending value {ending_value} not found in stream {stream_name}.")
        return None

    return get_subset_between_timestamps(dataframe_dictionary, start_timestamp, end_timestamp)
#endregion

#region [ Retrieving Data ]
def get_timestamp_from_value(stream_dataframe: pd.DataFrame, value: Any) -> float | None:
    """
    :param stream_dataframe: The stream dataframe from which you wish to perform the timestamp lookup.
    :param value: The value used to perform the timestamp lookup.
    :return: Returns the timestamp of the first sample with the given value, or None if no such sample exists.
    """
    exists = any(row[0] == value for row in stream_dataframe["Value"])
    if not exists: Log.warning(f"Value {value} not found in stream.")
    return stream_dataframe.loc[stream_dataframe["Value"].apply(lambda x: x[0] == value), "Timestamp"].iloc[0] if exists else None

def get_value_from_timestamp(stream_dataframe: pd.DataFrame, timestamp: float) -> Any | None:
    """
    :param stream_dataframe: The stream dataframe from which you wish to perform the value lookup.
    :param timestamp: The timestamp used to perform the value lookup.
    :return: Returns the value of the sample at the given timestamp, or None if no such sample exists.
    """
    exists = any(row[0] == timestamp for row in stream_dataframe["Timestamp"])
    if not exists: Log.warning(f"Timestamp {timestamp} not found in stream.")
    return stream_dataframe.loc[stream_dataframe["Timestamp"].apply(lambda x: x[0] == timestamp), "Value"].iloc[0] if exists else None

def get_sample_from_closest_timestamp(stream_dataframe: pd.DataFrame, timestamp: float) -> tuple[Any, float] | None:
    """
    :param stream_dataframe: The stream dataframe from which you wish to perform the value lookup.
    :param timestamp: The timestamp used to perform the value lookup.
    :return: Returns the timestamp of the sample at the time closest to the given timestamp, or None if no such sample exists.
    """
    if stream_dataframe.empty:
        Log.warning("Cannot find closest value: dataframe is empty.")
        return None

    _closest_index = (stream_dataframe["Timestamp"] - timestamp).abs().idxmin()
    return stream_dataframe.iloc[_closest_index]["Value"], stream_dataframe.iloc[_closest_index]["Timestamp"]

def get_value_from_closest_timestamp(stream_dataframe: pd.DataFrame, timestamp: float) -> Any | None:
    """
    :param stream_dataframe: The stream dataframe from which you wish to perform the value lookup.
    :param timestamp: The timestamp used to perform the value lookup.
    :return: Returns the value of the sample at the time closest to the given timestamp, or None if no such sample exists.
    """
    if stream_dataframe.empty:
        Log.warning("Cannot find closest value: dataframe is empty.")
        return None

    _closest_index = (stream_dataframe["Timestamp"] - timestamp).abs().idxmin()
    return stream_dataframe.iloc[_closest_index]["Value"]

def get_timestamp_from_closest_timestamp(stream_dataframe: pd.DataFrame, timestamp: float) -> float | None:
    """
    :param stream_dataframe: The stream dataframe from which you wish to perform the value lookup.
    :param timestamp: The timestamp used to perform the value lookup.
    :return: Returns the timestamp of the sample at the time closest to the given timestamp, or None if no such sample exists.
    """
    if stream_dataframe.empty:
        Log.warning("Cannot find closest value: dataframe is empty.")
        return None

    _closest_index = (stream_dataframe["Timestamp"] - timestamp).abs().idxmin()
    return stream_dataframe.iloc[_closest_index]["Timestamp"]
#endregion