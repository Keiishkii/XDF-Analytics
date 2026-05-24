#region [ Imports ]
from __future__ import annotations
from typing import Any
from enum import Enum
from IPython.core.display_functions import display
from typing import TYPE_CHECKING
from MINE.Log import Log
from MINE.Objects.ExperimentAnalytics import ExperimentAnalytics
from MINE.Objects.SessionAnalytics import SessionAnalytics

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