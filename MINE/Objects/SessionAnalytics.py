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
#endregion

#region [ Enum ][ Export Method ]
class ExportMethod(Enum):
    CSV = 1
    XLSX = 2
#endregion



#region [ Class ][ SessionAnalytics ]
class SessionAnalytics:
    """
    An object used for representing a single session of XDF files.
    Used by the ExperimentAnalytics class to organise data by session.
    """

    #region [ Properties ]
    def __init__(self):
        self.session_id: str | None = None
        self.stream_count: int = 0
        self.stream_information: pd.DataFrame | None = None
        self.stream_data_dictionary: dict[str, pd.DataFrame] = dict[str, pd.DataFrame]()
        self.is_valid: bool = False
        pass
    #endregion

    def __repr__(self):
        return f"<Session: {self.session_id}>"

    #region [ Initialisation ]

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

        session_analytics = cls()
        session_analytics.session_id = name

        session_analytics.stream_data_dictionary = {}
        session_analytics.stream_information = pd.DataFrame(columns=[
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

            unique_stream_name = get_unique_stream_name(stream_name, session_analytics.stream_data_dictionary)

            session_analytics.stream_data_dictionary[unique_stream_name] = stream_content_dataframe
            session_analytics.stream_information.loc[len(session_analytics.stream_information)] = [
                stream_info["stream_id"],
                unique_stream_name,
                stream_info["type"][0],
                channel_labels,
                stream_info["channel_count"][0],
                stream_info["channel_format"][0],
                stream_info["nominal_srate"][0],
                stream_info["effective_srate"],
                len(stream_content_dataframe)]

        session_analytics.session_id = name
        session_analytics.stream_count = len(xdf_data)
        session_analytics.stream_data_dictionary = dict(sorted(session_analytics.stream_data_dictionary.items()))
        session_analytics.stream_information.sort_values(by=["Stream Name"], inplace=True, ignore_index=True)

        return session_analytics
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
        clean = re.sub(r'[^\w\s.-]', '_', stream).strip()
        filename = f"Stream Data - {clean}"
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