#region [ Imports ]
from __future__ import annotations
from typing import Any
from enum import Enum
from IPython.core.display_functions import display
from typing import TYPE_CHECKING
from MINE.Log import Log
from MINE.Objects.SessionAnalytics import SessionAnalytics
from MINE.Objects.ExperimentProperties import ParticipantProperties, StreamProperties
from MINE.Objects.ExperimentProperties import StreamProperties
from dataclasses import dataclass

if TYPE_CHECKING: from MINE.StreamFilter import IStreamFilter, TimestampStreamFilter
if TYPE_CHECKING: from MINE.SessionFilters import ISessionFilter, ContainsStreamSessionFilter

import pandas as pd
import numpy as np
#endregion

#region [ Enum ][ Export Method ]
class ExportMethod(Enum):
    CSV = 1
    XLSX = 2
#endregion



#region [ Class ][ ExperimentAnalytics ]
@dataclass
class ExperimentAnalytics:
    """
    The global analysis class for an entire experiment.
    Used to load in several session of XDF files and process them concurrently.
    """

    #region [ Properties ]
    def __init__(self):
        #self.analytics_dataframe: pd.DataFrame | None = None
        self.session_list: list[SessionAnalytics] = []
        self.file_count: int = 0
    #endregion



    #region [ Initialisation ]
    @classmethod
    def create_from_paths(cls, participant_data: list[ParticipantProperties]) -> ExperimentAnalytics:
        """
        :param paths: A data frame representing the participant data, with columns "Participant_ID" and "File_Path" containing the participant ID and path to the xdf file, respectively..
        :return: A Multi-file MINE object containing the organised information for each of the xdf files.
        """

        experiment_analytics = cls()

        #experiment_analytics.analytics_dataframe = pd.DataFrame(columns=[
        #    "Session_ID",
        #    "Analysis_Object",
        #    "Streams",
        #    "Stream_Count",
        #    "Valid"
        #])

        for participant in participant_data:
            participant_id = participant.id
            path = participant.file_path

            Log.message(f"Processing: {participant_id}")

            session_analytics = SessionAnalytics.create_from_path(path, f"{participant_id}")
            experiment_analytics.session_list.append(session_analytics)

        experiment_analytics.file_count = len(experiment_analytics.session_list)
        return experiment_analytics
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

        filtered_session_analytics.file_count = len(filtered_session_analytics.session_list)
        return filtered_session_analytics
    #endregion

    def remove_all_provided_streams(self, stream_list: list[str]):
        # Remove unnecessary streams from the stream lists.
        for session_analytics in self.session_list:
            for key in list(session_analytics.stream_data_dictionary.keys()):
                if key in stream_list:
                    del session_analytics.stream_data_dictionary[key]
    def remove_all_but_provided_streams(self, stream_list: list[str]):
        # Remove unnecessary streams from the stream lists.
        for session_analytics in self.session_list:
            for key in list(session_analytics.stream_data_dictionary.keys()):
                if key not in stream_list:
                    del session_analytics.stream_data_dictionary[key]

    def convert_streams_to_expected_datatype(self, stream_properties: StreamProperties):
        for session_analytics in self.session_list:
            for stream_property in list(stream_properties):
                stream_name: str = stream_property.name
                if stream_property.type == "float":
                    session_analytics.stream_data_dictionary[stream_name]["Value"] = np.concatenate(session_analytics.stream_data_dictionary[stream_name]["Value"].to_numpy())

#endregion