from __future__ import annotations
from typing import TYPE_CHECKING
from MINE.Log import Log

import pandas as pd
import copy

if TYPE_CHECKING: from MINE.Analysis import ExperimentAnalytics


#region [ Class ][ ISessionFilter ]
class ISessionFilter:
    def __init__(self): pass
    def evaluate(self, multi_file_analytics: ExperimentAnalytics) -> ExperimentAnalytics:
        """
        :param multi_file_analytics: The MultiFileAnalytics object to be filtered.
        :return: Returns a new Analytics object containing the now filtered dataframes.
        """

        return multi_file_analytics
#endregion

class ContainsMarkersSessionFilter(ISessionFilter):
    def __init__(self, marker_stream: str, marker_list: str | list[str]):
        super().__init__()
        self.marker_stream_name = marker_stream
        self.marker_list = marker_list if isinstance(marker_list, list) else [marker_list]

    def evaluate(self, multi_file_analytics: ExperimentAnalytics) -> ExperimentAnalytics:
        """
        :param multi_file_analytics: The MultiFileAnalytics object to be filtered.
        :return: Returns a new Analytics object containing the now filtered dataframes.
        """
        filtered_multi_file_analytics = copy.deepcopy(multi_file_analytics)

        #region [ Setup ]
        analytics_dataframe = pd.DataFrame(columns=filtered_multi_file_analytics.analytics_dataframe.columns)
        #endregion [ Setup ]


        for _, session_row in filtered_multi_file_analytics.analytics_dataframe.iterrows():
            if self.marker_stream_name not in session_row["Analysis_Object"].stream_data_dictionary:
                Log.message(f"Skipping session '{session_row['Session_ID']}' as it does not contain the stream '{self.marker_stream_name}'.")
                continue

            marker_stream = session_row["Analysis_Object"].stream_data_dictionary[self.marker_stream_name]
            if marker_stream.empty:
                Log.message(f"Skipping session '{session_row['Session_ID']}' as the marker stream is empty.")
                continue

            failed = False
            for marker in self.marker_list:
                if marker in marker_stream["Value"].str[0].values: continue

                Log.message(f"Skipping session '{session_row['Session_ID']}' as it does not contain marker '{marker}'.")
                failed = True
                break

            if failed: continue

            analytics_dataframe.loc[len(analytics_dataframe)] = session_row


        #region [ End ]
        filtered_multi_file_analytics.analytics_dataframe = analytics_dataframe
        #endregion [ End ]

        return filtered_multi_file_analytics



class ContainsStreamSessionFilter(ISessionFilter):
    def __init__(self, stream_list: str | list[str]):
        super().__init__()
        self.stream_list = stream_list if isinstance(stream_list, list) else [stream_list]

    def evaluate(self, multi_file_analytics: ExperimentAnalytics) -> ExperimentAnalytics:
        """
        :param multi_file_analytics: The MultiFileAnalytics object to be filtered.
        :return: Returns a new Analytics object containing the now filtered dataframes.
        """

        filtered_multi_file_analytics = copy.deepcopy(multi_file_analytics)

        #region [ Setup ]
        analytics_dataframe = pd.DataFrame(columns=filtered_multi_file_analytics.analytics_dataframe.columns)
        #endregion [ Setup ]


        Log.message(f"Filtering sessions containing streams: {self.stream_list}")
        for _, session_row in filtered_multi_file_analytics.analytics_dataframe.iterrows():
            sessions_stream_list = session_row["Analysis_Object"].stream_data_dictionary.keys()

            failed = False
            for stream in self.stream_list:
                if stream in sessions_stream_list: continue

                Log.message(f"Skipping session '{session_row['Session_ID']}' as it does not contain stream '{stream}'.")
                failed = True
                break

            if failed: continue

            analytics_dataframe.loc[len(analytics_dataframe)] = session_row


        #region [ End ]
        filtered_multi_file_analytics.analytics_dataframe = analytics_dataframe
        #endregion [ End ]

        return filtered_multi_file_analytics
#endregion