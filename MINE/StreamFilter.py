from __future__ import annotations
from typing import TYPE_CHECKING

import copy

if TYPE_CHECKING: from MINE.Analysis import SessionAnalytics


#region [ Class ][ IStreamFilter ]
class IStreamFilter:
    def __init__(self): pass
    def evaluate(self, analytics: SessionAnalytics) -> SessionAnalytics: return analytics
#endregion

#region [ Class ][ TimestampStreamFilter ]
class TimestampStreamFilter(IStreamFilter):
    def __init__(self, start_timestamp: float, end_timestamp: float):
        super().__init__()
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp

    def evaluate(self, analytics: SessionAnalytics) -> SessionAnalytics:
        """
        :param analytics: The Analytics object to be filtered.
        :return: Returns a new Analytics object containing the now filtered dataframes.
        """

        filtered_analytics = copy.deepcopy(analytics)

        #region [ Setup ]
        stream_data_dictionary = filtered_analytics.stream_data_dictionary
        stream_information = filtered_analytics.stream_information
        #endregion [ Setup ]


        for stream_name, stream_dataframe in filtered_analytics.stream_data_dictionary.items():
            if stream_dataframe.empty: continue

            stream_dataframe = stream_dataframe[self.start_timestamp <= stream_dataframe["Timestamp"]]
            stream_dataframe = stream_dataframe[stream_dataframe["Timestamp"] <= self.end_timestamp]

            stream_data_dictionary[stream_name] = stream_dataframe
            stream_information.loc[stream_information["Stream Name"] == stream_name, "Samples"] = len(stream_dataframe)


        #region [ End ]
        filtered_analytics.stream_data_dictionary = stream_data_dictionary
        filtered_analytics.stream_information = stream_information
        #endregion [ End ]

        return filtered_analytics
#endregion