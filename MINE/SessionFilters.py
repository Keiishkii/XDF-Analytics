from __future__ import annotations
from typing import TYPE_CHECKING
from MINE.Log import Log

import pandas as pd
import copy

if TYPE_CHECKING: from MINE.Analysis import ExperimentAnalytics


#region [ Class ][ ISessionFilter ]
class ISessionFilter:
    def __init__(self): pass
    def evaluate(self, experiment_analytics: ExperimentAnalytics) -> ExperimentAnalytics:
        """
        :param experiment_analytics: The MultiFileAnalytics object to be filtered.
        :return: Returns a new Analytics object containing the now filtered dataframes.
        """

        return experiment_analytics
#endregion

class ContainsMarkersSessionFilter(ISessionFilter):
    def __init__(self, marker_stream: str, marker_list: str | list[str]):
        super().__init__()
        self.marker_stream_name = marker_stream
        self.marker_list = marker_list if isinstance(marker_list, list) else [marker_list]

    def evaluate(self, experiment_analytics: ExperimentAnalytics) -> ExperimentAnalytics:
        """
        :param experiment_analytics: The MultiFileAnalytics object to be filtered.
        :return: Returns a new Analytics object containing the now filtered dataframes.
        """

        filtered_experiment_analytics = copy.deepcopy(experiment_analytics)
        filtered_experiment_analytics.session_list.clear()

        for session_analytics in experiment_analytics.session_list:
            if self.marker_stream_name not in session_analytics.stream_data_dictionary:
                Log.message(f"Skipping session '{session_analytics.session_id}' as it does not contain the stream '{self.marker_stream_name}'.")
                continue

            marker_stream = session_analytics.stream_data_dictionary[self.marker_stream_name]
            if marker_stream.empty:
                Log.message(f"Skipping session '{session_analytics.session_id}' as the marker stream is empty.")
                continue

            failed = False
            for marker in self.marker_list:
                if marker in marker_stream["Value"].str[0].values: continue

                Log.message(f"Skipping session '{session_analytics.session_id}' as it does not contain marker '{marker}'.")
                failed = True
                break

            if failed:
                Log.message(f"{session_analytics.session_id} failed for containing a marker stream.")
                continue

            filtered_experiment_analytics.session_list.append(session_analytics)
            Log.message(f"{session_analytics.session_id} evaluated true for containing a marker stream.")

        return filtered_experiment_analytics



class ContainsStreamSessionFilter(ISessionFilter):
    def __init__(self, stream_list: str | list[str]):
        super().__init__()
        self.stream_list = stream_list if isinstance(stream_list, list) else [stream_list]

    def evaluate(self, experiment_analytics: ExperimentAnalytics) -> ExperimentAnalytics:
        """
        :param experiment_analytics: The MultiFileAnalytics object to be filtered.
        :return: Returns a new Analytics object containing the now filtered dataframes.
        """

        filtered_experiment_analytics = copy.deepcopy(experiment_analytics)
        filtered_experiment_analytics.session_list.clear()


        Log.message(f"Filtering sessions containing streams: {self.stream_list}")
        for session_analytics in experiment_analytics.session_list:
            stream_names = session_analytics.stream_data_dictionary.keys()

            failed = False
            for stream in self.stream_list:
                if stream in stream_names: continue

                Log.message(f"Skipping session '{session_analytics.session_id}' as it does not contain stream '{stream}'.")
                failed = True
                break

            if failed:
                Log.message(f"{session_analytics.session_id} for containing a required data streams.")
                continue

            filtered_experiment_analytics.session_list.append(session_analytics)
            Log.message(f"{session_analytics.session_id} evaluated true for containing a required data streams.")

        return filtered_experiment_analytics
#endregion