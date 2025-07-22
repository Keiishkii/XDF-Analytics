import pandas as pd

import Analysis.Log as log

def generate_subfigure_dataframe(dataframe_dictionary: dict[str, pd.DataFrame] | list[dict[str, pd.DataFrame]], layout_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    :param dataframe_dictionary: The dictionary of dataframes to be plotted, each representing a stream from the xdf file. This can also be passed as a list of dictionaries to render all on the same figure.
    :param layout_dataframe: The layout dataframe containing the information on how to arrange the subfigures.
    :return: A dataframe containing the information on how to arrange the subfigures.
    """

    def is_dictionary_valid(dataframe_dictionary: dict[str, pd.DataFrame], stream: str, channel: int) -> bool:
        if s

    subfigure_dataframe = pd.DataFrame(columns = ["Label", "Stream", "Channel Index", "Axis Limits", "Valid Dataframes"])

    for _, row in layout_dataframe.iterrows():
        stream = row["Stream"]
        channels = row["Channels"]
        axis_limits = row["Axis Limits"]

        for channel in channels:
            label = f"{row['Subfigure Prefix']}: {channel}"

            dataframe_dictionary_list = dataframe_dictionary if isinstance(dataframe_dictionary, list) else [dataframe_dictionary]
            valid_dataframes = [stream in dictionary for dictionary in dataframe_dictionary_list]


            subfigure_dataframe.loc[len(subfigure_dataframe)] = [
                label,
                stream,
                channel,
                axis_limits

            ]


def generate_subfigure_dataframe(dataframe_dictionary: dict, subfigure_layout_dataframe: pd.DataFrame):

    for _, row in subfigure_layout_dataframe.iterrows():
        _stream = row["Stream"]

        if not is_stream_in_dataframe(dataframe_dictionary, _stream): continue
        if not is_stream_null_or_empty(dataframe_dictionary, _stream): continue

        _shape = np.array(dataframe_dictionary[_stream]["Value"].tolist()).shape
        _channel_count = _shape[1]
        _channels = list(range(_channel_count)) if ("Channels" not in row or row["Channels"] is None) else row["Channels"]

        for _channel_index in _channels:
            if not is_valid_channel(_stream, _channel_index, _channel_count): continue

            _subfigure_label = f"{row['Subfigure Prefix']}: {_channel_index}"
            _subfigure_dataframe.loc[len(_subfigure_dataframe)] = [_subfigure_label, _stream, _channel_index, row["Axis Limits"]]

    return _subfigure_dataframe



def generate_figure(dataframe_dictionary: dict[str, pd.DataFrame] | list[dict[str, pd.DataFrame]], layout_dataframe: pd.DataFrame):
    """
    :param dataframe_dictionary: The dictionary of dataframes to be plotted, each representing a stream from the xdf file. This can also be passed as a list of dictionaries to render all on the same figure.
    :param layout_dataframe: The layout dataframe containing the information on how to arrange the subfigures.
    """

    subfigure_dataframe = generate_subfigure_dataframe(global_stream_dataframe_dictionary, layout_dataframe)
    subfigure_count = len(_subfigure_dataframe)

    print(_subfigure_count)

    _figure, _axes = plt.subplots(nrows=_subfigure_count, ncols=1, figsize = (30, _subfigure_count * 3))

    #[ Plot Data ]
    #plot_data(session_subset_dataframe_dictionary, _axes, _subfigure_dataframe, [0.5, 0.5, 0.5, 0.5])
    #plot_data(block_a_subset_dataframe_dictionary, _axes, _subfigure_dataframe, "orange")
    #plot_data(block_b_subset_dataframe_dictionary, _axes, _subfigure_dataframe, "orange")
    #plot_data(block_c_subset_dataframe_dictionary, _axes, _subfigure_dataframe, "orange")

    #plot_markers(session_subset_dataframe_dictionary["UESA_Markers"], _axes)

    plot_data(block_a_subset_dataframe_dictionary, _axes, _subfigure_dataframe, "blue")
    plot_markers(block_a_subset_dataframe_dictionary["UESA_Markers"], _axes)

    resize_axis_for_equal_bounds(_axes)
    plt.tight_layout()

    data_exports.export_figure(_figure, "/home/keiishkii/Git/MINE/DamlaDataAnalysis/Exports", "Figure - Global Data.png")
    print("Exported...")

    display(_figure)
    print("Rendered...")

    plt.close(_figure)
    print("Finished...")

generate_global_data_figure()