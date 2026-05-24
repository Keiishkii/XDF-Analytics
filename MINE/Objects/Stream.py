import pandas as pd

class Stream:
    stream_id: str | None = None
    stream_data: pd.DataFrame | None = None