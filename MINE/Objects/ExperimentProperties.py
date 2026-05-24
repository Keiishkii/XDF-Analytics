from dataclasses import dataclass

@dataclass
class ParticipantProperties:
    id: str
    file_path: str

@dataclass
class StreamProperties:
    name: str
    type: str

@dataclass
class ExperimentProperties:
    required_markers: list[str]
    required_streams: list[StreamProperties]
    participant_files: list[ParticipantProperties]