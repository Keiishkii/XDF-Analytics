import json
from dataclasses import dataclass
from typing import Type, TypeVar, Any, Dict

T = TypeVar("T")

from typing import get_origin, get_args

class JSON:
    @staticmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        kwargs = {}

        for field, field_type in field_types.items():
            value = data.get(field)

            # Nested dataclass
            if hasattr(field_type, "__dataclass_fields__"):
                kwargs[field] = JSON.from_dict(field_type, value)

            # List[T]
            elif get_origin(field_type) is list:
                (item_type,) = get_args(field_type)

                new_list = []
                for item in value:
                    if hasattr(item_type, "__dataclass_fields__"):
                        new_list.append(JSON.from_dict(item_type, item))
                    else:
                        new_list.append(item_type(item))
                kwargs[field] = new_list

            # Primitive
            else:
                kwargs[field] = value

        return cls(**kwargs)


    @staticmethod
    def load_from_json(path: str, cls: Type[T]) -> T:
        with open(path, "r") as f:
            data = json.load(f)
        return JSON.from_dict(cls, data)

