"""Iterator over serialized Timecourses stored in a zip archive."""

import src.constants as cn  # type: ignore
from src.timecourse import Timecourse  # type: ignore

import pickle
import zipfile
from typing import Iterator


class TimecourseIteratorItem:
    """A model name paired with its deserialized Timecourse."""

    def __init__(self, model_name: str, timecourse: Timecourse) -> None:
        self.model_name = model_name
        self.timecourse = timecourse


class TimecourseIterator:
    """Iterates over serialized Timecourses in the timecourse zip archive."""

    def __init__(self, zip_path: str = cn.TIMECOURSE_ZIP_PATH) -> None:
        self.zip_path = zip_path

    def __iter__(self) -> Iterator[TimecourseIteratorItem]:
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            for name in sorted(zf.namelist()):
                if not name.endswith('_timecourse.pkl'):
                    continue
                model_name = name[: -len('_timecourse.pkl')]
                with zf.open(name) as entry_f:
                    dct = pickle.load(entry_f)
                timecourse = Timecourse(
                    model=dct['model'],
                    start_time=dct['start_time'],
                    end_time=dct['end_time'],
                    num_point=dct['num_point'],
                    timecourse_df=dct['timecourse_df'],
                    jacobian_collection_arr=dct['jacobian_collection_arr'],
                )
                yield TimecourseIteratorItem(model_name=model_name, timecourse=timecourse)
