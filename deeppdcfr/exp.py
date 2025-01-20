import logging
import os
from shutil import SameFileError, copyfile

from sacred import Experiment
from sacred.observers import FileStorageObserver
import yaml
from deeppdcfr.utils import get_server_id

ex = Experiment("default")
logger = logging.getLogger("mylogger")
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s [%(levelname).1s] %(filename)s:%(lineno)d - %(message)s ",
    "%Y-%m-%d %H:%M:%S",
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

ex.logger = logger


class ServerFileStorageObserver(FileStorageObserver):
    server_id = get_server_id()

    def _maximum_existing_run_id(self):
        dir_nrs = []
        for folder in os.listdir(self.basedir):
            if "-" in folder:
                a, b = folder.split("-")
                if int(a) == "server_id":
                    b = int(b)
                    dir_nrs.append(b)
        if dir_nrs:
            return max(dir_nrs)
        else:
            return 0

    def _make_run_dir(self, _id):
        os.makedirs(self.basedir, exist_ok=True)
        self.dir = None
        if _id is None:
            fail_count = 0
            _id = self._maximum_existing_run_id() + 1
            while self.dir is None:
                try:
                    self._make_dir(_id)
                except FileExistsError:  # Catch race conditions
                    if fail_count < 1000:
                        fail_count += 1
                        _id += 1
                    else:  # expect that something else went wrong
                        raise
        else:
            self.dir = os.path.join(self.basedir, "{}-{}".format(self.server_id, _id))
            os.mkdir(self.dir)

    def _make_dir(self, _id):
        new_dir = os.path.join(self.basedir, "{}-{}".format(self.server_id, _id))
        os.mkdir(new_dir)
        self.dir = new_dir  # set only if mkdir is successful

    def save_file(self, filename, target_name=None):
        target_name = target_name or os.path.basename(filename)
        blacklist = ["run.json", "config.json", "stdout.log", "metrics.json"]
        blacklist = [os.path.join(self.dir, x) for x in blacklist]
        dest_file = os.path.join(self.dir, target_name)
        if dest_file in blacklist:
            raise FileExistsError(
                "You are trying to overwrite a file necessary for the "
                "FileStorageObserver. "
                "The list of blacklisted files is: {}".format(blacklist)
            )
        try:
            copyfile(filename, dest_file)
        except SameFileError:
            pass

    def save_cout(self):
        with open(os.path.join(self.dir, "stdout.log"), "ab") as f:
            f.write(self.cout[self.cout_write_cursor :].encode("utf-8"))
            self.cout_write_cursor = len(self.cout)

    def started_event(
        self, ex_info, command, host_info, start_time, config, meta_info, _id
    ):
        self._make_run_dir(_id)

        ex_info["sources"] = self.save_sources(ex_info)

        self.run_entry = {
            "experiment": dict(ex_info),
            "command": command,
            "host": dict(host_info),
            "start_time": start_time.isoformat(),
            "meta": meta_info,
            "status": "RUNNING",
            "resources": [],
            "artifacts": [],
            "heartbeat": None,
        }
        self.config = config
        self.info = {}
        self.cout = ""
        self.cout_write_cursor = 0

        self.save_json(self.run_entry, "run.json")
        self.save_json(self.config, "config.json")
        self.save_yaml(self.config, "config.yaml")
        self.save_cout()

        return os.path.relpath(self.dir, self.basedir) if _id is None else _id

    def queued_event(
        self, ex_info, command, host_info, queue_time, config, meta_info, _id
    ):
        self._make_run_dir(_id)

        self.run_entry = {
            "experiment": dict(ex_info),
            "command": command,
            "host": dict(host_info),
            "meta": meta_info,
            "status": "QUEUED",
        }
        self.config = config
        self.info = {}

        self.save_json(self.run_entry, "run.json")
        self.save_json(self.config, "config.json")
        self.save_yaml(self.config, "config.yaml")

        if self.copy_sources:
            for s, _ in ex_info["sources"]:
                self.save_file(s)

        return os.path.relpath(self.dir, self.basedir) if _id is None else _id

    def save_yaml(self, obj, filename):
        if isinstance(obj, dict):
            del_keywords = ["folder", "search_hyper", "writer_strings"]
            for key in del_keywords:
                if key in obj:
                    del obj[key]
        with open(os.path.join(self.dir, filename), "w") as f:
            yaml.dump(obj, f)
            # json.dump(flatten(obj), f, sort_keys=True, indent=2)
            f.flush()
