import os

# 禁止numpy使用多进程
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

from xhlib.exp import ServerFileStorageObserver, ex
from xhlib.logger import Logger
from xhlib.utils import init_object, load_module, run_method


@ex.config
def config():
    seed = 0
    algo_name = "CFR"
    game_name = "KuhnPoker"
    log_folder = "logs"
    group = "default"

    # logger
    writer_strings = ["stdout"]
    search_hyper = False
    save_log = False
    if save_log:
        folder = Path(__file__).parents[1] / log_folder / group / algo_name / game_name
        writer_strings += ["csv", "sacred", "tensorboard"]
        ex.observers.append(ServerFileStorageObserver(folder))


@ex.automain
def main(algo_name, search_hyper, _config, _run):
    configs = dict(_config)
    if configs["save_log"]:
        configs["folder"] = configs["folder"] / str(_run._id)
    logger = init_object(Logger, configs)
    solver_class = load_module("xdcfr:{}".format(algo_name))

    if search_hyper:
        run_method(solver_class.search_hyper, configs)
    else:
        solver = init_object(solver_class, configs, logger=logger)
        solver.solve()
