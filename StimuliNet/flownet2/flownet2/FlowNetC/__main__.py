from argparse import ArgumentParser as ap
from argparse import Namespace
from flow_dataset.mpi_sintel import load
import sys
import os
from .flownet_c import FlowNetC
from ..pipeline import Pipeline
from ..config import config

weights_path = os.path.split(__file__)[0]

def parser() -> ap:
    parser = ap(add_help=True)
    parser.add_argument("-graph", "--get-graph", help="dumps an event file with the current graph", required=False, action="store_true")
    parser.add_argument("-train", "--train", help="starts training the current block with the designated resolution", required=False,
                        action="store_true")
    return parser

def main(args: Namespace):
    if args.train:
       resolution = tuple(args.train)
       pipeline = Pipeline(FlowNetC, "DEFAULT", resolution, resolution, checkpoint_path=weights_path, config=config.LOSS_EVENT+config.SAVE_FLOW)
       X_src_train, X_src_test, X_dest_train, X_dest_test, y_train, y_test = load(resolution, resolution)
       pipeline.fit(X_src_train, X_src_test, X_dest_train, X_dest_test, y_rain, y_test)
    if args.get_graph:
       pipeline = Pipeline(FlowNetC, "DEFAULT", (512, 512), (512, 512), checkpoint_path=weights_path, config=config.LOSS_EVENT+config.SAVE_FLOW)
       pipeline.save_graph()

main(parser().parse_args(sys.argv[1:]))
