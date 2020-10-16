from argparse import ArgumentParser as ap
from argparse import NameSpace
from flow_dataset.mpi_sintel import load
import sys
from .flownet_c import FlowNetC
from ..pipeline import Pipeline
from ..config import config

def parser() -> ap:
    parser = ap(add_help=True)
    parser.add_argument("-graph", "--get-graph", help="dumps an event file with the current graph", required=False)
    parser.add_argument("-train", "--train", help="starts training the current block", required=False, action="store_true")
    return parser

def main(args: NameSpace):
    if args.train:
       pipeline = Pipeline(FlowNetC, "DEFAULT", (512, 512), (512, 512), config=config.LOSS_EVENT+config.SAVE_FLOW)
       X_train, X_test, y_train, y_test = load((512, 512), (512, 512))
       pipeline.fit(X_train, X_test, y_train, y_test)
    if args.get_graph:
       pipeline = Pipeline(FlowNetC, "DEFAULT", (512, 512), (512, 512), config=config.LOSS_EVENT+config.SAVE_FLOW)
       pipeline.save_graph()

main(parser().parse_args(sys.argv[1:]))
