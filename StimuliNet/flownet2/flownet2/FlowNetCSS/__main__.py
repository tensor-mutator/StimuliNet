from argparse import ArgumentParser as ap
from argparse import Namespace
from flow_dataset.mpi_sintel import load
import sys
import os
import re
from .flownet_css import FlowNetCSS
from ..FlowNetCS import flownet_cs
from ..pipeline import Pipeline
from ..config import config

weights_path = os.path.split(__file__)[0]

def parser() -> ap:
    parser = ap(add_help=True)
    parser.add_argument("-graph", "--get-graph", help="dumps an event file with the FlowNetCSS graph", required=False, action="store_true")
    parser.add_argument("-train", "--train", help="starts training the FlowNetCSS block with the designated resolution", required=False)
    return parser

def main(args: Namespace) -> None:
    if args.train:
       frozen_config = [dict(scope="FlowNetCS", path=os.path.join(os.path.split(flownet_cs.__file__)[0], "weights"))]
       resolution = tuple(list(map(lambda x: int(x), re.findall(r'[0-9]{1,}', args.train))))
       pipeline = Pipeline(FlowNetCSS, "LONG_SCHEDULE", resolution, resolution, checkpoint_path=weights_path, frozen_config=frozen_config,
                           config=config.LOSS_EVENT+config.SAVE_FLOW)
       X_src_train, X_src_test, X_dest_train, X_dest_test, y_train, y_test = load(resolution, resolution)
       pipeline.fit(X_src_train, X_src_test, X_dest_train, X_dest_test, y_train, y_test)
    if args.get_graph:
       pipeline = Pipeline(FlowNetCSS, "LONG_SCHEDULE", (512, 512), (512, 512), checkpoint_path=weights_path, config=config.LOSS_EVENT+config.SAVE_FLOW)
       pipeline.save_graph(weights_path)

main(parser().parse_args(sys.argv[1:]))
