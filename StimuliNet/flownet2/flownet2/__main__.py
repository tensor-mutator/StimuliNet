from argparse import ArgumentParser as ap
from argparse import Namespace
from flow_dataset.mpi_sintel import load
import sys
import os
import re
from .flownet_2 import FlowNet2
from .FlowNetCSS import flownet_css
from .FlowNetSD import flownet_sd
from .pipeline import Pipeline
from .config import config

weights_path = os.path.split(__file__)[0]

def parser() -> ap:
    parser = ap(add_help=True)
    parser.add_argument("-graph", "--get-graph", help="dumps an event file with the FlowNet2.0 graph", required=False, action="store_true")
    parser.add_argument("-train", "--train", help="starts training the FlowNet2.0 block with the designated resolution", required=False)
    return parser

def main(args: Namespace) -> None:
    if args.train:
       frozen_config = [dict(scope="FlowNetCSS", path=os.path.join(os.path.split(flownet_css.__file__)[0], "weights")),
                        dict(scope="FlowNetSD", path=os.path.join(os.path.split(flownet_sd.__file__)[0], "weights"),
                             patch=[dict(op="conv2d", val=37), dict(op="conv2d_transpose", val=24)])]
       resolution = tuple(list(map(lambda x: int(x), re.findall(r'[0-9]{1,}', args.train))))
       pipeline = Pipeline(FlowNet2, "LONG_SCHEDULE", resolution, resolution, checkpoint_path=weights_path, frozen_config=frozen_config,
                           config=config.LOSS_EVENT+config.SAVE_FLOW)
       X_src_train, X_src_test, X_dest_train, X_dest_test, y_train, y_test = load(resolution, resolution)
       pipeline.fit(X_src_train, X_src_test, X_dest_train, X_dest_test, y_train, y_test)
    if args.get_graph:
       pipeline = Pipeline(FlowNet2, "LONG_SCHEDULE", (512, 512), (512, 512), checkpoint_path=weights_path, config=config.LOSS_EVENT+config.SAVE_FLOW)
       pipeline.save_graph(weights_path)

main(parser().parse_args(sys.argv[1:]))
