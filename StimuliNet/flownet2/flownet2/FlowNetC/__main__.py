from .flownet_c import FlowNetC
from ..pipeline import Pipeline
from ..config import config
from flow_dataset.mpi_sintel import load

def main():
    pipeline = Pipeline(FlowNetC, "DEFAULT", (512, 512), (512, 512), config=config.LOSS_EVENT+config.SAVE_FLOW)
    X_train, X_test, y_train, y_test = load((512, 512), (512, 512))
    pipeline.fit(X_train, X_test, y_train, y_test)

main()
