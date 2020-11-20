# A <i>FlowNet2.0</i> library
A <i>FlowNet2.0</i> implementation with TensorFlow

#### Installation steps
```bash
(flownet2) poetry install
```

#### Intitiating training of <i>FlowNetC</i> with performance visualization
```bash
(flownet2) poetry run python3.6 -m flownet2.FlowNetC --train "(64, 64)"
(flownet2) tensorboard --logdir flownet2/FlowNetC/weights
```

#### Intitiating training of <i>FlowNetCS</i> with performance visualization
```bash
(flownet2) poetry run python3.6 -m flownet2.FlowNetCS --train "(64, 64)"
(flownet2) tensorboard --logdir flownet2/FlowNetCS/weights
```

#### Intitiating training of <i>FlowNetCSS</i> with performance visualization
```bash
(flownet2) poetry run python3.6 -m flownet2.FlowNetCSS --train "(64, 64)"
(flownet2) tensorboard --logdir flownet2/FlowNetCSS/weights
```

#### Intitiating training of <i>FlowNetSD</i> with performance visualization
```bash
(flownet2) poetry run python3.6 -m flownet2.FlowNetSD --train "(64, 64)"
(flownet2) tensorboard --logdir flownet2/FlowNetSD/weights
```

#### Intitiating training of <i>FlowNet2.0</i> with performance visualization
```bash
(flownet2) poetry run python3.6 -m flownet2 --train "(64, 64)"
(flownet2) tensorboard --logdir flownet2/weights
```
