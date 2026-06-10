# RoboManoLayer

Build robot-friendly MANO hand models for PyTorch and MuJoCo.

- `ManoLayer` compatibility without the `chumpy` dependency.
- Beta-specific MJCF export for robotics simulation.
- Conversion between MANO poses, MJCF `qpos`, link poses, vertices, and joints.

## Installation

```bash
uv add git+https://github.com/JYChen18/manolayer.git
```

Download the official MANO model files from the
[MANO website](https://mano.is.tue.mpg.de/) and place them under:

```text
~/.manolayer/assets/mano/models/MANO_RIGHT.pkl
~/.manolayer/assets/mano/models/MANO_LEFT.pkl
```

You can also pass a custom `mano_assets_root` when constructing a layer.

## Usage

### ManoLayer Usage

`ManoLayer` follows [lixiny/manotorch](https://github.com/lixiny/manotorch).

```python
from manolayer import ManoLayer

mano_layer = ManoLayer(side="right", center_idx=0)
output = mano_layer(pose_coeffs, betas=betas)
```

### RoboManoLayer Export

`RoboManoLayer` supports exporting MJCF models for robotics simulation.

```python
from pathlib import Path

import torch
from manolayer import RoboManoLayer

robo_layer = RoboManoLayer(side="right", betas=torch.zeros(10))
saved_folder = robo_layer.export_xml(Path("exports/robomano"))
```

`export_xml` writes to `exports/robomano/right/beta_xxxxxxxxxx` and returns that
folder. The folder contains `betas.txt`, `right.xml`, `right_ball.xml`, and
`meshes/`. If the folder already exists, export is skipped; `betas.txt` is read
to warn when the saved beta differs from the current beta by more than `1e-5`.

### RoboManoLayer Usage

`RoboManoLayer` supports staged use: convert MANO poses to MJCF `qpos`, run
forward kinematics, then recover vertices and joints.

```python
import torch
from manolayer import RoboManoLayer

robo_layer = RoboManoLayer(side="right", betas=torch.zeros(10))
qpos = robo_layer.pose_to_qpos(pose_coeffs, center_idx=0)
link_poses = robo_layer.forward_kinematics(qpos) # In simulation, use link poses from the simulator here.
output = robo_layer.get_verts_joints(link_poses)
```

## Acknowledgements

- [MANO](https://mano.is.tue.mpg.de/)
- [manotorch](https://github.com/lixiny/manotorch)
- [manopth](https://github.com/hassony2/manopth)
- [smplx](https://github.com/vchoutas/smplx)
