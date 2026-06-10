from contextlib import contextmanager
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

from loguru import logger
import roma
import torch

from manolayer import RoboManoLayer
from manolayer.robomano_layer import _beta_tag
from manolayer.robomano_utils import mano_xml_frame_matrix


def _stl_triangle_count(path: Path):
    with path.open("rb") as stl_file:
        stl_file.seek(80)
        return struct.unpack("<I", stl_file.read(4))[0]


def _vec(text: str):
    return torch.tensor([float(value) for value in text.split()])


def _assert_same_rotation(actual: str, expected: str):
    actual_matrix = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(_vec(actual)))
    expected_matrix = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(_vec(expected)))
    assert torch.allclose(actual_matrix, expected_matrix, atol=3e-6, rtol=3e-6)


def _assert_rotation_matrix(actual: str, expected: torch.Tensor):
    actual_matrix = roma.unitquat_to_rotmat(roma.quat_wxyz_to_xyzw(_vec(actual)))
    assert torch.allclose(actual_matrix, expected, atol=3e-6, rtol=3e-6)


@contextmanager
def _capture_loguru_warnings():
    messages = []
    handler_id = logger.add(
        lambda message: messages.append(str(message)),
        level="WARNING",
        format="{message}",
    )
    try:
        yield messages
    finally:
        logger.remove(handler_id)


def test_robomano_export_xml_writes_meshes_and_both_joint_modes(
    mano_assets_root,
    tmp_path,
):
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.zeros(10),
    )

    saved_folder = layer.export_xml(tmp_path)
    mesh_folder = saved_folder / "meshes"
    reduced_xml = saved_folder / "right.xml"
    ball_xml = saved_folder / "right_ball.xml"

    assert saved_folder.parent == tmp_path / "right"
    assert saved_folder.name.startswith("beta_")
    assert len(saved_folder.name) == len("beta_") + 10

    reduced = ET.parse(reduced_xml).getroot()
    ball = ET.parse(ball_xml).getroot()
    mesh_files = sorted(mesh_folder.glob("*.stl"))

    assert reduced.attrib["model"] == "mano_right"
    assert ball.attrib["model"] == "mano_right_ball"
    assert len(mesh_files) == 16
    assert all(_stl_triangle_count(path) > 0 for path in mesh_files)
    assert len(reduced.findall("./actuator/position")) == 20
    assert len(ball.findall("./actuator/position")) == 45
    assert len(reduced.findall("./worldbody//joint")) == 20
    assert len(ball.findall("./worldbody//joint")) == 15


def test_robomano_export_xml_writes_beta_txt(
    mano_assets_root,
    tmp_path,
):
    betas = torch.linspace(-0.02, 0.02, 10)
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=betas,
    )

    saved_folder = layer.export_xml(tmp_path)
    saved_betas = torch.tensor(
        [float(value) for value in (saved_folder / "betas.txt").read_text().split()]
    )

    assert torch.allclose(saved_betas, betas, atol=0.0, rtol=0.0)


def test_robomano_export_xml_existing_folder_skips_saving(
    mano_assets_root,
    tmp_path,
):
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.zeros(10),
    )
    saved_folder = tmp_path / "right" / _beta_tag(layer._shape_betas)
    saved_folder.mkdir(parents=True)
    (saved_folder / "betas.txt").write_text("0 0 0 0 0 0 0 0 0 0\n")

    assert layer.export_xml(tmp_path) == saved_folder
    assert not (saved_folder / "right.xml").exists()
    assert not (saved_folder / "right_ball.xml").exists()
    assert not (saved_folder / "meshes").exists()


def test_robomano_export_xml_existing_folder_warns_on_beta_mismatch(
    mano_assets_root,
    tmp_path,
):
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.full((10,), 2e-5),
    )
    saved_folder = tmp_path / "right" / _beta_tag(layer._shape_betas)
    saved_folder.mkdir(parents=True)
    (saved_folder / "betas.txt").write_text("0 0 0 0 0 0 0 0 0 0\n")

    with _capture_loguru_warnings() as messages:
        assert layer.export_xml(tmp_path) == saved_folder
    assert any("beta difference" in message for message in messages)
    assert not (saved_folder / "right.xml").exists()


def test_robomano_export_xml_existing_folder_warns_on_missing_beta_txt(
    mano_assets_root,
    tmp_path,
):
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.zeros(10),
    )
    saved_folder = tmp_path / "right" / _beta_tag(layer._shape_betas)
    saved_folder.mkdir(parents=True)

    with _capture_loguru_warnings() as messages:
        assert layer.export_xml(tmp_path) == saved_folder
    assert any("missing betas.txt in folder" in message for message in messages)
    assert not (saved_folder / "right.xml").exists()


def test_robomano_zero_beta_reduced_xml_uses_robowrapper_joint_frames(
    mano_assets_root,
    tmp_path,
):
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="right",
        betas=torch.zeros(10),
    )

    saved_folder = layer.export_xml(tmp_path)
    root = ET.parse(saved_folder / "right.xml").getroot()

    index1y = root.find(".//body[@name='index1y']")
    index1x = root.find(".//body[@name='index1x']")
    index2 = root.find(".//body[@name='index2']")
    index3 = root.find(".//body[@name='index3']")
    thumb1y = root.find(".//body[@name='thumb1y']")
    thumb2 = root.find(".//body[@name='thumb2']")
    thumb3 = root.find(".//body[@name='thumb3']")

    assert torch.allclose(
        _vec(index1y.attrib["pos"]),
        torch.tensor([-0.0880972, -0.00520036, 0.020686]),
        atol=1e-6,
        rtol=1e-6,
    )
    _assert_same_rotation(
        index1y.attrib["quat"],
        "0.0207461 -0.704984 -0.0206397 -0.708619",
    )
    assert "quat" not in index1x.attrib
    assert torch.allclose(
        _vec(index2.attrib["pos"]),
        torch.tensor([0.00238357, -0.00591432, -0.0323765]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in index2.attrib
    assert torch.allclose(
        _vec(index3.attrib["pos"]),
        torch.tensor([0.0, 0.0, -0.0221942]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in index3.attrib
    _assert_same_rotation(
        index2.find("geom").attrib["quat"],
        "0.0207461 0.704984 0.0206397 0.708619",
    )

    _assert_same_rotation(
        thumb1y.attrib["quat"],
        "0.457776 -0.494578 -0.459441 -0.578574",
    )
    assert torch.allclose(
        _vec(thumb2.attrib["pos"]),
        torch.tensor([0.0252649, -0.0175957, 0.0]),
        atol=1e-6,
        rtol=1e-6,
    )
    _assert_same_rotation(
        thumb2.attrib["quat"],
        "0.574412 -0.601175 0.038173 0.554241",
    )
    assert torch.allclose(
        _vec(thumb3.attrib["pos"]),
        torch.tensor([0.0, 0.0, -0.0270942]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in thumb3.attrib
    _assert_same_rotation(
        thumb2.find("geom").attrib["quat"],
        "0.303832 0.79185 -0.375505 0.373706",
    )


def test_robomano_left_zero_beta_reduced_xml_uses_mirrored_joint_frames(
    mano_assets_root,
    tmp_path,
):
    layer = RoboManoLayer(
        mano_assets_root=mano_assets_root,
        side="left",
        betas=torch.zeros(10),
    )

    saved_folder = layer.export_xml(tmp_path)
    root = ET.parse(saved_folder / "left.xml").getroot()

    index1y = root.find(".//body[@name='index1y']")
    index1x = root.find(".//body[@name='index1x']")
    index2 = root.find(".//body[@name='index2']")
    index3 = root.find(".//body[@name='index3']")
    thumb1y = root.find(".//body[@name='thumb1y']")
    thumb2 = root.find(".//body[@name='thumb2']")
    thumb3 = root.find(".//body[@name='thumb3']")

    index_frame = torch.from_numpy(mano_xml_frame_matrix("left", 1)).float()
    thumb1_frame = torch.from_numpy(mano_xml_frame_matrix("left", 13)).float()
    thumb2_frame = torch.from_numpy(mano_xml_frame_matrix("left", 14)).float()

    assert torch.allclose(
        _vec(index1y.attrib["pos"]),
        torch.tensor([0.0880972, -0.00520036, 0.020686]),
        atol=1e-6,
        rtol=1e-6,
    )
    _assert_rotation_matrix(index1y.attrib["quat"], index_frame)
    assert "quat" not in index1x.attrib
    assert torch.allclose(
        _vec(index2.attrib["pos"]),
        torch.tensor([-0.00238357, -0.00591432, -0.0323765]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in index2.attrib
    assert torch.allclose(
        _vec(index3.attrib["pos"]),
        torch.tensor([0.0, 0.0, -0.0221942]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in index3.attrib
    _assert_rotation_matrix(index2.find("geom").attrib["quat"], index_frame.T)

    _assert_rotation_matrix(thumb1y.attrib["quat"], thumb1_frame)
    assert torch.allclose(
        _vec(thumb2.attrib["pos"]),
        torch.tensor([-0.0252649, -0.0175957, 0.0]),
        atol=1e-6,
        rtol=1e-6,
    )
    _assert_rotation_matrix(thumb2.attrib["quat"], thumb1_frame.T @ thumb2_frame)
    assert torch.allclose(
        _vec(thumb3.attrib["pos"]),
        torch.tensor([0.0, 0.0, -0.0270942]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert "quat" not in thumb3.attrib
    _assert_rotation_matrix(thumb2.find("geom").attrib["quat"], thumb2_frame.T)
