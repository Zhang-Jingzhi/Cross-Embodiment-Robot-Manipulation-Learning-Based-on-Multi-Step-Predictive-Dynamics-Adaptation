"""Generate controlled robot dynamics XML variants for MetaWorld.

This module creates perturbed robot-specific XMLs by scaling robot-body inertial
parameters and/or joint damping/armature parameters. The generated files live
next to the original robot/task XMLs so existing relative asset paths keep
working.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


ENV_BODY_MASS_SCALE = "METAWORLD_ROBOT_BODY_MASS_SCALE"
ENV_JOINT_DAMPING_SCALE = "METAWORLD_ROBOT_JOINT_DAMPING_SCALE"
ENV_JOINT_ARMATURE_SCALE = "METAWORLD_ROBOT_JOINT_ARMATURE_SCALE"
ENV_VARIANT_TAG = "METAWORLD_DYNAMICS_VARIANT_TAG"
ENV_VARIANT_ROBOTS = "METAWORLD_DYNAMICS_VARIANT_ROBOTS"


@dataclass(frozen=True)
class DynamicsVariantSpec:
    body_mass_scale: float = 1.0
    joint_damping_scale: float = 1.0
    joint_armature_scale: float = 1.0
    variant_tag: str = ""

    def is_identity(self) -> bool:
        return (
            abs(self.body_mass_scale - 1.0) < 1e-12
            and abs(self.joint_damping_scale - 1.0) < 1e-12
            and abs(self.joint_armature_scale - 1.0) < 1e-12
        )


def _format_scale(scale: float) -> str:
    text = f"{scale:.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def build_variant_tag(spec: DynamicsVariantSpec) -> str:
    if spec.variant_tag:
        return spec.variant_tag
    return (
        f"bm{_format_scale(spec.body_mass_scale)}"
        f"_jd{_format_scale(spec.joint_damping_scale)}"
        f"_ja{_format_scale(spec.joint_armature_scale)}"
    )


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.strip().split()]


def format_float_list(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def scale_numeric_attr(element: ET.Element, attr: str, scale: float) -> None:
    if attr not in element.attrib:
        return
    if abs(scale - 1.0) < 1e-12:
        return
    raw = element.attrib[attr].strip()
    if not raw:
        return
    values = parse_float_list(raw)
    scaled = [value * scale for value in values]
    element.set(attr, format_float_list(scaled))


def scale_base_robot_xml(
    src_path: Path,
    dst_path: Path,
    spec: DynamicsVariantSpec,
) -> None:
    tree = ET.parse(src_path)
    root = tree.getroot()

    for element in root.iter():
        if element.tag == "inertial":
            scale_numeric_attr(element, "mass", spec.body_mass_scale)
            scale_numeric_attr(element, "diaginertia", spec.body_mass_scale)
        elif element.tag == "joint":
            scale_numeric_attr(element, "damping", spec.joint_damping_scale)
            scale_numeric_attr(element, "armature", spec.joint_armature_scale)

    ET.indent(tree, space="  ")
    tree.write(dst_path, encoding="utf-8")


def rewrite_task_xml_include(
    src_path: Path,
    dst_path: Path,
    old_base_filename: str,
    new_base_filename: str,
) -> None:
    tree = ET.parse(src_path)
    root = tree.getroot()

    for element in root.iter("include"):
        include_path = element.attrib.get("file", "")
        if Path(include_path).name == old_base_filename:
            element.set("file", include_path.replace(old_base_filename, new_base_filename))

    ET.indent(tree, space="  ")
    tree.write(dst_path, encoding="utf-8")


def create_dynamics_variant_xml(
    task_xml_path: Path,
    base_xml_path: Path,
    spec: DynamicsVariantSpec,
    regenerate: bool = False,
) -> Path:
    """Create a perturbed task XML plus perturbed base XML and return task path."""
    if spec.is_identity():
        return task_xml_path

    tag = build_variant_tag(spec)
    variant_base_path = base_xml_path.with_name(f"{base_xml_path.stem}__dyn_{tag}.xml")
    variant_task_path = task_xml_path.with_name(f"{task_xml_path.stem}__dyn_{tag}.xml")

    if regenerate or not variant_base_path.exists():
        scale_base_robot_xml(base_xml_path, variant_base_path, spec)

    if regenerate or not variant_task_path.exists():
        rewrite_task_xml_include(
            src_path=task_xml_path,
            dst_path=variant_task_path,
            old_base_filename=base_xml_path.name,
            new_base_filename=variant_base_path.name,
        )

    return variant_task_path
