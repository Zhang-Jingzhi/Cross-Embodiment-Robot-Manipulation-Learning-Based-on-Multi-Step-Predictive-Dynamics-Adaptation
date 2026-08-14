"""Robot arm configuration manager for Metaworld environments.

This module provides utilities to dynamically switch between different robot arms
without manually editing XML files. It creates environment-specific XML files
based on a robot configuration.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

RobotType = Literal[
    "sawyer",
    "kuka",
    "panda",
    "ur5e",
    "ur10e",
    "gen3",
    "xarm7",
    "unitree_z1",
    "viperx",
]


class RobotConfigManager:
    """Manages robot configurations for Metaworld environments."""
    
    # Mapping of robot types to their XML dependencies
    ROBOT_DEPENDENCIES = {
        "sawyer": "xyz_base_dependencies.xml",
        "kuka": "xyz_base_dependencies_kuka.xml", 
        "panda": "xyz_base_dependencies_panda.xml",
        "ur5e": "xyz_base_dependencies_ur5e.xml",
        "ur10e": "xyz_base_dependencies_ur10e.xml",
        "gen3": "xyz_base_dependencies_gen3.xml",
        "xarm7": "xyz_base_dependencies_xarm7.xml",
        "unitree_z1": "xyz_base_dependencies_unitree.xml",
        "viperx": "xyz_base_dependencies_viperx.xml",
    }
    
    ROBOT_BASE_FILES = {
        "sawyer": "xyz_base.xml",
        "kuka": "xyz_base_kuka.xml",
        "panda": "xyz_base_panda.xml",
        "ur5e": "xyz_base_ur5e.xml",
        "ur10e": "xyz_base_ur10e.xml",
        "gen3": "xyz_base_gen3.xml",
        "xarm7": "xyz_base_xarm7.xml",
        "unitree_z1": "xyz_base_unitree.xml",
        "viperx": "xyz_base_viperx.xml",
    }
    
    def __init__(self, assets_dir: str | Path | None = None):
        """Initialize the robot config manager.
        
        Args:
            assets_dir: Path to Metaworld assets directory. 
                       If None, uses default location.
        """
        if assets_dir is None:
            # Try to find assets directory
            # robot_config_manager.py is in metaworld/envs/
            # assets_v2 is in metaworld/envs/assets_v2
            current_file = Path(__file__).resolve()
            assets_dir = current_file.parent / "assets_v2"
        
        self.assets_dir = Path(assets_dir)
        self.sawyer_xyz_dir = self.assets_dir / "sawyer_xyz"
        self.objects_assets_dir = self.assets_dir / "objects" / "assets"

    def get_robot_base_xml_path(self, robot_type: RobotType) -> Path:
        return self.objects_assets_dir / self.ROBOT_BASE_FILES[robot_type]
        
    def create_robot_specific_xml(
        self, 
        env_name: str, 
        robot_type: RobotType,
        output_dir: str | Path | None = None
    ) -> Path:
        """Create a robot-specific XML file for an environment.
        
        Args:
            env_name: Environment name (e.g., "push-v2", "reach-v2")
            robot_type: Type of robot arm
            output_dir: Directory to save the generated XML. 
                       If None, saves to a generated configs directory.
        
        Returns:
            Path to the generated XML file.
        """
        # Find the base XML file
        xml_filename = f"sawyer_{env_name.replace('-v2', '_v2')}.xml"
        base_xml_path = self.sawyer_xyz_dir / xml_filename
        
        if not base_xml_path.exists():
            raise FileNotFoundError(f"Base XML not found: {base_xml_path}")
        
        # Read the base XML
        with open(base_xml_path, "r") as f:
            content = f.read()
        
        # Extract the robot-specific section
        robot_section = self._extract_robot_section(content, robot_type)
        
        if robot_section is None:
            raise ValueError(
                f"Could not find {robot_type} section in {xml_filename}. "
                f"The XML might not have this robot configuration."
            )
        
        # Create output directory
        if output_dir is None:
            output_dir = self.assets_dir / robot_type
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the robot-specific XML
        output_path = output_dir / xml_filename
        with open(output_path, "w") as f:
            f.write(robot_section)
        
        return output_path
    
    def _extract_robot_section(
        self, 
        content: str, 
        robot_type: RobotType
    ) -> str | None:
        """Extract the XML section for a specific robot type.
        
        Args:
            content: Full XML content with multiple robot configurations
            robot_type: Type of robot to extract
            
        Returns:
            XML content for the specified robot, or None if not found.
        """
        # Look for the robot's dependency file in the content
        dependency_file = self.ROBOT_DEPENDENCIES[robot_type]
        base_file = self.ROBOT_BASE_FILES[robot_type]
        
        import re
        
        # Strategy: Find all <!-- ... --> comment blocks AND all <mujoco>...</mujoco> blocks
        # Then check which one contains our target robot
        
        # Find comment blocks that contain mujoco
        comment_pattern = r'<!--(.*?)-->'
        comments = re.findall(comment_pattern, content, re.DOTALL)
        
        for comment in comments:
            if '<mujoco>' in comment and dependency_file in comment and base_file in comment:
                # Extract the mujoco block from within the comment
                mujoco_match = re.search(r'<mujoco>.*?</mujoco>', comment, re.DOTALL)
                if mujoco_match:
                    return mujoco_match.group(0)
        
        # Also check non-commented mujoco blocks
        mujoco_pattern = r'<mujoco>.*?</mujoco>'
        mujoco_blocks = re.findall(mujoco_pattern, content, re.DOTALL)
        
        for block in mujoco_blocks:
            if dependency_file in block and base_file in block:
                return block
        
        return None
    
    def generate_all_robot_configs(
        self,
        env_names: list[str] | None = None,
        robot_types: list[RobotType] | None = None,
        output_base_dir: str | Path | None = None
    ) -> dict[str, dict[str, Path]]:
        """Generate XML configs for multiple environments and robots.
        
        Args:
            env_names: List of environment names. If None, uses common envs.
            robot_types: List of robot types. If None, uses all supported.
            output_base_dir: Base directory for outputs.
        
        Returns:
            Dictionary mapping env_name -> robot_type -> xml_path
        """
        if env_names is None:
            env_names = [
                "reach-v2", "push-v2", "pick_place-v2", 
                "door_pull", "drawer", "button_press", "button_press_topdown", "peg_insertion_side", "window_horizontal"
            ]
        
        if robot_types is None:
            robot_types = list(self.ROBOT_DEPENDENCIES.keys())
        
        results = {}
        
        for env_name in env_names:
            results[env_name] = {}
            for robot_type in robot_types:
                try:
                    output_dir = None
                    if output_base_dir:
                        output_dir = Path(output_base_dir) / robot_type
                    
                    xml_path = self.create_robot_specific_xml(
                        env_name, robot_type, output_dir
                    )
                    results[env_name][robot_type] = xml_path
                    print(f"✓ Generated {env_name} for {robot_type}: {xml_path}")
                except Exception as e:
                    print(f"✗ Failed to generate {env_name} for {robot_type}: {e}")
        
        return results


def get_robot_xml_path(
    env_name: str,
    robot_type: RobotType = "sawyer",
    assets_dir: str | Path | None = None,
    regenerate: bool = False
) -> Path:
    """Get the XML path for a specific environment and robot.
    
    This function handles the mapping from MetaWorld task names (e.g., window-open-v2)
    to the actual underlying XML filenames (e.g., sawyer_window_horizontal.xml).
    """
    manager = RobotConfigManager(assets_dir)
    
    TASK_TO_XML_SUFFIX = {
        "reach-v2": "reach_v2",
        "push-v2": "push_v2",
        "pick-place-v2": "pick_place_v2",
        
        # Window tasks
        "window-open-v2": "window_horizontal",
        "window-close-v2": "window_horizontal",
        
        # Door tasks - both user-facing names and class-derived names
        "door-open-v2": "door_pull",
        "door-close-v2": "door_pull",
        "door-v2": "door_pull",  # SawyerDoorEnvV2 -> door-v2 (class-derived)
        "door-lock-v2": "door_lock",
        "door-unlock-v2": "door_lock",
        
        # Drawer tasks - both user-facing names and class-derived names
        "drawer-open-v2": "drawer",
        "drawer-close-v2": "drawer",
        "drawer-v2": "drawer",  # SawyerDrawerOpenEnvV2 -> drawer-open-v2 (also matches)
        
        # Button tasks
        "button-press-v2": "button_press",
        "button-press-topdown-v2": "button_press_topdown",
        
        # Peg tasks
        "peg-insert-side-v2": "peg_insertion_side",
        "peg-insertion-side-v2": "peg_insertion_side",
        
        # Additional class-derived mappings for tasks where class name doesn't match task name
        "nut-assembly-v2": "assembly_peg",  # SawyerNutAssemblyEnvV2
        "nut-disassemble-v2": "assembly_peg",  # SawyerNutDisassembleEnvV2
        "assembly-v2": "assembly_peg",
        "disassemble-v2": "assembly_peg",
        
        # Sweep tasks
        "sweep-v2": "sweep_v2",
        "sweep-into-v2": "sweep_v2",  # SawyerSweepIntoGoalEnvV2
        "sweep-into-goal-v2": "sweep_v2",
        
        # Handle tasks
        "handle-press-v2": "handle_press",
        "handle-press-side-v2": "handle_press_sideways",
        "handle-pull-v2": "handle_press",
        "handle-pull-side-v2": "handle_press_sideways",
        
        # Dial tasks  
        "dial-turn-v2": "dial",
        
        # Faucet tasks
        "faucet-open-v2": "faucet",
        "faucet-close-v2": "faucet",
        
        # Shelf tasks
        "shelf-place-v2": "shelf_placing",
        
        # Lever tasks
        "lever-pull-v2": "lever_pull",
        
        # Coffee tasks
        "coffee-button-v2": "coffee",
        "coffee-pull-v2": "coffee",
        "coffee-push-v2": "coffee",
        
        # Box tasks
        "box-close-v2": "box",
        
        # Stick tasks
        "stick-push-v2": "stick_obj",
        "stick-pull-v2": "stick_obj",
        
        # Plate slide tasks
        "plate-slide-v2": "plate_slide",
        "plate-slide-side-v2": "plate_slide_sideway",
        "plate-slide-back-v2": "plate_slide",
        "plate-slide-back-side-v2": "plate_slide_sideway",
        
        # Peg unplug
        "peg-unplug-side-v2": "peg_unplug_side",
        
        # Hand insert
        "hand-insert-v2": "table_with_hole",
        
        # Basketball
        "basketball-v2": "basketball",
        
        # Bin picking
        "bin-picking-v2": "bin_picking",
        
        # Hammer
        "hammer-v2": "hammer",
        
        # Soccer
        "soccer-v2": "soccer",
        
        # Pick out of hole
        "pick-out-of-hole-v2": "pick_out_of_hole",
        
        # Push variants
        "push-back-v2": "push_back_v2",
        "push-wall-v2": "push_wall_v2",
        
        # Reach wall
        "reach-wall-v2": "reach_wall_v2",
        
        # Pick place wall
        "pick-place-wall-v2": "pick_place_wall_v2",
    }


    if env_name in TASK_TO_XML_SUFFIX:
        xml_base_suffix = TASK_TO_XML_SUFFIX[env_name]
    else:
        xml_base_suffix = env_name.replace('-v2', '_v2')

    output_dir = manager.assets_dir / robot_type
    target_xml_filename = f"sawyer_{xml_base_suffix}.xml"
    output_path = output_dir / target_xml_filename
    
    if output_path.exists() and not regenerate:
        return output_path
    
    return manager.create_robot_specific_xml(xml_base_suffix, robot_type)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate robot-specific XML configurations for Metaworld"
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name (e.g., push-v2). If not specified, generates for all common envs."
    )
    parser.add_argument(
        "--robot",
        type=str,
        choices=["sawyer", "kuka", "panda", "ur5e", "ur10e"],
        help="Robot type. If not specified, generates for all robots."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated XMLs"
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=None,
        help="Path to Metaworld assets directory"
    )
    
    args = parser.parse_args()
    
    manager = RobotConfigManager(args.assets_dir)
    
    if args.env and args.robot:
        # Generate single config
        xml_path = manager.create_robot_specific_xml(
            args.env, args.robot, args.output_dir
        )
        print(f"✓ Generated: {xml_path}")
    else:
        # Generate multiple configs
        env_names = [args.env] if args.env else None
        robot_types = [args.robot] if args.robot else None
        
        results = manager.generate_all_robot_configs(
            env_names, robot_types, args.output_dir
        )
        
        print(f"\n✓ Generated {sum(len(r) for r in results.values())} configurations")
