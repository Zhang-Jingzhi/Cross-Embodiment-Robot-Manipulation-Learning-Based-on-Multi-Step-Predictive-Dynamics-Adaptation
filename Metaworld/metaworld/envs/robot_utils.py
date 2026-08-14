"""Utilities for integrating robot configuration into Metaworld environments.

This module provides functions to patch environment classes with robot-specific
XML loading capabilities, enabling automatic robot switching without manual editing.

IMPORTANT: Uses environment variable METAWORLD_ROBOT_TYPE to persist robot type
across subprocess boundaries (required for multiprocessing with 'spawn' context).
"""

from __future__ import annotations

import os
from pathlib import Path
from functools import wraps


# Environment variable name for robot type (persists across processes)
_ROBOT_TYPE_ENV_VAR = "METAWORLD_ROBOT_TYPE"
_DYNAMICS_VARIANT_ROBOTS_ENV_VAR = "METAWORLD_DYNAMICS_VARIANT_ROBOTS"
_BODY_MASS_SCALE_ENV_VAR = "METAWORLD_ROBOT_BODY_MASS_SCALE"
_JOINT_DAMPING_SCALE_ENV_VAR = "METAWORLD_ROBOT_JOINT_DAMPING_SCALE"
_JOINT_ARMATURE_SCALE_ENV_VAR = "METAWORLD_ROBOT_JOINT_ARMATURE_SCALE"
_DYNAMICS_VARIANT_TAG_ENV_VAR = "METAWORLD_DYNAMICS_VARIANT_TAG"

# Flag to track if patches have been applied in this process
_PATCHES_APPLIED = False


def set_robot_type(robot_type: str | None):
    """Set the global robot type for environment creation.
    
    Uses environment variable to persist across subprocess boundaries.
    
    Args:
        robot_type: Robot arm type (e.g., "ur5e", "ur10e", "panda").
                   None to use default (sawyer).
    """
    if robot_type:
        os.environ[_ROBOT_TYPE_ENV_VAR] = robot_type
        print(f"[RobotConfig] Set global robot type to: {robot_type} (env var: {_ROBOT_TYPE_ENV_VAR})")
    else:
        # Clear the environment variable
        if _ROBOT_TYPE_ENV_VAR in os.environ:
            del os.environ[_ROBOT_TYPE_ENV_VAR]
        print(f"[RobotConfig] Cleared robot type (using default sawyer)")


def get_robot_type() -> str | None:
    """Get the current global robot type from environment variable.
    
    Returns:
        Robot type string or None if not set.
    """
    return os.environ.get(_ROBOT_TYPE_ENV_VAR)


def _parse_dynamics_variant_spec(robot_type: str | None):
    if not robot_type:
        return None

    target_robots = os.environ.get(_DYNAMICS_VARIANT_ROBOTS_ENV_VAR, "").strip()
    if target_robots:
        allowed = {item.strip() for item in target_robots.split(",") if item.strip()}
        if allowed and robot_type not in allowed:
            return None

    body_mass_scale = float(os.environ.get(_BODY_MASS_SCALE_ENV_VAR, "1.0") or 1.0)
    joint_damping_scale = float(os.environ.get(_JOINT_DAMPING_SCALE_ENV_VAR, "1.0") or 1.0)
    joint_armature_scale = float(os.environ.get(_JOINT_ARMATURE_SCALE_ENV_VAR, "1.0") or 1.0)
    variant_tag = os.environ.get(_DYNAMICS_VARIANT_TAG_ENV_VAR, "").strip()

    from metaworld.envs.dynamics_variant_manager import DynamicsVariantSpec

    spec = DynamicsVariantSpec(
        body_mass_scale=body_mass_scale,
        joint_damping_scale=joint_damping_scale,
        joint_armature_scale=joint_armature_scale,
        variant_tag=variant_tag,
    )
    if spec.is_identity():
        return None
    return spec


def get_robot_specific_xml_path(env_name: str, default_xml_path: str) -> str:
    """Get robot-specific XML path, or fall back to default.
    
    Args:
        env_name: Environment name (e.g., "push-v2", "reach-v2")
        default_xml_path: Default XML path to use if robot_type not set
        
    Returns:
        Path to XML file (robot-specific if available, default otherwise)
    """
    robot_type = get_robot_type()
    
    # Debug: Always print what robot type is being used
    print(f"[RobotConfig] get_robot_specific_xml_path called: env={env_name}, robot_type={robot_type}")
    
    # If no robot type specified, use default
    if not robot_type or robot_type == "sawyer" or robot_type == "unknown":
        print(f"[RobotConfig] Using default XML (no robot type or sawyer/unknown): {default_xml_path}")
        return default_xml_path
    
    try:
        from metaworld.envs.dynamics_variant_manager import create_dynamics_variant_xml
        from metaworld.envs.robot_config_manager import RobotConfigManager, get_robot_xml_path
        
        # Try to get or generate robot-specific XML
        robot_xml = get_robot_xml_path(env_name, robot_type, regenerate=False)
        variant_spec = _parse_dynamics_variant_spec(robot_type)
        if variant_spec is not None:
            manager = RobotConfigManager()
            base_xml = manager.get_robot_base_xml_path(robot_type)
            robot_xml = create_dynamics_variant_xml(
                task_xml_path=Path(robot_xml),
                base_xml_path=base_xml,
                spec=variant_spec,
                regenerate=False,
            )
            print(
                "[RobotConfig] Using dynamics-variant XML "
                f"(body_mass={variant_spec.body_mass_scale}, "
                f"joint_damping={variant_spec.joint_damping_scale}, "
                f"joint_armature={variant_spec.joint_armature_scale}): {robot_xml}"
            )
        
        if robot_xml.exists():
            print(f"[RobotConfig] Using {robot_type} XML for {env_name}: {robot_xml}")
            return str(robot_xml)
        else:
            print(f"[RobotConfig] Robot-specific XML not found, using default: {default_xml_path}")
            return default_xml_path
            
    except Exception as e:
        print(f"[RobotConfig] Error loading robot-specific XML: {e}")
        print(f"[RobotConfig] Falling back to default: {default_xml_path}")
        import traceback
        traceback.print_exc()
        return default_xml_path


def wrap_model_name_property(original_property_getter, env_name: str):
    """Wrap a model_name property to use robot-specific XML.
    
    Args:
        original_property_getter: Original property getter function
        env_name: Environment name for this class
        
    Returns:
        Wrapped property getter function
    """
    @wraps(original_property_getter)
    def robot_aware_model_name(self) -> str:
        """Get model name, using robot-specific XML if available."""
        # Get default path from original getter
        default_path = original_property_getter(self)
        
        # Try to get robot-specific path
        return get_robot_specific_xml_path(env_name, default_path)
    
    return robot_aware_model_name


def patch_all_metaworld_envs():
    """Patch all Metaworld v2 environments to use robot-specific XMLs.
    
    This should be called after importing metaworld but before creating environments.
    Uses _PATCHES_APPLIED flag to ensure patching only happens once per process.
    """
    global _PATCHES_APPLIED
    
    # Skip if already patched in this process
    if _PATCHES_APPLIED:
        print(f"[RobotConfig] Patches already applied in this process")
        return
        
    try:
        import metaworld.envs.mujoco.sawyer_xyz.v2 as v2_module
        
        # Get all environment classes
        env_classes = [
            (name, getattr(v2_module, name))
            for name in dir(v2_module)
            if name.startswith("Sawyer") and name.endswith("V2")
        ]
        
        patched_count = 0
        for class_name, env_class in env_classes:
            # Extract environment name from class name
            # E.g., "SawyerPushEnvV2" -> "push-v2"
            env_name_part = class_name.replace("Sawyer", "").replace("EnvV2", "").replace("V2", "")
            # Convert CamelCase to kebab-case
            import re
            env_name = re.sub(r'(?<!^)(?=[A-Z])', '-', env_name_part).lower() + "-v2"
            
            # Check if class has model_name property
            if hasattr(env_class, 'model_name') and isinstance(getattr(env_class, 'model_name'), property):
                # Get the original property
                original_property = getattr(env_class, 'model_name')
                original_fget = original_property.fget
                
                # Wrap it
                wrapped_fget = wrap_model_name_property(original_fget, env_name)
                
                # Replace the property
                setattr(env_class, 'model_name', property(wrapped_fget))
                patched_count += 1
        
        _PATCHES_APPLIED = True
        robot_type = get_robot_type()
        print(f"[RobotConfig] Patched {patched_count} Metaworld v2 environments (robot_type={robot_type})")
        
    except Exception as e:
        print(f"[RobotConfig] Warning: Could not patch Metaworld environments: {e}")


def ensure_robot_config():
    """Ensure robot configuration is applied in the current process.
    
    This function should be called at the start of any subprocess that creates
    Metaworld environments. It checks the environment variable and applies
    patches if needed.
    
    Returns:
        Robot type if set, None otherwise.
    """
    robot_type = get_robot_type()
    if robot_type and robot_type not in ("sawyer", "unknown"):
        print(f"[RobotConfig] Subprocess detected robot_type={robot_type}, applying patches...")
        patch_all_metaworld_envs()
    return robot_type


def initialize_robot_config_from_hydra(config):
    """Initialize robot configuration from Hydra config.
    
    This should be called early in the experiment setup to extract
    robot_type from config and set it globally.
    
    Args:
        config: Hydra configuration object
    """
    robot_type = None
    
    # Try to get robot_type from experiment config
    if hasattr(config, 'experiment') and hasattr(config.experiment, 'robot_type'):
        robot_type = config.experiment.robot_type
    
    # Set global robot type
    if robot_type:
        set_robot_type(robot_type)
        print(f"[RobotConfig] Initialized from config: robot_type={robot_type}")
        
        # Patch all Metaworld environments
        patch_all_metaworld_envs()
        
        # Pre-generate XML for the specified environment if possible
        if hasattr(config, 'env') and hasattr(config.env, 'benchmark'):
            try:
                env_name = None
                if hasattr(config.env.benchmark, 'env_name'):
                    env_name = config.env.benchmark.env_name
                
                if env_name:
                    from metaworld.envs.robot_config_manager import get_robot_xml_path
                    xml_path = get_robot_xml_path(env_name, robot_type)
                    print(f"[RobotConfig] Pre-generated XML: {xml_path}")
            except Exception as e:
                print(f"[RobotConfig] Could not pre-generate XML: {e}")
    else:
        print("[RobotConfig] No robot_type specified, using default (sawyer)")
    
    return robot_type
