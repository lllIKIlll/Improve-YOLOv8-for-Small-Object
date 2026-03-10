# -*- coding: utf-8 -*-
"""
Register CBAM and CA (Coordinate Attention) modules to ultralytics package
Some ultralytics versions do not import CBAM in tasks.py, causing KeyError
Run: python scripts/patch_ultralytics.py
"""

import sys
from pathlib import Path

# Add project root directory to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def patch_ultralytics():
    """Inject CA module support into ultralytics"""
    import ultralytics
    ultralytics_path = Path(ultralytics.__file__).parent
    nn_path = ultralytics_path / "nn"
    modules_path = nn_path / "modules"
    tasks_path = nn_path / "tasks.py"

    if not tasks_path.exists():
        print(f"Error: {tasks_path} not found")
        return False

    # 1. Add CA to conv.py or block.py
    conv_file = modules_path / "conv.py"
    block_file = modules_path / "block.py"

    ca_code = '''
class h_sigmoid(nn.Module):
    """Hard Sigmoid activation for lightweight networks."""
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class CA(nn.Module):
    """Coordinate Attention (CA) for small object detection
    
    In AMP (FP16) training, pooling and attention calculations are forced to use float32 to avoid numerical overflow.
    """
    def __init__(self, c1, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c1 // reduction)
        self.conv1 = nn.Conv2d(c1, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_sigmoid()
        self.conv_h = nn.Conv2d(mip, c1, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, c1, 1, 1, 0)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w
'''

    # Write independent attention file
    attention_file = modules_path / "attention_ca.py"
    attention_content = '''# CA (Coordinate Attention) for small object detection
import torch
import torch.nn as nn

''' + ca_code
    attention_file.write_text(attention_content, encoding="utf-8")

    # 2. Update __init__.py
    init_file = modules_path / "__init__.py"
    init_content = init_file.read_text(encoding="utf-8")
    if "attention_ca" not in init_content:
        if "from .conv import" in init_content:
            init_content = init_content.replace(
                "from .conv import",
                "from .attention_ca import CA  # noqa: F401\nfrom .conv import",
                1,
            )
        else:
            init_content = init_content.rstrip() + "\nfrom .attention_ca import CA  # noqa: F401\n"
        if '"CBAM"' in init_content and '"CA"' not in init_content:
            init_content = init_content.replace('"CBAM"', '"CA", "CBAM"', 1)
        init_file.write_text(init_content, encoding="utf-8")

    # 3. Update tasks.py: Add CBAM, CA import and parse_model processing
    tasks_content = tasks_path.read_text(encoding="utf-8")

    # Add CBAM import (if missing)
    if "CBAM" not in tasks_content:
        tasks_content = tasks_content.replace(
            "    Concat,",
            "    CBAM,\n    Concat,",
            1,
        )

    # Add CA import
    if "CA" not in tasks_content or "CA," not in tasks_content:
        if "CBAM," in tasks_content:
            tasks_content = tasks_content.replace("CBAM,", "CA, CBAM,", 1)
        elif "CBAM\n" in tasks_content and "CA" not in tasks_content:
            tasks_content = tasks_content.replace("CBAM\n", "CA, CBAM\n", 1)

    # Add args processing for CBAM/CA in parse_model: args = [ch[f], *args] for both CBAM and CA
    if "m in {CA" not in tasks_content and "m in frozenset({CBAM" not in tasks_content:
        # Insert before "elif m is CBFuse:"
        insert_block = """        elif m in frozenset({CBAM, CA}):
            args = [ch[f], *args]
        elif m is CBFuse:"""
        if "elif m is CBFuse:" in tasks_content and "m in frozenset({CBAM" not in tasks_content:
            tasks_content = tasks_content.replace(
                "        elif m is CBFuse:",
                insert_block,
                1,
            )
        else:
            # Backup: Insert before else branch
            if "        else:\n            c2 = ch[f]" in tasks_content:
                tasks_content = tasks_content.replace(
                    "        else:\n            c2 = ch[f]",
                    "        elif m in frozenset({CBAM, CA}):\n            args = [ch[f], *args]\n        else:\n            c2 = ch[f]",
                    1,
                )

    tasks_path.write_text(tasks_content, encoding="utf-8")

    print("CBAM 与 CA 模块已成功注册到 ultralytics！")
    return True


if __name__ == "__main__":
    success = patch_ultralytics()
    sys.exit(0 if success else 1)
