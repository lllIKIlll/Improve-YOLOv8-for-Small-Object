# -*- coding: utf-8 -*-
"""
在导入 ultralytics 之前调用，将 CA 模块注册到 ultralytics 的 parse_model 中
这样可以在 YAML 配置中使用 CA 模块
"""

import sys


def register_custom_modules():
    """注册 CA 到 ultralytics.nn.tasks"""
    try:
        from ultralytics.nn import tasks
        from ultralytics.nn.modules import block, conv

        # 导入自定义 CA 模块
        from modules.attention import CA

        # 添加到 tasks 的全局命名空间，使 parse_model 能解析 CA
        if not hasattr(tasks, "CA"):
            tasks.CA = CA
            tasks.globals()["CA"] = CA

        # 确保 parse_model 能正确处理 CA
        # 检查 tasks 模块的 parse_model 中是否需要特殊处理
        import ultralytics.nn.tasks as tasks_module

        if "CA" not in dir(tasks_module):
            setattr(tasks_module, "CA", CA)

        return True
    except Exception as e:
        print(f"注册自定义模块失败: {e}")
        return False


if __name__ == "__main__":
    register_custom_modules()
    from ultralytics import YOLO

    # 测试 CA 模型是否能正确加载
    model = YOLO("models/yolov8n-ca.yaml")
    model.info()
    print("CA 模块注册成功！")
