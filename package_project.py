#!/usr/bin/env python3
"""项目打包脚本：生成包含代码与文档的 zip 文件。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import zipfile

# 需要忽略的目录或文件模式，可按需扩展
IGNORED_NAMES = {".git", "__pycache__", "dist"}


def should_ignore(path: Path) -> bool:
    """判断是否需要忽略当前路径。"""
    parts = set(path.parts)
    return bool(parts & IGNORED_NAMES)


def add_to_zip(zip_handle: zipfile.ZipFile, root: Path, file_path: Path) -> None:
    """将指定文件添加到 Zip 包中，保持相对路径结构。"""
    relative_path = file_path.relative_to(root)
    zip_handle.write(file_path, arcname=relative_path.as_posix())


def build_zip(source_dir: Path, output_path: Path) -> Path:
    """遍历仓库目录，将代码与文档打包成 Zip 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_handle:
        for item in source_dir.rglob("*"):
            if item.is_dir():
                if should_ignore(item):
                    # 若命中忽略列表，则跳过整个子树
                    continue
            else:
                if should_ignore(item):
                    continue
                add_to_zip(zip_handle, source_dir, item)
    return output_path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="打包项目代码与文档")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/guozhong_project.zip"),
        help="Zip 文件输出路径，默认存放在 dist 目录下。",
    )
    return parser.parse_args()


def main() -> None:
    """执行打包流程，并在控制台输出结果路径。"""
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_path = args.output
    archive = build_zip(project_root, output_path)
    print(f"打包完成：{archive}")


if __name__ == "__main__":
    main()
