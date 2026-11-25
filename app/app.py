#!/usr/bin/env python3
"""DefectVision — production-grade UI for the surface defect detection system.

The detection functions, business flow and data interfaces are unchanged:
images go through scripts.predict.load_model() and scripts.predict.predict_image().
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import design  # noqa: E402
from scripts.predict import load_model, predict_image  # noqa: E402

CLASS_NAMES = {0: "broken_large", 1: "broken_small", 2: "contamination"}
CLASS_LABELS = {
    "broken_large": "大面积破损",
    "broken_small": "小面积破损",
    "contamination": "表面污染",
}

st.set_page_config(
    page_title="DefectVision · 工业表面缺陷检测",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
design.apply_theme()


# --------------------------------------------------------------------------- #
# Session state and data helpers
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "weights": str(ROOT / "outputs" / "best.pt"),
        "conf": 0.25,
        "iou": 0.45,
        "imgsz": 416,
        "result": None,
        "result_hash": None,
        "result_settings": None,
        "result_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_result() -> None:
    st.session_state["result"] = None
    st.session_state["result_hash"] = None
    st.session_state["result_error"] = None


def load_metrics() -> dict:
    path = ROOT / "outputs" / "evaluation" / "metrics.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {
        "mAP50": 0.0,
        "mAP50_95": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "per_class_mAP50_95": [],
    }


@st.cache_resource(show_spinner=False)
def cached_model(weights_path: str):
    return load_model(weights_path)


def decode_upload(uploaded) -> np.ndarray:
    data = np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法解码该图像，请确认文件未损坏。")
    return image


def current_settings() -> tuple:
    return (
        float(st.session_state["conf"]),
        float(st.session_state["iou"]),
        int(st.session_state["imgsz"]),
        str(st.session_state["weights"]),
    )


def status_badge() -> str:
    return "模型就绪" if Path(st.session_state["weights"]).is_file() else "缺少权重"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def detection_page() -> None:
    metrics = load_metrics()
    design.page_header(
        "检测工作台",
        "上传瓶体图像，识别大破损、小破损与表面污染；结果可逐项追溯。",
        status_badge(),
    )
    design.kpi_row(
        [
            (f"{metrics['mAP50']:.3f}", "mAP@0.5", "增强后模型"),
            (f"{metrics['mAP50_95']:.3f}", "mAP@0.5:0.95", "验证集 32 张"),
            ("3", "缺陷类别", "broken / contamination"),
            ("416px", "推理分辨率", "CPU 友好配置"),
        ]
    )

    left, right = st.columns([4, 7], gap="large")

    with left:
        design.card("图像输入", "支持 PNG、JPG、BMP；单张图像实时推理。")
        uploaded = st.file_uploader(
            "上传瓶体图像",
            type=["png", "jpg", "jpeg", "bmp"],
            key="uploader",
            on_change=clear_result,
            help="建议使用与训练数据相近的光照和角度。",
        )
        design.close_card()

        design.card("推理参数", "参数会即时用于下一次检测，不影响已生成的结果。")
        st.slider(
            "置信度阈值",
            0.05, 0.95, step=0.05,
            key="conf",
            help="阈值越低，越容易显示低置信度候选框。",
        )
        st.slider(
            "NMS IoU",
            0.10, 0.90, step=0.05,
            key="iou",
            help="重叠框超过该比例时只保留得分较高的一个。",
        )
        st.selectbox(
            "输入尺寸",
            [320, 416, 512, 640],
            key="imgsz",
        )
        run = st.button("开始检测", type="primary", use_container_width=True)
        st.markdown(
            '<div class="dv-hint">模型权重路径可在「模型配置」页修改。</div>',
            unsafe_allow_html=True,
        )
        design.close_card()

    with right:
        design.card("检测结果", "检测框、类别与置信度会同时展示。")
        if uploaded is None:
            design.empty_state(
                "等待上传图像",
                "选择一张瓶体图像后即可开始检测。界面会先展示原图，再给出结构化结果。",
                "↥",
            )
            design.close_card()
            return

        image = decode_upload(uploaded)
        image_hash = hashlib.md5(uploaded.getvalue()).hexdigest()

        if run:
            clear_result()
            weight_path = Path(st.session_state["weights"])
            if not weight_path.is_file():
                st.session_state["result_error"] = f"找不到权重文件：{weight_path}"
            else:
                try:
                    with st.spinner("正在加载模型并推理…"):
                        model = cached_model(str(weight_path))
                        annotated, detections = predict_image(
                            model,
                            image,
                            conf=float(st.session_state["conf"]),
                            iou=float(st.session_state["iou"]),
                            imgsz=int(st.session_state["imgsz"]),
                        )
                    st.session_state["result"] = (annotated, detections)
                    st.session_state["result_hash"] = image_hash
                    st.session_state["result_settings"] = current_settings()
                except Exception as exc:  # noqa: BLE001 - surface to UI
                    st.session_state["result_error"] = str(exc)

        if st.session_state.get("result_error"):
            design.banner("error", "检测失败", st.session_state["result_error"])
            design.close_card()
            return

        result = st.session_state.get("result")
        if result is None or st.session_state.get("result_hash") != image_hash:
            preview_col, placeholder_col = st.columns(2)
            with preview_col:
                st.caption("原图预览")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            with placeholder_col:
                design.empty_state("尚未检测", "点击左侧「开始检测」生成结果。", "◌")
            design.close_card()
            return

        annotated, detections = result
        if st.session_state.get("result_settings") != current_settings():
            design.banner("info", "参数已更新", "当前结果基于上一次参数，点击「开始检测」刷新。")

        preview_col, result_col = st.columns(2)
        with preview_col:
            st.caption("原图")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        with result_col:
            st.caption("检测视图")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown('<div class="dv-divider"></div>', unsafe_allow_html=True)

        if detections:
            rows = [
                {
                    "类别": item["class_name"],
                    "置信度": round(item["confidence"], 4),
                    "坐标(x1,y1,x2,y2)": [round(v, 1) for v in item["box"]],
                }
                for item in detections
            ]
            design.detection_table(rows, error=True)
            design.banner(
                "error",
                f"检测到 {len(detections)} 处缺陷或污染",
                "结果已按报错标红，请复核高亮区域并按产线流程处理。",
            )
        else:
            design.detection_table(
                [{
                    "类别": "没有缺陷",
                    "置信度": "-",
                    "坐标(x1,y1,x2,y2)": "-",
                }],
                error=False,
            )
            design.banner(
                "ok",
                "产品完好，没有缺陷",
                f"当前置信度阈值 {float(st.session_state['conf']):.2f}；如需复查可调低后重新检测。",
            )
        design.close_card()


def model_page() -> None:
    metrics = load_metrics()
    design.page_header("模型配置", "管理推理权重与参数，查看当前模型能力与消融对照。", status_badge())

    config_col, info_col = st.columns([5, 7], gap="large")
    with config_col:
        design.card("推理设置", "与检测工作台共用同一组参数。")
        st.text_input(
            "模型权重路径",
            key="weights",
            help="默认使用 outputs/best.pt（增强数据训练后的模型）。",
        )
        st.slider("置信度阈值", 0.05, 0.95, step=0.05, key="conf")
        st.slider("NMS IoU", 0.10, 0.90, step=0.05, key="iou")
        st.selectbox("输入尺寸", [320, 416, 512, 640], key="imgsz")
        design.banner(
            "info" if Path(st.session_state["weights"]).is_file() else "warn",
            status_badge(),
            "权重文件可用，可直接检测。" if Path(st.session_state["weights"]).is_file()
            else "请检查权重路径，或先运行训练脚本。",
        )
        design.close_card()

    with info_col:
        design.card("模型档案", "YOLOv8n · CBAM 注意力 · WIoU 损失")
        st.markdown(
            """
            <div class="dv-hint">
              基于 MVTec AD bottle 类别训练。缺陷训练图经翻转、旋转与亮度增强后由 51 张扩充至 255 张。
              输入图像会按所选尺寸缩放，输出类别、检测框与置信度。
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="dv-divider"></div>', unsafe_allow_html=True)
        design.section_title("验证集指标")
        st.dataframe(
            [
                {"指标": "mAP@0.5", "数值": round(float(metrics["mAP50"]), 4)},
                {"指标": "mAP@0.5:0.95", "数值": round(float(metrics["mAP50_95"]), 4)},
                {"指标": "Precision", "数值": round(float(metrics["precision"]), 4)},
                {"指标": "Recall", "数值": round(float(metrics["recall"]), 4)},
            ],
            use_container_width=True,
            hide_index=True,
        )
        design.close_card()

        design.card("消融对照", "同一验证集，不同数据与损失配置。")
        st.dataframe(
            [
                {"配置": "CBAM + CIoU（51 张缺陷）", "mAP@0.5": 0.0987},
                {"配置": "CBAM + WIoU（51 张缺陷）", "mAP@0.5": 0.1035},
                {"配置": "CBAM + WIoU（255 张缺陷，当前）", "mAP@0.5": 0.5676},
            ],
            use_container_width=True,
            hide_index=True,
        )
        design.close_card()


def dataset_page() -> None:
    design.page_header("数据集概览", "了解训练与验证数据构成，便于判断结果置信度。", "MVTec AD · bottle")

    left, right = st.columns([5, 7], gap="large")
    with left:
        design.card("数据构成", "缺陷图经增强后用于训练；验证集保持原始图像。")
        counts = [("大面积破损", 20), ("小面积破损", 22), ("表面污染", 21)]
        for label, value in counts:
            design.bar_row(label, value, 22)
        st.markdown('<div class="dv-divider"></div>', unsafe_allow_html=True)
        design.kpi_row(
            [
                ("255", "训练缺陷图", "含 204 张增强图"),
                ("32", "验证图像", "12 缺陷 + 20 正常"),
                ("209", "训练负样本", "正常瓶体"),
            ]
        )
        design.close_card()

    with right:
        design.card("验证集样本", "从 12 张缺陷验证图中抽取的示例。")
        val_root = ROOT / "data" / "yolo_bottle_aug"
        label_dir = val_root / "labels" / "val"
        image_dir = val_root / "images" / "val"
        samples = []
        if label_dir.is_dir():
            for lab in sorted(label_dir.glob("*.txt")):
                lines = [line for line in lab.read_text(encoding="utf-8").splitlines() if line.strip()]
                if not lines:
                    continue
                cls = int(lines[0].split()[0])
                img_path = image_dir / f"{lab.stem}.png"
                if img_path.is_file():
                    data = np.fromfile(str(img_path), dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    if img is not None:
                        samples.append((CLASS_NAMES.get(cls, str(cls)), img))

        if not samples:
            design.empty_state("暂无样本", "数据集尚未生成，请先运行数据准备脚本。", "▦")
        else:
            cols = st.columns(3)
            for idx, (cls, img) in enumerate(samples[:6]):
                with cols[idx % 3]:
                    st.image(
                        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                        caption=CLASS_LABELS.get(cls, cls),
                        use_container_width=True,
                    )
        design.close_card()


def guide_page() -> None:
    design.page_header("使用指南", "从上传到复核，完成一次标准质检流程。", "快速上手")

    left, right = st.columns([6, 6], gap="large")
    with left:
        design.card("标准流程")
        design.step("01", "上传图像", "在检测工作台选择一张瓶体图像，建议使用稳定光照。")
        design.step("02", "设置阈值", "置信度 0.25 是平衡点；误报多则提高，漏检多则降低。")
        design.step("03", "开始检测", "系统返回检测框、类别与置信度；未发现缺陷会明确提示产品完好。")
        design.step("04", "复核结果", "结合原图与检测视图判断，必要时保存图像用于追溯。")
        design.close_card()

    with right:
        design.card("常见问题")
        with st.expander("为什么同一处缺陷出现多个框？"):
            st.write("模型在边界不确定时会给出多个候选框。适当提高 NMS IoU 或置信度阈值可减少重复。")
        with st.expander("没有检测到缺陷代表什么？"):
            st.write("表示当前阈值下模型未发现候选缺陷，界面会明确显示“产品完好，没有缺陷”。")
        with st.expander("如何提升精度？"):
            st.write("补充真实缺陷样本是最有效的方式；当前版本已通过数据增强将缺陷图扩充到 255 张。")
        design.close_card()

        design.card("无障碍与交互")
        st.markdown(
            """
            <div class="dv-hint">
              所有主要操作均可通过键盘完成；焦点状态有清晰描边。触控目标不小于 44px。
              布局在窄屏下自动堆叠，关键结果与操作始终可见。
            </div>
            """,
            unsafe_allow_html=True,
        )
        design.close_card()


# --------------------------------------------------------------------------- #
# App entry
# --------------------------------------------------------------------------- #
init_state()
pages = [
    st.Page(detection_page, title="检测工作台"),
    st.Page(model_page, title="模型配置"),
    st.Page(dataset_page, title="数据集概览"),
    st.Page(guide_page, title="使用指南"),
]
navigation = st.navigation(pages, position="sidebar")
navigation.run()
