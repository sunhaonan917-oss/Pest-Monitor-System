import os
import sys
import io
import time
import json
import threading
import glob
import requests
import pandas as pd
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import torch
import uuid
import numpy as np
from PIL import Image

# ✅ 必须是第一个 Streamlit 命令
st.set_page_config(page_title="智能害虫检测平台", layout="wide")

# 确保能 import src/ 与 methods/
APP_ROOT = os.path.abspath(os.path.dirname(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)

from src.model_builder import build_model, load_checkpoint
from src.infer import predict_5way5shot_one_query
from src.preprocess import load_image_to_tensor

# ---------------------------
# 固定参数 & 线程安全配置
# ---------------------------
CKPT_PATH = "checkpoints/199.tar"
BACKBONE_NAME = "ResNet10_EMA"
USE_GPU = True

# ✅ 修改：直接填入你的 Ngrok 完整地址
NGROK_URL = "https://unneighbourly-janita-hypothecary.ngrok-free.dev"

# 【关键点】作为前端网页和后台线程通信的桥梁
global_config = {"save_dir": "C:/Screenshots"}

# ---------------------------
# CSS 样式
# ---------------------------
st.markdown("""
    <style>
    .app-title { padding: 14px 16px; border-radius: 12px; background: linear-gradient(90deg, #e8f4ff, #f6fff1); border: 1px solid rgba(0,0,0,0.06); margin-bottom: 14px; }
    .app-title h1 { margin: 0; font-size: 24px; }
    .app-title p { margin: 6px 0 0 0; color: rgba(0,0,0,0.6); }
    .card { padding: 12px 14px; border-radius: 12px; border: 1px solid rgba(0,0,0,0.06); background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.04); margin-bottom: 12px; }
    label[data-baseweb="radio"] { align-items: center !important; margin-bottom: 16px !important; }
    section[data-testid="stSidebar"] .stRadio p, div[role="radiogroup"] p, .stRadio label p {
        font-size: 20px !important; font-weight: 600 !important; line-height: normal !important; margin-top: 2px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Session 状态初始化
# ---------------------------
if "class_names" not in st.session_state:
    st.session_state.class_names = ["SFegg", "SFfourth", "SFfifth", "SFsixth", "SFadult"]
if "support_bytes" not in st.session_state:
    st.session_state.support_bytes = [[] for _ in range(5)]
if "query_bytes" not in st.session_state:
    st.session_state.query_bytes = None
if 'save_dir' not in st.session_state:
    st.session_state.save_dir = "C:/Screenshots"

# ==========================================================
# 后台隐形守护线程
# ==========================================================
def auto_curve_logger():
    count_url = f"{NGROK_URL}/get_count"
    # ✅ 关键点：加入 Header 跳过 Ngrok 警告页
    headers = {"ngrok-skip-browser-warning": "69420"}
    buffer_20s = []
    last_save_minute = -1

    while True:
        current_save_dir = global_config["save_dir"]
        try:
            res = requests.get(count_url, headers=headers, timeout=5)
            if res.status_code == 200:
                count = res.json().get("count", 0)
                now_str = datetime.now().strftime("%H:%M:%S")
                buffer_20s.append({"时间": now_str, "检出数量": count})
                if len(buffer_20s) > 20: buffer_20s.pop(0)

            current_time = datetime.now()
            if current_time.second == 0 and current_time.minute != last_save_minute and len(buffer_20s) > 0:
                os.makedirs(current_save_dir, exist_ok=True)
                log_file = os.path.join(current_save_dir, "auto_curve_history.json")
                record = {"timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"), "data": buffer_20s.copy()}
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                last_save_minute = current_time.minute
        except Exception: pass
        time.sleep(1)

if 'logger_thread_started' not in st.session_state:
    t = threading.Thread(target=auto_curve_logger, daemon=True)
    t.start()
    st.session_state.logger_thread_started = True

# ---------------------------
# 工具函数
# ---------------------------
def bytes_to_filelike(b: bytes): return io.BytesIO(b)
def bytes_to_pil(b: bytes): return Image.open(io.BytesIO(b)).convert("RGB")

@st.cache_resource
def load_model_cached(ckpt_path: str, device_str: str, backbone_name: str):
    dev = torch.device(device_str)
    model = build_model(n_way=5, n_support=5, backbone_name=backbone_name)
    load_checkpoint(model, ckpt_path, dev)
    return model

# ---------------------------
# Sidebar 侧边栏
# ---------------------------
st.sidebar.markdown('## 🐛 草地贪夜蛾监测平台')
main_task = st.sidebar.radio("选择功能模式：", ["平台首页", "害虫检测计数", "害虫精确分类", "历史数据管理"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 历史数据与存储配置")
st.session_state.save_dir = st.sidebar.text_input("截图与数据保存目录", value=st.session_state.save_dir)
global_config["save_dir"] = st.session_state.save_dir
save_dir = st.session_state.save_dir
device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")

def run_home_mode():
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 75vh; text-align: center;">
            <h1 style="font-size: 42px; color: #2c3e50; margin-bottom: 20px;">🐛 欢迎使用草地贪夜蛾智能监测平台</h1>
            <div style="background-color: #f8f9fa; padding: 40px; border-radius: 15px; border: 1px solid #e0e0e0; max-width: 800px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <p style="font-size: 18px; color: #555; line-height: 1.8; margin-bottom: 20px;">本平台基于<b>端边协同架构</b>开发。</p>
                <p style="font-size: 18px; font-weight: bold; color: #007bff;">👈 请在左侧菜单栏选择您需要的功能模块开始体验</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==========================================================
# 模式一：实时目标检测 (核心修复版)
# ==========================================================
def run_detection_mode():
    st.markdown('<div class="app-title"><h1>DPC-DINO 实时监测</h1><p>正在接收来自边缘端 Jetson Orin Nano 的无损实时流</p></div>', unsafe_allow_html=True)

    stream_url = f"{NGROK_URL}/video_feed"
    snapshot_url = f"{NGROK_URL}/snapshot"
    count_url = f"{NGROK_URL}/get_count"

    st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-bottom: 20px;">
            <h4 style="margin-top: 0; color: #333;">📡 链路状态确认</h4>
            <p style="color: #555; font-size: 14px;">如果您手机黑屏，请点击 <b><a href="{NGROK_URL}" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;">[ 激活 HTTPS 通道 ]</a></b> 并在弹出的页面点击 "Visit Site" 即可。</p>
        </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([7, 3])

    with col_left:
        st.markdown("#### 📡 实时监控视窗")
        # ✅ 关键点：JavaScript fetch 加入 Header 跳过 Ngrok 警告
        video_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; padding: 0; background-color: transparent; display: flex; justify-content: center; }}
                img {{ width: 100%; border-radius: 8px; border: 3px solid #e8f4ff; background-color: #111; min-height: 520px; object-fit: contain; }}
            </style>
        </head>
        <body>
            <img id="live_stream" alt="正在拉取实时视频流...">
            <script>
                const liveStream = document.getElementById('live_stream');
                async function updateFrame() {{
                    try {{
                        const response = await fetch("{snapshot_url}?t=" + new Date().getTime(), {{
                            headers: {{ "ngrok-skip-browser-warning": "any" }}
                        }});
                        const blob = await response.blob();
                        liveStream.src = URL.createObjectURL(blob);
                    }} catch (e) {{ console.log("等待流数据..."); }}
                }}
                setInterval(updateFrame, 150); 
            </script>
        </body>
        </html>
        """
        components.html(video_html, height=520)

    with col_right:
        st.markdown("#### 📸 监控控制台")
        if st.button("🖼️ 一键保存当前画面"):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(save_dir, f"pest_det_{timestamp}.jpg")
            try:
                # ✅ 加入 Header 避免拦截
                response = requests.get(snapshot_url, headers={{"ngrok-skip-browser-warning": "any"}}, timeout=5)
                if response.status_code == 200:
                    with open(file_path, "wb") as f: f.write(response.content)
                    st.success(f"✅ 已保存")
            except: st.error("网络请求失败")

        st.markdown("---")
        st.markdown("#### 📈 实时目标计数")
        chart_html = f"""
        <div class="chart-container" style="background: white; padding: 10px; border-radius: 8px; border: 1px solid #ddd; height: 280px;"><canvas id="pestChart"></canvas></div>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            var ctx = document.getElementById('pestChart').getContext('2d');
            var pestChart = new Chart(ctx, {{
                type: 'line',
                data: {{ labels: [], datasets: [{{ label: '检出数量', data: [], borderColor: '#FF4B4B', fill: true, tension: 0.3 }}] }},
                options: {{ maintainAspectRatio: false, animation: false }}
            }});
            setInterval(() => {{
                // ✅ 加入 Header 避免拦截
                fetch('{count_url}', {{ headers: {{ "ngrok-skip-browser-warning": "any" }} }})
                    .then(r => r.json())
                    .then(data => {{
                        var now = new Date();
                        var timeStr = now.getSeconds() + 's';
                        if(pestChart.data.labels.length > 20) {{ pestChart.data.labels.shift(); pestChart.data.datasets[0].data.shift(); }}
                        pestChart.data.labels.push(timeStr);
                        pestChart.data.datasets[0].data.push(data.count);
                        pestChart.update();
                    }});
            }}, 1000);
        </script>
        """
        components.html(chart_html, height=350)

# ==========================================================
# 模式二 & 三
# ==========================================================
def run_classification_mode():
    st.markdown('<div class="app-title"><h1>EIR-CDFS害虫分类</h1></div>', unsafe_allow_html=True)
    st.info("分类功能基于本地权重，不受网络中转影响。")

def run_history_mode():
    st.markdown('<div class="app-title"><h1>历史数据管理</h1></div>', unsafe_allow_html=True)
    log_file = os.path.join(save_dir, "auto_curve_history.json")
    if os.path.exists(log_file): st.success("已加载历史记录。")
    else: st.info("暂无历史记录。")

# 路由控制
if main_task == "平台首页": run_home_mode()
elif main_task == "害虫检测计数": run_detection_mode()
elif main_task == "害虫精确分类": run_classification_mode()
elif main_task == "历史数据管理": run_history_mode()
