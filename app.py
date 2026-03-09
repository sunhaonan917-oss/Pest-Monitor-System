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
import numpy as np
from PIL import Image

# ✅ 必须是第一个 Streamlit 命令
st.set_page_config(page_title="智能害虫检测平台", layout="wide")

# 确保路径
APP_ROOT = os.path.abspath(os.path.dirname(__file__))
if APP_ROOT not in sys.path: sys.path.insert(0, APP_ROOT)

from src.model_builder import build_model, load_checkpoint
from src.infer import predict_5way5shot_one_query
from src.preprocess import load_image_to_tensor

# ---------------------------
# 🚀 核心配置：请确认你的 Ngrok 网址
# ---------------------------
# 注意：末尾不要带斜杠
NGROK_URL = "https://unneighbourly-janita-hypothecary.ngrok-free.dev"

# ✅ 破解 Ngrok 警告页的“免死金牌”
NGROK_HEADERS = {"ngrok-skip-browser-warning": "69420"}

# ---------------------------
# 固定参数 & UI 样式
# ---------------------------
CKPT_PATH = "checkpoints/199.tar"
BACKBONE_NAME = "ResNet10_EMA"
USE_GPU = True

if 'save_dir' not in st.session_state:
    st.session_state.save_dir = "C:/Screenshots"
global_config = {"save_dir": st.session_state.save_dir}

st.markdown("""
    <style>
    .app-title { padding: 14px 16px; border-radius: 12px; background: linear-gradient(90deg, #e8f4ff, #f6fff1); border: 1px solid rgba(0,0,0,0.06); margin-bottom: 14px; }
    label[data-baseweb="radio"] { align-items: center !important; margin-bottom: 16px !important; }
    section[data-testid="stSidebar"] .stRadio p { font-size: 20px !important; font-weight: 600 !important; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# 后台守护线程：恢复自动记录功能
# ---------------------------
def auto_curve_logger():
    count_url = f"{NGROK_URL}/get_count"
    buffer_20s = []
    last_save_minute = -1
    while True:
        try:
            # ✅ 加入 headers 破解拦截
            res = requests.get(count_url, headers=NGROK_HEADERS, timeout=5)
            if res.status_code == 200:
                count = res.json().get("count", 0)
                buffer_20s.append({"时间": datetime.now().strftime("%H:%M:%S"), "检出数量": count})
                if len(buffer_20s) > 20: buffer_20s.pop(0)

            current_time = datetime.now()
            if current_time.second == 0 and current_time.minute != last_save_minute and len(buffer_20s) > 0:
                os.makedirs(global_config["save_dir"], exist_ok=True)
                log_file = os.path.join(global_config["save_dir"], "auto_curve_history.json")
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"), "data": buffer_20s.copy()}, ensure_ascii=False) + "\n")
                last_save_minute = current_time.minute
        except: pass
        time.sleep(1)

if 'logger_thread_started' not in st.session_state:
    threading.Thread(target=auto_curve_logger, daemon=True).start()
    st.session_state.logger_thread_started = True

# --- 侧边栏 ---
st.sidebar.markdown('## 🐛 草地贪夜蛾监测平台')
main_task = st.sidebar.radio("选择功能模式：", ["平台首页", "害虫检测计数", "害虫精确分类", "历史数据管理"], index=1)
st.session_state.save_dir = st.sidebar.text_input("截图与数据保存目录", value=st.session_state.save_dir)
global_config["save_dir"] = st.session_state.save_dir

# ==========================================================
# 模式一：实时目标检测 (✅ 功能完全复原 + 修正语法)
# ==========================================================
def run_detection_mode():
    st.markdown('<div class="app-title"><h1>DPC-DINO 实时监测</h1></div>', unsafe_allow_html=True)
    
    snapshot_url = f"{NGROK_URL}/snapshot"
    count_url = f"{NGROK_URL}/get_count"

    col_left, col_right = st.columns([7, 3])

    with col_left:
        st.markdown("#### 📡 实时监控视窗")
        # ✅ 修正了双大括号转义，彻底解决崩溃问题
        video_html = f"""
        <div style="background-color: #111; border-radius: 8px; height: 520px;">
            <img id="live_stream" style="width: 100%; height: 100%; object-fit: contain;">
        </div>
        <script>
            const img = document.getElementById('live_stream');
            async function update() {{
                try {{
                    const r = await fetch("{snapshot_url}?t=" + new Date().getTime(), {{
                        headers: {{ "ngrok-skip-browser-warning": "any" }}
                    }});
                    const blob = await r.blob();
                    img.src = URL.createObjectURL(blob);
                }} catch (e) {{ }}
            }}
            setInterval(update, 150);
        </script>
        """
        components.html(video_html, height=520)

    with col_right:
        st.markdown("#### 📸 监控控制台")

        # 🖼️ 复原：保存按钮
        if st.button("🖼️ 一键保存当前画面"):
            save_path = os.path.join(global_config["save_dir"], f"pest_det_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
            try:
                response = requests.get(snapshot_url, headers=NGROK_HEADERS, timeout=10)
                if response.status_code == 200:
                    with open(save_path, "wb") as f: f.write(response.content)
                    st.success("✅ 已保存")
            except: st.error("截屏出错, 请检查网络。")

        # 📝 复原：记录按钮
        if st.button("📝 记录当前数量到 TXT"):
            txt_path = os.path.join(global_config["save_dir"], "pest_count_log.txt")
            try:
                res = requests.get(count_url, headers=NGROK_HEADERS, timeout=5)
                if res.status_code == 200:
                    count = res.json().get("count", 0)
                    with open(txt_path, "a", encoding="utf-8") as f:
                        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 数量: {count} 只\n")
                    st.success("✅ 已记录")
            except: st.error("记录失败")

        st.markdown("---")
        st.markdown("#### 📈 实时目标计数")
        # ✅ 修正语法并加入 Header 破解
        chart_html = f"""
        <canvas id="pestChart" style="background: white; border: 1px solid #ddd; height: 280px;"></canvas>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            var ctx = document.getElementById('pestChart').getContext('2d');
            var pestChart = new Chart(ctx, {{
                type: 'line',
                data: {{ labels: [], datasets: [{{ label: '检出数量', data: [], borderColor: '#FF4B4B', tension: 0.3 }}] }},
                options: {{ maintainAspectRatio: false, animation: false }}
            }});
            setInterval(() => {{
                fetch('{count_url}', {{ headers: {{ "ngrok-skip-browser-warning": "any" }} }})
                    .then(r => r.json())
                    .then(data => {{
                        if(pestChart.data.labels.length > 20) {{ pestChart.data.labels.shift(); pestChart.data.datasets[0].data.shift(); }}
                        pestChart.data.labels.push(new Date().getSeconds() + 's');
                        pestChart.data.datasets[0].data.push(data.count);
                        pestChart.update();
                    }});
            }}, 1000);
        </script>
        """
        components.html(chart_html, height=350)

# 其他模式模块占位
if main_task == "平台首页": st.write("欢迎回来！")
elif main_task == "害虫检测计数": run_detection_mode()
elif main_task == "害虫精确分类": st.write("分类模块已就绪")
elif main_task == "历史数据管理": st.write("历史数据正常归档中")
