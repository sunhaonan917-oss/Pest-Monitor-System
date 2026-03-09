import os
import sys
import io
import time
import json
import threading
import requests
import pandas as pd
# ✅ 确保引入北京时间计算工具
from datetime import datetime, timedelta
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

# ✅ Ngrok 网址与防拦截配置
NGROK_URL = "https://unneighbourly-janita-hypothecary.ngrok-free.dev"
NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "any",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 自动在当前目录下建立文件夹
SAFE_SAVE_DIR = os.path.join(APP_ROOT, "pest_records")
os.makedirs(SAFE_SAVE_DIR, exist_ok=True)

# 【关键点】作为前端网页和后台线程通信的桥梁
global_config = {"save_dir": SAFE_SAVE_DIR}

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
# 后台隐形守护线程：每分钟自动保存过去20秒的曲线
# ==========================================================
def auto_curve_logger():
    count_url = f"{NGROK_URL}/get_count"
    buffer_20s = []
    last_save_minute = -1

    while True:
        current_save_dir = global_config["save_dir"]

        try:
            res = requests.get(count_url, headers=NGROK_HEADERS, timeout=5)
            
            # ✅ [核心修改1]：强制获取北京时间（UTC+8）
            beijing_now = datetime.utcnow() + timedelta(hours=8)

            if res.status_code == 200:
                count = res.json().get("count", 0)
                # 使用北京时间格式化字符串
                now_str = beijing_now.strftime("%H:%M:%S")
                buffer_20s.append({"时间": now_str, "检出数量": count})

                if len(buffer_20s) > 20:
                    buffer_20s.pop(0)

            # ✅ [核心修改2]：保存逻辑。要求秒数在0-5之间，分钟数变化，且缓存充足
            if beijing_now.second <= 5 and beijing_now.minute != last_save_minute and len(buffer_20s) >= 10:
                os.makedirs(current_save_dir, exist_ok=True)
                log_file = os.path.join(current_save_dir, "auto_curve_history.json")

                record = {
                    "timestamp": beijing_now.strftime("%Y-%m-%d %H:%M:%S"),
                    "data": buffer_20s.copy()
                }
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                last_save_minute = beijing_now.minute  

        except Exception as e:
            pass  

        time.sleep(1)  


if 'logger_thread_started' not in st.session_state:
    t = threading.Thread(target=auto_curve_logger, daemon=True)
    t.start()
    st.session_state.logger_thread_started = True


# ---------------------------
# 工具函数 & Sidebar
# ---------------------------
def bytes_to_filelike(b: bytes): return io.BytesIO(b)
def bytes_to_pil(b: bytes): return Image.open(io.BytesIO(b)).convert("RGB")

@st.cache(allow_output_mutation=True)
def load_model_cached(ckpt_path: str, device_str: str, backbone_name: str):
    dev = torch.device(device_str)
    model = build_model(n_way=5, n_support=5, backbone_name=backbone_name)
    load_checkpoint(model, ckpt_path, dev)
    return model

st.sidebar.markdown('## 🐛 草地贪夜蛾监测平台')
main_task = st.sidebar.radio("选择功能模式：", ["平台首页", "害虫检测计数", "害虫精确分类", "历史数据管理"], index=0)
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 历史数据与存储配置")
st.session_state.save_dir = st.sidebar.text_input("截图与数据保存目录", value=st.session_state.save_dir)
global_config["save_dir"] = SAFE_SAVE_DIR
save_dir = SAFE_SAVE_DIR
device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")

# ==========================================================
# 模式一：实时目标检测 (JS端也修正为本地时间)
# ==========================================================
def run_detection_mode():
    st.markdown('<div class="app-title"><h1>DPC-DINO 实时监测</h1><p>正在接收来自边缘端 Jetson Orin Nano 的无损实时流</p></div>', unsafe_allow_html=True)
    
    snapshot_url = f"{NGROK_URL}/snapshot"
    count_url = f"{NGROK_URL}/get_count"

    col_left, col_right = st.columns([7, 3])

    with col_left:
        st.markdown("#### 📡 实时监控视窗")
        video_html = f"""
        <html><body style="margin:0;background:#111;display:flex;justify-content:center;">
            <img id="live" style="width:100%;min-height:520px;object-fit:contain;">
            <script>
                setInterval(() => {{ document.getElementById('live').src = "{snapshot_url}?t=" + Date.now(); }}, 150);
            </script>
        </body></html>
        """
        components.html(video_html, height=520)

    with col_right:
        st.markdown("#### 📸 监控控制台")
        # 🟢 前端下载按钮 (JS获取的是用户电脑本地时间，通常是准确的)
        download_html = f"""
        <html><body style="margin:0;background:transparent;">
            <style>.btn {{ width:100%; padding:10px; margin-bottom:10px; cursor:pointer; background:#fff; border:1px solid #ddd; border-radius:8px; font-family:sans-serif; }} .btn:hover {{ border-color:#FF4B4B; color:#FF4B4B; }}</style>
            <button class="btn" onclick="saveImg()">🖼️ 一键保存当前画面</button>
            <button class="btn" onclick="saveTxt()">📝 记录当前数量到 TXT</button>
            <script>
                async function saveImg() {{
                    const res = await fetch("{snapshot_url}?t=" + Date.now(), {{ headers: {{ "ngrok-skip-browser-warning": "any" }} }});
                    const blob = await res.blob();
                    const a = document.createElement('a');
                    a.href = URL.createObjectURL(blob);
                    // JS获取本地时间命名
                    const d = new Date();
                    const ts = d.getFullYear() + ('0'+(d.getMonth()+1)).slice(-2) + ('0'+d.getDate()).slice(-2) + '_' + ('0'+d.getHours()).slice(-2) + ('0'+d.getMinutes()).slice(-2);
                    a.download = 'pest_' + ts + '.jpg';
                    a.click();
                }}
                async function saveTxt() {{
                    const res = await fetch("{count_url}", {{ headers: {{ "ngrok-skip-browser-warning": "any" }} }});
                    const data = await res.json();
                    const d = new Date().toLocaleString('zh-CN');
                    const blob = new Blob(["[" + d + "] 数量: " + data.count], {{ type: 'text/plain' }});
                    const a = document.createElement('a');
                    a.href = URL.createObjectURL(blob);
                    a.download = 'count_' + Date.now() + '.txt';
                    a.click();
                }}
            </script>
        </body></html>
        """
        components.html(download_html, height=130)

        st.markdown("---")
        st.markdown("#### 📈 实时目标计数")
        chart_html = f"""
        <html><head><script src="https://cdn.jsdelivr.net/npm/chart.js"></script></head>
        <body><div style="height:280px;"><canvas id="pestChart"></canvas></div>
            <script>
                var ctx = document.getElementById('pestChart').getContext('2d');
                var pestChart = new Chart(ctx, {{
                    type: 'line', data: {{ labels: [], datasets: [{{ label: '检出数量', data: [], borderColor: '#FF4B4B', tension: 0.3 }}] }},
                    options: {{ maintainAspectRatio: false, animation: false }}
                }});
                setInterval(() => {{
                    fetch('{count_url}', {{ headers: {{ "ngrok-skip-browser-warning": "any" }} }})
                        .then(r => r.json()).then(d => {{
                            let s = new Date().getSeconds() + 's';
                            if(pestChart.data.labels.length > 20) {{ pestChart.data.labels.shift(); pestChart.data.datasets[0].data.shift(); }}
                            pestChart.data.labels.push(s); pestChart.data.datasets[0].data.push(d.count);
                            pestChart.update();
                        }});
                }}, 1000);
            </script>
        </body></html>
        """
        components.html(chart_html, height=350)

# ==========================================================
# 模式三：历史数据管理 (展示北京时间记录)
# ==========================================================
def run_history_mode():
    st.markdown('<div class="app-title"><h1>历史数据管理</h1><p>查看历史自动检测趋势记录</p></div>', unsafe_allow_html=True)
    log_file = os.path.join(save_dir, "auto_curve_history.json")
    if os.path.exists(log_file):
        records = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): records.append(json.loads(line))
        if records:
            st.success(f"✅ 已找到 **{len(records)}** 组记录（北京时间）。")
            # 倒序展示最新的
            options = [r["timestamp"] for r in reversed(records)]
            selected = st.selectbox("⏳ 选择回溯时间点:", options)
            for r in records:
                if r["timestamp"] == selected:
                    df = pd.DataFrame(r["data"])
                    df.set_index("时间", inplace=True)
                    st.markdown(f"#### 📊 {selected} 趋势图")
                    st.line_chart(df, height=350)
                    break
        else:
            st.info("🕒 记录为空，请等待下个整分自动保存。")
    else:
        st.info("📂 暂无历史记录文件。")

# 路由分发
if main_task == "平台首页": st.write("欢迎来到首页")
elif main_task == "害虫检测计数": run_detection_mode()
elif main_task == "历史数据管理": run_history_mode()
