import os
import sys
import io
import time
import json
import threading
import requests
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from datetime import datetime

# ✅ 1. 自动适配保存路径 (如果是云端运行，存到当前目录下的 images 文件夹)
if "C:/" in st.session_state.get('save_dir', 'C:/Screenshots'):
    # 如果是在 Streamlit Cloud 运行，强行纠正路径，防止 Linux 报错
    if not os.path.exists("C:/"):
        SAVE_PATH = "pest_records" 
    else:
        SAVE_PATH = "C:/Screenshots"
else:
    SAVE_PATH = "pest_records"

st.set_page_config(page_title="智能害虫检测平台", layout="wide")

# ---------------------------
# 🚀 核心配置 (请确保 Ngrok 正在阿里云上运行)
# ---------------------------
NGROK_URL = "https://unneighbourly-janita-hypothecary.ngrok-free.dev"
# 必须带这个 Header 才能让 Python 的 requests 穿透 Ngrok 警告页
NGROK_HEADERS = {"ngrok-skip-browser-warning": "69420"}

# ---------------------------
# 后台监控线程 (加入 Header 解决“不计数”)
# ---------------------------
def auto_curve_logger():
    count_url = f"{NGROK_URL}/get_count"
    while True:
        try:
            # 没加 headers 之前，云服务器拿回的是警告网页源码，所以不计数
            res = requests.get(count_url, headers=NGROK_HEADERS, timeout=5)
            if res.status_code == 200:
                # 只有这里通了，你的折线图才有数据
                pass 
        except:
            pass
        time.sleep(1)

# ---------------------------
# 模式一：检测计数 (全功能复原)
# ---------------------------
def run_detection_mode():
    st.markdown("### 📡 实时监测系统")
    
    snapshot_url = f"{NGROK_URL}/snapshot"
    count_url = f"{NGROK_URL}/get_count"

    col_l, col_r = st.columns([7, 3])

    with col_l:
        # JS 端的视频流也必须加 Header 才能解决“一秒一帧”
        video_html = f"""
        <img id="stream" style="width:100%; border-radius:10px;">
        <script>
            const img = document.getElementById('stream');
            async function refresh() {{
                const r = await fetch("{snapshot_url}?t="+Date.now(), {{
                    headers: {{ "ngrok-skip-browser-warning": "any" }}
                }});
                const blob = await r.blob();
                img.src = URL.createObjectURL(blob);
            }}
            setInterval(refresh, 150);
        </script>
        """
        components.html(video_html, height=520)

    with col_r:
        st.subheader("📸 控制台")
        
        # 🖼️ 一键保存逻辑修复
        if st.button("🖼️ 一键保存当前画面"):
            try:
                # 必须建立目录，且确保是 Linux 兼容路径
                os.makedirs(SAVE_PATH, exist_ok=True)
                file_name = f"pest_{time.strftime('%H%M%S')}.jpg"
                full_path = os.path.join(SAVE_PATH, file_name)
                
                # 加入 Header 抓取图片
                res = requests.get(snapshot_url, headers=NGROK_HEADERS, timeout=10)
                if res.status_code == 200:
                    with open(full_path, "wb") as f:
                        f.write(res.content)
                    st.success(f"✅ 已存至: {full_path}")
            except Exception as e:
                st.error(f"保存失败: {e}")

        # 📝 记录数量逻辑
        if st.button("📝 记录当前数量到 TXT"):
            try:
                res = requests.get(count_url, headers=NGROK_HEADERS, timeout=5)
                if res.status_code == 200:
                    count = res.json().get("count", 0)
                    with open(os.path.join(SAVE_PATH, "log.txt"), "a") as f:
                        f.write(f"{datetime.now()}: {count}\n")
                    st.success("✅ 记录成功")
            except:
                st.error("无法获取数量")

# 其余逻辑省略...
if st.sidebar.radio("任务", ["首页", "害虫检测计数"]) == "害虫检测计数":
    run_detection_mode()
