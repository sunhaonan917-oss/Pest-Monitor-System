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

# ✅ [唯一修改的地方]：在这里加入了 User-Agent，把代码伪装成真实的 Google Chrome 浏览器，Ngrok 就不会拦截了！
NGROK_URL = "https://unneighbourly-janita-hypothecary.ngrok-free.dev"
NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "any",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# 原变量保留不动，但在下方拼接时弃用，避免生成错乱URL
JETSON_IP = "100.104.20.74"
JETSON_PORT = "5000"

# ✅ 自动在当前目录下建立文件夹，彻底解决 C盘 权限报错
SAFE_SAVE_DIR = os.path.join(APP_ROOT, "pest_records")
os.makedirs(SAFE_SAVE_DIR, exist_ok=True)

# 【关键点】作为前端网页和后台线程通信的桥梁
global_config = {"save_dir": SAFE_SAVE_DIR}

# ---------------------------
# CSS 样式 (100% 保持你原来的)
# ---------------------------
st.markdown("""
    <style>
    .app-title { padding: 14px 16px; border-radius: 12px; background: linear-gradient(90deg, #e8f4ff, #f6fff1); border: 1px solid rgba(0,0,0,0.06); margin-bottom: 14px; }
    .app-title h1 { margin: 0; font-size: 24px; }
    .app-title p { margin: 6px 0 0 0; color: rgba(0,0,0,0.6); }
    .card { padding: 12px 14px; border-radius: 12px; border: 1px solid rgba(0,0,0,0.06); background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.04); margin-bottom: 12px; }

    /* 👇 修复侧边栏单选按钮垂直不对齐的问题 */
    label[data-baseweb="radio"] {
        align-items: center !important; /* 强制红点和文字垂直居中对齐 */
        margin-bottom: 16px !important; /* 用外边距拉开选项之间的距离，不再挤在一起 */
    }

    section[data-testid="stSidebar"] .stRadio p,
    div[role="radiogroup"] p,
    .stRadio label p {
        font-size: 20px !important;  
        font-weight: 600 !important; 
        line-height: normal !important; /* 恢复正常行高，防止文字隐形框变形 */
        margin-top: 2px !important;     /* 微调文字重心，使其与红点绝对水平 */
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
    st.session_state.save_dir = SAFE_SAVE_DIR


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
            if res.status_code == 200:
                count = res.json().get("count", 0)
                now_str = datetime.now().strftime("%H:%M:%S")
                buffer_20s.append({"时间": now_str, "检出数量": count})

                if len(buffer_20s) > 20:
                    buffer_20s.pop(0)

            current_time = datetime.now()
            if current_time.second == 0 and current_time.minute != last_save_minute and len(buffer_20s) > 0:
                os.makedirs(current_save_dir, exist_ok=True)
                log_file = os.path.join(current_save_dir, "auto_curve_history.json")

                record = {
                    "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "data": buffer_20s.copy()
                }
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                last_save_minute = current_time.minute  

        except Exception as e:
            pass  

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


@st.cache(allow_output_mutation=True)
def load_model_cached(ckpt_path: str, device_str: str, backbone_name: str):
    dev = torch.device(device_str)
    model = build_model(n_way=5, n_support=5, backbone_name=backbone_name)
    load_checkpoint(model, ckpt_path, dev)
    return model


# ---------------------------
# Sidebar 侧边栏
# ---------------------------
st.sidebar.markdown('## 🐛 草地贪夜蛾监测平台')

main_task = st.sidebar.radio(
    "选择功能模式：",
    ["平台首页", "害虫检测计数", "害虫精确分类", "历史数据管理"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 历史数据与存储配置")
st.session_state.save_dir = st.sidebar.text_input("截图与数据保存目录", value=st.session_state.save_dir)

global_config["save_dir"] = st.session_state.save_dir
save_dir = st.session_state.save_dir

device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")

# ==========================================================
# 模式零：平台首页
# ==========================================================
def run_home_mode():
    st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 75vh; text-align: center;">
            <h1 style="font-size: 42px; color: #2c3e50; margin-bottom: 20px;">🐛 欢迎使用草地贪夜蛾智能监测平台</h1>
            <div style="background-color: #f8f9fa; padding: 40px; border-radius: 15px; border: 1px solid #e0e0e0; max-width: 800px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <p style="font-size: 18px; color: #555; line-height: 1.8; margin-bottom: 20px;">
                    本平台基于<b>端边协同架构</b>开发，致力于为非结构化农田环境提供高效、精准的病虫害监测解决方案。系统深度集成了面向复杂场景的深度学习核心算法：
                </p>
                <ul style="text-align: left; font-size: 16px; color: #444; line-height: 1.8; margin-bottom: 30px; display: inline-block;">
                    <li>📡 <b>DPC-DINO 目标检测：</b> 部署于边缘节点，实现实时视频流监控与目标捕获。</li>
                    <li>🔬 <b>EIR-CDFS 精确分类：</b> 部署于中心网页端，完成高精度的小样本害虫特征识别。</li>
                    <li>📊 <b>自动化数据归档：</b> 隐形守护线程周期性收割检测数据，构建历史趋势溯源体系。</li>
                </ul>
                <p style="font-size: 18px; font-weight: bold; color: #007bff;">
                    👈 请在左侧菜单栏选择您需要的功能模块开始体验
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)


# ==========================================================
# 模式一：实时目标检测
# ==========================================================
def run_detection_mode():
    st.markdown('<div class="app-title"><h1>DPC-DINO 实时监测</h1><p>正在接收来自边缘端 Jetson Orin Nano 的无损实时流</p></div>',
                unsafe_allow_html=True)

    stream_url = f"{NGROK_URL}/video_feed"
    snapshot_url = f"{NGROK_URL}/snapshot"
    count_url = f"{NGROK_URL}/get_count"

    st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #007bff; margin-bottom: 20px;">
            <h4 style="margin-top: 0; color: #333;">📡 边缘节点流媒体初始化</h4>
            <p style="color: #555; font-size: 14px;">系统检测到边缘端处于待机休眠状态。请 <b><a href="{stream_url}" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;">[ 点击此处激活 Jetson 硬件通道 ]</a></b>。<br>
            <span style="font-size: 13px; color: #888;">* 激活后请保留该安全通道页，返回本监控中心即可接入无损实时流。</span></p>
        </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([7, 3])

    with col_left:
        st.markdown("#### 📡 实时监控视窗")
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
            <img id="live_stream" alt="正在建立安全连接，拉取实时视频流...">
            <script>
                var liveStream = document.getElementById('live_stream');
                setInterval(() => {{
                    fetch("{snapshot_url}?t=" + new Date().getTime(), {{
                        headers: {{ "ngrok-skip-browser-warning": "any" }}
                    }})
                    .then(response => response.blob())
                    .then(blob => {{
                        liveStream.src = URL.createObjectURL(blob);
                    }})
                    .catch(err => console.log('等待边缘端响应...'));
                }}, 150); 
            </script>
        </body>
        </html>
        """
        components.html(video_html, height=520)

    with col_right:
        st.markdown("#### 📸 监控控制台")

        # 🟢 前端下载按钮，图片和数量直接下载到本机
        download_buttons_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .btn {{
                    display: inline-flex; align-items: center; justify-content: center;
                    width: 100%; padding: 0.5rem 1rem; margin-bottom: 15px;
                    background-color: white; border: 1px solid rgba(49, 51, 63, 0.2);
                    border-radius: 0.5rem; color: rgb(49, 51, 63);
                    font-size: 1rem; font-weight: 400; cursor: pointer;
                    text-decoration: none; font-family: "Source Sans Pro", sans-serif;
                    transition: all 0.2s ease;
                }}
                .btn:hover {{ border-color: rgb(255, 75, 75); color: rgb(255, 75, 75); }}
            </style>
        </head>
        <body style="margin: 0; padding: 0; background-color: transparent;">
            <button class="btn" onclick="saveImage()">🖼️ 一键保存当前画面</button>
            <button class="btn" onclick="saveTxt()">📝 记录当前数量到 TXT</button>

            <script>
                async function saveImage() {{
                    try {{
                        const res = await fetch("{snapshot_url}?t=" + Date.now(), {{
                            headers: {{ "ngrok-skip-browser-warning": "any" }}
                        }});
                        const blob = await res.blob();
                        const a = document.createElement('a');
                        a.href = URL.createObjectURL(blob);
                        const timeStr = new Date().toISOString().replace(/[:.-]/g, '').slice(0, 15);
                        a.download = 'pest_det_' + timeStr + '.jpg';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }} catch(e) {{ alert("下载失败，请确认视频流是否正常。"); }}
                }}

                async function saveTxt() {{
                    try {{
                        const res = await fetch("{count_url}", {{
                            headers: {{ "ngrok-skip-browser-warning": "any" }}
                        }});
                        const data = await res.json();
                        const timeStr = new Date().toLocaleString('zh-CN');
                        const text = "[" + timeStr + "] 发现草地贪夜蛾目标数量: " + data.count + " 只\\n";
                        
                        const blob = new Blob([text], {{ type: 'text/plain;charset=utf-8' }});
                        const a = document.createElement('a');
                        a.href = URL.createObjectURL(blob);
                        const fileTime = new Date().toISOString().replace(/[:.-]/g, '').slice(0, 15);
                        a.download = 'pest_count_' + fileTime + '.txt';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }} catch(e) {{ alert("下载计数失败。"); }}
                }}
            </script>
        </body>
        </html>
        """
        components.html(download_buttons_html, height=130)

        st.markdown("---")
        st.markdown("#### 📈 实时目标计数")

        chart_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ margin: 0; padding: 0; background-color: transparent; }}
                .chart-container {{ background: white; padding: 10px; border-radius: 8px; border: 1px solid #ddd; height: 280px; width: 100%; box-sizing: border-box; }}
            </style>
        </head>
        <body>
            <div class="chart-container">
                <canvas id="pestChart"></canvas>
            </div>
            <script>
                var ctx = document.getElementById('pestChart').getContext('2d');
                var pestChart = new Chart(ctx, {{
                    type: 'line',
                    data: {{
                        labels: [],
                        datasets: [{{
                            label: '检出数量',
                            data: [],
                            borderColor: '#FF4B4B',
                            backgroundColor: 'rgba(255, 75, 75, 0.1)',
                            borderWidth: 2, fill: true, tension: 0.3, pointRadius: 2
                        }}]
                    }},
                    options: {{ 
                        maintainAspectRatio: false, 
                        animation: false,
                        scales: {{ 
                            y: {{ beginAtZero: true, suggestedMax: 5, ticks: {{ stepSize: 1 }} }},
                            x: {{ display: true }}
                        }} 
                    }}
                }});

                setInterval(() => {{
                    fetch('{count_url}', {{ headers: {{ "ngrok-skip-browser-warning": "any" }} }})
                        .then(response => response.json())
                        .then(data => {{
                            var now = new Date();
                            var timeStr = now.getSeconds() + 's';
                            if(pestChart.data.labels.length > 30) {{
                                pestChart.data.labels.shift();
                                pestChart.data.datasets[0].data.shift();
                            }}
                            pestChart.data.labels.push(timeStr);
                            pestChart.data.datasets[0].data.push(data.count);
                            pestChart.update();
                        }})
                        .catch(err => console.log('等待边缘端响应...'));
                }}, 1000);
            </script>
        </body>
        </html>
        """
        components.html(chart_html, height=350)


# ==========================================================
# 模式二：小样本分类逻辑
# ==========================================================
def run_classification_mode():
    st.markdown('<div class="app-title"><h1>EIR-CDFS害虫分类</h1><p>本地计算资源推理</p></div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    page = st.sidebar.radio("分类操作步骤", ["① 上传支持集", "② 上传需分类图片"])

    if not os.path.exists(CKPT_PATH):
        st.error(f"找不到权重文件：{CKPT_PATH}")
        return
    model = load_model_cached(CKPT_PATH, str(device), BACKBONE_NAME)

    if page == "① 上传支持集":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("第1步：自定义类别名")
        for i in range(5):
            st.session_state.class_names[i] = st.text_input(f"类别 {i}", st.session_state.class_names[i], key=f"cn_{i}")

        st.subheader("第2步：上传参考图 (每类5张)")
        for i in range(5):
            with st.expander(f"上传 {st.session_state.class_names[i]}"):
                files = st.file_uploader("选择图片", type=["png", "jpg"], accept_multiple_files=True, key=f"up_{i}")
                if files: st.session_state.support_bytes[i] = [f.getvalue() for f in files[:5]]
                imgs = st.session_state.support_bytes[i]
                if imgs: st.image([bytes_to_pil(b) for b in imgs], width=120)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("第3步：上传待识别图片")
        q = st.file_uploader("上传 query", type=["png", "jpg"], key="qu")
        if q: st.session_state.query_bytes = q.getvalue()
        if st.session_state.query_bytes:
            st.image(bytes_to_pil(st.session_state.query_bytes), caption="待识别图", width=400)
            if st.button("开始推理"):
                support_tensors = []
                support_labels = []
                for i in range(5):
                    for b in st.session_state.support_bytes[i]:
                        support_tensors.append(load_image_to_tensor(bytes_to_filelike(b)))
                        support_labels.append(i)
                if len(support_tensors) < 25:
                    st.error("Support 数据不足：每类必须上传 5 张图片。")
                else:
                    with st.spinner("推理中..."):
                        x_s = torch.stack(support_tensors).to(device)
                        y_s = torch.tensor(support_labels).to(device)
                        x_q = load_image_to_tensor(bytes_to_filelike(st.session_state.query_bytes)).unsqueeze(0).to(
                            device)
                        logits = predict_5way5shot_one_query(model, x_s, y_s, x_q, device)
                        res = torch.argmax(logits).item()
                        st.success(f"识别结果：{st.session_state.class_names[res]}")
        st.markdown('</div>', unsafe_allow_html=True)


# ==========================================================
# 模式三：历史数据洞察 (去除了画廊，完美保留折线图分析)
# ==========================================================
def run_history_mode():
    st.markdown('<div class="app-title"><h1>历史数据管理</h1><p>查看历史自动检测趋势记录</p></div>', unsafe_allow_html=True)

    # 选项卡已删除，直接在当前页面展示历史折线图
    log_file = os.path.join(save_dir, "auto_curve_history.json")
    if os.path.exists(log_file):
        records = []
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if records:
            st.success(f"✅ 在本地库中找到 **{len(records)}** 组由后台自动保存的历史曲线记录。")
            # 倒序排列，让最新的记录在最上面
            options = [r["timestamp"] for r in reversed(records)]
            selected_time = st.selectbox("⏳ 请选择要回溯的历史时间节点 (系统每逢整分自动记录):", options)

            # 重绘图表
            for r in records:
                if r["timestamp"] == selected_time:
                    df = pd.DataFrame(r["data"])
                    df.set_index("时间", inplace=True)
                    st.markdown(f"#### 📊 {selected_time} 前 20 秒目标数量走势")
                    # 极其美观的 Streamlit 原生折线图
                    st.line_chart(df, height=350, use_container_width=True)
                    break
        else:
            st.info("🕒 数据记录文件为空。系统启动后，每逢整分（如 12:01:00）会自动保存一次数据，请稍后查看。")
    else:
        st.info(f"📂 暂无历史曲线。系统将在后台静默收集数据，并在整分时存入 {save_dir}/auto_curve_history.json 中。")


# ---------------------------
# 主程序路由
# ---------------------------
if main_task == "平台首页":
    run_home_mode()
elif main_task == "害虫检测计数":
    run_detection_mode()
elif main_task == "害虫精确分类":
    run_classification_mode()
elif main_task == "历史数据管理":
    run_history_mode()
