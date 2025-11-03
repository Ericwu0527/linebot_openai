from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

import os
import time
import traceback
import math
import sqlite3
import json

# 引入 Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ======================= 基本設定 =======================
DB_FILE = "knowledge_base.db"
RAG_CONFIDENCE_THRESHOLD = 1.5  # 放寬門檻
RESET_DB = True  # ✅ 首次部署時設定 True，初始化後改回 False
# =========================================================

# 🔹 如果設定為 True，自動刪除舊資料庫
if RESET_DB and os.path.exists(DB_FILE):
    os.remove(DB_FILE)
    print("🗑 已刪除舊的 knowledge_base.db，將重新建立。")

# 初始化 Flask
app = Flask(__name__)

# LINE Bot 設定
line_bot_api = LineBotApi(os.getenv("CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("CHANNEL_SECRET"))

# Gemini 初始化
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("⚠️ 未設定 GEMINI_API_KEY 環境變數")

try:
    client = genai.Client()
except Exception as e:
    print(f"初始化 Gemini 客戶端失敗: {e}")
    client = None

# ======================= SQLite 相關 =======================
def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            embedding_json TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("✅ SQLite 資料庫設定完成。")


def euclidean_distance(vec1, vec2):
    if len(vec1) != len(vec2):
        return float("inf")
    return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vec1, vec2)))


def get_embedding(text):
    """取得文字的向量"""
    if not client:
        return None
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=[text],
        )
        return result.embeddings[0].values  # ✅ 正確格式
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None


def initialize_knowledge_base():
    """初始化預設知識"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM knowledge_base")
    count = cursor.fetchone()[0]

    if count == 0:
        print("🔧 初始化 RAG 知識庫中...")
        default_data = [
            "本公司的營業時間是週一至週五，早上九點到下午六點。",
            "退貨政策：非特價商品可在購買後30天內憑發票退貨。",
            "技術支援請發送電子郵件至 support@mycompany.com。",
            "工作考成分數是多少？工作考成分數為 6.5 分。",
            "績效考評由部門主管負責，每年進行兩次。"
        ]
        for content in default_data:
            embedding = get_embedding(content)
            if embedding:
                cursor.execute(
                    "INSERT INTO knowledge_base (content, embedding_json) VALUES (?, ?)",
                    (content, json.dumps(embedding)),
                )
        conn.commit()
        print("✅ RAG 知識庫初始化完成。")
    conn.close()


def add_new_knowledge(content):
    """新增知識到資料庫"""
    embedding = get_embedding(content)
    if not embedding:
        print(f"[Error] 無法為內容生成 Embedding: {content[:30]}")
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO knowledge_base (content, embedding_json) VALUES (?, ?)",
        (content, json.dumps(embedding)),
    )
    conn.commit()
    conn.close()
    print(f"✅ 成功新增知識: {content[:30]}...")


def query_knowledge_base(query_text, top_k=3):
    """檢索知識庫"""
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return "", False

    results = []
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT content, embedding_json FROM knowledge_base")
    rows = cursor.fetchall()
    conn.close()

    for row in rows:
        content = row["content"]
        item_embedding = json.loads(row["embedding_json"])
        distance = euclidean_distance(query_embedding, item_embedding)
        results.append((distance, content))

    results.sort(key=lambda x: x[0])

    print(f"\n[RAG DEBUG] 查詢: {query_text}")
    for d, c in results[:3]:
        print(f"  距離 {d:.4f} → {c}")

    is_high_confidence = results and results[0][0] < RAG_CONFIDENCE_THRESHOLD
    context = "\n".join([c for _, c in results[:top_k]])

    if is_high_confidence:
        print("[RAG] 命中高信心資料庫內容 ✅")

    return context, is_high_confidence


# ======================= Gemini 回覆 =======================
def GEMINI_response(user_text):
    if not client:
        return "⚠️ Gemini 客戶端未成功初始化。"

    rag_context, is_high_confidence = query_knowledge_base(user_text, top_k=3)

    if rag_context:
        if is_high_confidence:
            system_instruction = (
                "你是一位客服助理，必須且只能根據以下 CONTEXT 回答問題，"
                "不得使用外部資訊。若無法回答，請說明資料不足。\n"
                f"CONTEXT:\n---\n{rag_context}\n---"
            )
            tools_config = []
        else:
            system_instruction = (
                "你是一位客服助理，請優先使用 CONTEXT 回答問題，"
                "若 CONTEXT 無法回答，可使用一般知識搜尋。\n"
                f"CONTEXT:\n---\n{rag_context}\n---"
            )
            tools_config = [{"google_search": {}}]
    else:
        system_instruction = "你是一位助理，請使用一般知識回答問題。"
        tools_config = [{"google_search": {}}]

    config = types.GenerateContentConfig(
        temperature=0.5,
        max_output_tokens=1500,
        tools=tools_config,
        system_instruction=system_instruction,
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_text,
            config=config,
        )
        return response.text.strip() if response.text else "⚠️ 未獲得回覆。"
    except Exception as e:
        print(traceback.format_exc())
        return f"⚠️ 發生錯誤：{e}"


# ======================= Flask 路由 =======================
@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    print(f"[User Message]: {user_msg}")
    reply_text = GEMINI_response(user_msg)
    print(f"[Gemini Reply]: {reply_text}")
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))


@app.route("/resetdb", methods=["GET"])
def reset_db():
    """🔧 一鍵重建資料庫（Render用）"""
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        print("🗑 已刪除舊 knowledge_base.db")
    setup_db()
    initialize_knowledge_base()
    return "✅ 資料庫已重建完成。"


# ======================= 啟動 Flask =======================
if __name__ == "__main__":
    setup_db()
    initialize_knowledge_base()

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
