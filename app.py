from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

import os
import sys
import time
import traceback
import math
import json
from datetime import datetime

# 引入 Firestore 與 GenAI
from google.cloud import firestore
from google.cloud.firestore_v1.base_query as firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ======================= 設定區域 =======================
KNOWLEDGE_COLLECTION = "knowledge_base"
REMINDER_COLLECTION = "reminders"
RAG_CONFIDENCE_THRESHOLD = 0.5

# 初始企業知識
initial_knowledge_data = [
    {"content": "本公司的營業時間是週一至週五，早上九點到下午六點。"},
    {"content": "退貨政策：非特價商品可在購買後30天內憑發票退票。"},
    {"content": "技術支援請發送電子郵件至 support@mycompany.com。"},
    {"content": "114年工作考成分數(立法院提刪通過)為 6.91 分。"},
    {"content": "114年工作考成分數(立法院提刪未通過)為 6.04 分。"},
    {"content": "114年工作考成分數(含不可抗力因素)為 6.46 分。"},
]

app = Flask(__name__)

# Channel Access Token / Secret (從環境變數讀取)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化 Gemini & Firestore
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務初始化成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具定義 (Tools) =======================

def save_reminder_tool(content: str):
    """
    當使用者提到未來的行程、計畫、動作、預約或想記下的事情時（例如：『明天我要去...』、『幫我記下...』），請務必調用此工具。
    參數 content：應包含時間與行程的簡短描述。
    """
    return {"status": "intent_detected", "content": content}

# ======================= 核心邏輯函式 =======================

def cosine_distance(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude_v1 = math.sqrt(sum(v1 * v1 for v1 in vec1))
    magnitude_v2 = math.sqrt(sum(v2 * v2 for v2 in vec2))
    return 1.0 - (dot_product / (magnitude_v1 * magnitude_v2)) if magnitude_v1 != 0 and magnitude_v2 != 0 else 1.0

def get_embedding(text):
    if not client: return None
    try:
        result = client.models.embed_content(model='text-embedding-004', contents=[text])
        return result.embeddings[0].values
    except:
        return None

def record_reminder(user_id, raw_text):
    """將行程寫入 Firestore"""
    print(f"DEBUG: 正在寫入行程 -> {raw_text}", flush=True)
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        return True, f"✅ 已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        print(f"❌ Firestore 寫入失敗: {e}", flush=True)
        return False, f"❌ 記錄失敗：{e}"

def get_user_reminders(user_id):
    """讀取行程 (已修正新版語法警告)"""
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .order_by('recorded_at', direction=firestore.Query.DESCENDING)\
                 .limit(10).stream()
        return [f"• {d.to_dict()['raw_text']}" for d in docs]
    except Exception as e:
        print(f"⚠️ 讀取行程時發生錯誤: {e}", flush=True)
        return []

def query_knowledge_base(query_text):
    """RAG 知識庫檢索"""
    query_embedding = get_embedding(query_text)
    if not query_embedding: return "", False
    results = []
    try:
        docs = db.collection(KNOWLEDGE_COLLECTION).stream()
        for doc in docs:
            data = doc.to_dict()
            if 'embedding_json' in data:
                dist = cosine_distance(query_embedding, json.loads(data['embedding_json']))
                results.append((dist, data['content']))
        results.sort(key=lambda x: x[0])
        if results and results[0][0] < RAG_CONFIDENCE_THRESHOLD:
            return results[0][1], True
    except:
        pass
    return "", False

# ======================= Gemini 回應與分流 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ API 未就緒"

    # 1. 準備上下文
    rag_context, _ = query_knowledge_base(user_text)
    reminders = get_user_reminders(user_id)
    personal_context = "\n".join(reminders) if reminders else "目前無未完成行程。"

    # 2. 強化後的系統指令
    system_instruction = (
        "你是一位行動派助理。請根據以下規則與使用者對話：\n"
        "1. 當使用者提到行程或計畫（例如：『明天要去...』、『等等要去...』）時，請『直接調用』save_reminder_tool，不要問詢問。\n"
        "2. 如果使用者詢問關於公司規定或考成，請根據【企業知識】回答。\n"
        "3. 如果使用者詢問關於他自己的行程，請參考【個人行程】。\n"
        f"【企業知識】：{rag_context}\n"
        f"【個人行程】：{personal_context}"
    )

    try:
        config = types.GenerateContentConfig(
            temperature=0.2, # 降低隨機性，讓意圖更穩定
            tools=[save_reminder_tool],
            system_instruction=system_instruction,
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_text,
            config=config,
        )

        # 3. 處理 Function Calling 邏輯
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    # 抓取 AI 提取出的行程內容
                    fn = part.function_call
                    extracted_text = fn.args.get("content", user_text)
                    success, msg = record_reminder(user_id, extracted_text)
                    return msg
        
        return response.text if response.text else "抱歉，請再說一次。"
    except Exception as e:
        print(f"❌ Gemini 生成錯誤: {e}", flush=True)
        return "⚠️ 服務暫時無法回應，請稍後。"

# ======================= Flask 路由 =======================

@app.route('/')
def index():
    # UptimeRobot 訪問點
    return "✅ LINE Bot is active and running!"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@app.route("/resetdb")
def reset_db():
    """手動重建知識庫"""
    if not db: return "Firestore Error"
    try:
        docs = db.collection(KNOWLEDGE_COLLECTION).list_documents()
        batch = db.batch()
        for doc in docs: batch.delete(doc)
        batch.commit()
        initialize_knowledge_base()
        return "✅ 知識庫已重設"
    except Exception as e:
        return f"Error: {e}"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    user_id = event.source.user_id
    
    # 輸出日誌到 Render
    print(f"\n[LINE User Message]: {user_msg}", flush=True)
    
    reply_text = GEMINI_response(user_msg, user_id)
    
    print(f"[AI Response]: {reply_text}\n", flush=True)
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

# ======================= 啟動作業 =======================

def initialize_knowledge_base():
    """初始化 RAG 資料"""
    try:
        docs = list(db.collection(KNOWLEDGE_COLLECTION).limit(1).get())
        if not docs:
            print("正在初始化企業知識庫...", flush=True)
            for i, item in enumerate(initial_knowledge_data):
                emb = get_embedding(item['content'])
                if emb:
                    db.collection(KNOWLEDGE_COLLECTION).document(f"k_{i}").set({
                        'content': item['content'],
                        'embedding_json': json.dumps(emb)
                    })
    except:
        pass

if __name__ == "__main__":
    initialize_knowledge_base()
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
