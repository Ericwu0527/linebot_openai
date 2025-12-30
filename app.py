from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

import os
import time
import traceback
import math
import json
from datetime import datetime

# 引入 Firestore 與 GenAI
from google.cloud import firestore
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ======================= 設定區域 =======================
KNOWLEDGE_COLLECTION = "knowledge_base"
REMINDER_COLLECTION = "reminders"
RAG_CONFIDENCE_THRESHOLD = 0.5

initial_knowledge_data = [
    {"content": "本公司的營業時間是週一至週五，早上九點到下午六點。"},
    {"content": "退貨政策：非特價商品可在購買後30天內憑發票退票。"},
    {"content": "技術支援請發送電子郵件至 support@mycompany.com。"},
    {"content": "114年工作考成分數(立法院提刪通過)為 6.91 分。"},
    {"content": "114年工作考成分數(立法院提刪未通過)為 6.04 分。"},
    {"content": "114年工作考成分數(含不可抗力因素)為 6.46 分。"},
]

app = Flask(__name__)

# Channel Access Token / Secret
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化 Gemini & Firestore
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("服務初始化成功")
except Exception as e:
    print(f"初始化失敗: {e}")

# ======================= Function Calling 工具定義 =======================

def save_reminder_tool(content: str):
    """
    當使用者想要『記下』、『提醒』、『儲存』或『預約』任何行程、事件時，調用此工具。
    參數 content 應包含時間與事件細節。
    """
    return {"status": "processing", "content": content}

# ======================= 輔助函式 =======================

def cosine_distance(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude_v1 = math.sqrt(sum(v1 * v1 for v1 in vec1))
    magnitude_v2 = math.sqrt(sum(v2 * v2 for v2 in vec2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 1.0
    return 1.0 - (dot_product / (magnitude_v1 * magnitude_v2))

def get_embedding(text):
    if not client: return None
    try:
        result = client.models.embed_content(model='text-embedding-004', contents=[text])
        return result.embeddings[0].values
    except:
        return None

def record_reminder(user_id, raw_text):
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        return True, f"✅ 已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        return False, f"❌ 記錄失敗：{e}"

def get_user_reminders(user_id):
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where('user_id', '==', user_id)\
                 .where('is_completed', '==', False)\
                 .order_by('recorded_at', direction=firestore.Query.DESCENDING)\
                 .limit(10).stream()
        return [f"• {d.to_dict()['raw_text']}" for d in docs]
    except:
        return []

def query_knowledge_base(query_text):
    query_embedding = get_embedding(query_text)
    if not query_embedding: return "", False
    results = []
    try:
        docs = db.collection(KNOWLEDGE_COLLECTION).stream()
        for doc in docs:
            data = doc.to_dict()
            dist = cosine_distance(query_embedding, json.loads(data['embedding_json']))
            results.append((dist, data['content']))
        results.sort(key=lambda x: x[0])
        if results and results[0][0] < RAG_CONFIDENCE_THRESHOLD:
            return results[0][1], True
    except:
        pass
    return "", False

# ======================= 核心 AI 邏輯 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "系統未就緒"

    rag_context, _ = query_knowledge_base(user_text)
    reminders = get_user_reminders(user_id)
    personal_context = "\n".join(reminders) if reminders else "目前無未完成行程。"

    tools = [save_reminder_tool]
    system_instruction = (
        "你是一位專業助理。你具備查詢企業知識與記錄個人行程的能力。\n"
        "1. 如果使用者要記錄行程，請調用 save_reminder_tool。\n"
        "2. 關於公司規定請參考企業知識，關於使用者行程請參考個人行程。\n"
        f"【企業知識】：{rag_context}\n"
        f"【個人行程】：{personal_context}"
    )

    try:
        config = types.GenerateContentConfig(
            temperature=0.3,
            tools=tools,
            system_instruction=system_instruction,
        )

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_text,
            config=config,
        )

        # 檢查 Function Call
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    fn = part.function_call
                    extracted_text = fn.args.get("content", user_text)
                    success, msg = record_reminder(user_id, extracted_text)
                    return msg
        
        return response.text if response.text else "我不太明白，能換個說法嗎？"
    except:
        return "⚠️ 服務忙碌中，請稍後。"

# ======================= Flask Routes (包含喚醒與重設功能) =======================

@app.route('/')
def index():
    # UptimeRobot 訪問這裡來保持 Render 喚醒
    return "✅ LINE Bot Flask App is running on Render!"

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
    """手動清除 Firestore 知識庫並重建"""
    if not db: return "❌ Firestore 未初始化"
    try:
        docs = db.collection(KNOWLEDGE_COLLECTION).list_documents()
        batch = db.batch()
        count = 0
        for doc in docs:
            batch.delete(doc)
            count += 1
        if count > 0: batch.commit()
        initialize_knowledge_base()
        return f"✅ 已重設 {count} 筆資料並重新初始化。"
    except Exception as e:
        return f"❌ 失敗: {e}"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    user_id = event.source.user_id
    reply_text = GEMINI_response(user_msg, user_id)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

# ======================= 初始化與啟動 =======================

def initialize_knowledge_base():
    if not db: return
    try:
        docs = db.collection(KNOWLEDGE_COLLECTION).limit(1).get()
        if len(list(docs)) == 0:
            for i, item in enumerate(initial_knowledge_data):
                emb = get_embedding(item['content'])
                if emb:
                    db.collection(KNOWLEDGE_COLLECTION).document(f"k_{i}").set({
                        'content': item['content'],
                        'embedding_json': json.dumps(emb)
                    })
            print("知識庫初始化完成")
    except Exception as e:
        print(f"初始化錯誤: {e}")

if __name__ == "__main__":
    initialize_knowledge_base()
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
