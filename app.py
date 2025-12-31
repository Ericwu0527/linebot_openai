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

# ======================= Firestore 引入區 =======================
from google.cloud import firestore
import google.cloud.firestore_v1 as firestore_module 
from google.cloud.firestore_v1.base_query import FieldFilter 

# 引入 Google GenAI
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ======================= 設定區域 =======================
KNOWLEDGE_COLLECTION = "knowledge_base"
REMINDER_COLLECTION = "reminders"
RAG_CONFIDENCE_THRESHOLD = 0.5

app = Flask(__name__)

# LINE Bot 設定
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化服務
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務初始化成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具與輔助函式 =======================

def save_reminder_tool(content: str):
    """
    用於儲存使用者的行程或備忘錄。當使用者提到時間與要做的事情時調用。
    """
    return {"status": "intent_detected", "content": content}

def record_reminder(user_id, raw_text):
    """將行程寫入 Firestore"""
    print(f"DEBUG: 準備寫入資料庫 -> {raw_text}", flush=True)
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        print("DEBUG: 資料庫寫入成功", flush=True)
        return True, f"✅ 已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        print(f"❌ 資料庫寫入失敗: {e}", flush=True)
        return False, f"❌ 記錄失敗：{e}"

def get_user_reminders(user_id):
    """從 Firestore 讀取未完成行程"""
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .order_by('recorded_at', direction=firestore_module.Query.DESCENDING)\
                 .limit(10).stream()
        return [f"• {d.to_dict()['raw_text']}" for d in docs]
    except Exception as e:
        print(f"⚠️ 讀取行程警告: {e}", flush=True)
        return []

def get_embedding(text):
    if not client: return None
    try:
        result = client.models.embed_content(model='text-embedding-004', contents=[text])
        return result.embeddings[0].values
    except:
        return None

def cosine_distance(vec1, vec2):
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude_v1 = math.sqrt(sum(v1 * v1 for v1 in vec1))
    magnitude_v2 = math.sqrt(sum(v2 * v2 for v2 in vec2))
    return 1.0 - (dot_product / (magnitude_v1 * magnitude_v2)) if magnitude_v1 != 0 and magnitude_v2 != 0 else 1.0

# ======================= 核心 AI 回應邏輯 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ API 未就緒"
    
    # 1. 抓取行程作為上下文 (Context)
    reminders_list = get_user_reminders(user_id)
    personal_context = "\n".join(reminders_list) if reminders_list else "目前沒有任何記錄。"
    
    try:
        config = types.GenerateContentConfig(
            temperature=0, 
            tools=[save_reminder_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="AUTO", # 智慧判斷模式
                )
            ),
            system_instruction=(
                "你是一位高效行程秘書。請遵守以下邏輯：\n"
                "1. 當使用者提到計畫（如：明天要去...、記下...）時，『必須調用』save_reminder_tool。\n"
                "2. 當使用者詢問行程（如：我明天要去哪？、我有什麼事？）時，『禁止調用工具』，"
                "請直接根據下方的【個人行程記錄】回答使用者。\n\n"
                f"【個人行程記錄】:\n{personal_context}"
            )
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=user_text, 
            config=config
        )

        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts
            
            # 優先處理 Function Call (記錄動作)
            for part in parts:
                if part.function_call:
                    print(f"DEBUG: [意圖識別] AI 決定記錄行程", flush=True)
                    fn = part.function_call
                    extracted_text = fn.args.get("content", user_text)
                    success, msg = record_reminder(user_id, extracted_text)
                    return msg
            
            # 處理一般文字回覆 (回答問題)
            for part in parts:
                if part.text:
                    return part.text

        return "我不確定這是否為一個行程，若要記錄請說『幫我記下...』"
        
    except Exception as e:
        print(f"❌ Gemini 錯誤:\n{traceback.format_exc()}", flush=True)
        return "⚠️ 服務忙碌中，請稍後。"

# ======================= Flask 路由與處理 =======================

@app.route('/')
def index():
    return "✅ LINE Bot is active!"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    user_id = event.source.user_id
    
    print(f"\n[User]: {user_msg}", flush=True)
    reply_text = GEMINI_response(user_msg, user_id)
    print(f"[AI]: {reply_text}\n", flush=True)
