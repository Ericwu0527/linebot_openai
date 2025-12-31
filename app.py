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

# Firestore 模組
from google.cloud import firestore
import google.cloud.firestore_v1 as firestore_module 
from google.cloud.firestore_v1.base_query import FieldFilter 

# Google GenAI
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ======================= 設定區域 =======================
KNOWLEDGE_COLLECTION = "knowledge_base"
REMINDER_COLLECTION = "reminders"
RAG_CONFIDENCE_THRESHOLD = 0.5

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務初始化成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具定義 (必須極度簡單) =======================

def save_reminder_tool(content: str):
    """
    用於記錄使用者的行程。當使用者提到時間、計畫或要做的事情時，必須調用此工具。
    """
    return {"status": "intent_detected", "content": content}

# ======================= 核心寫入函式 (加強報錯) =======================

def record_reminder(user_id, raw_text):
    print(f"DEBUG: 嘗試寫入 Firestore -> {raw_text}", flush=True)
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        print("DEBUG: Firestore 寫入成功！", flush=True)
        return True, f"✅ 已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        # 如果這裡印出 Permission Denied，就是步驟 1 的規則沒改好
        print(f"❌ Firestore 寫入報錯: {e}", flush=True)
        return False, f"❌ 寫入資料庫失敗，請檢查規則設定。詳細錯誤：{e}"

# ======================= 改版後的回應邏輯 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ API 未就緒"
    
    # RAG 檢索
    from_db = []
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .order_by('recorded_at', direction=firestore_module.Query.DESCENDING)\
                 .limit(5).stream()
        from_db = [f"• {d.to_dict()['raw_text']}" for d in docs]
    except Exception as e:
        print(f"查詢行程報錯: {e}")

    personal_context = "\n".join(from_db) if from_db else "目前無行程。"
    
    # 極度強硬的指令
    system_instruction = (
        "你是一個專業的『行程紀錄秘書』。\n"
        "當使用者提到『明天』、『後天』、『幾點』或『要去哪裡』時，你『禁止』用對話詢問，"
        "必須『立刻』調用 save_reminder_tool 來儲存行程。\n"
        "只有在使用者詢問『我明天要去哪？』時，你才查閱【個人行程】並用文字回答。\n"
        f"【個人行程】:\n{personal_context}"
    )
    
    try:
        # 使用 1.5-flash 提升 Function Calling 穩定度
        config = types.GenerateContentConfig(
            temperature=0, # 設為 0 確保最不囉唆，直接執行工具
            tools=[save_reminder_tool], 
            system_instruction=system_instruction
        )
        
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=user_text, 
            config=config
        )

        # 優先檢查是否有工具調用
        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts
            
            # 尋找是否有 function_call
            for part in parts:
                if part.function_call:
                    fn = part.function_call
                    extracted_text = fn.args.get("content", user_text)
                    success, msg = record_reminder(user_id, extracted_text)
                    return msg
            
            # 如果 AI 堅持不調用工具而只回傳文字
            for part in parts:
                if part.text:
                    return part.text

        return "我不確定這是否為行程，如果您想記錄，請說『幫我記下：...』"
        
    except Exception as e:
        print(f"❌ Gemini 錯誤: {e}", flush=True)
        return f"⚠️ 服務錯誤: {e}"

# ======================= Flask & LINE 保持不變 =======================

@app.route('/')
def index(): return "✅ Service Alive"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try: handler.handle(body, signature)
    except InvalidSignatureError: abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    user_id = event.source.user_id
    print(f"\n[User]: {user_msg}", flush=True)
    reply_text = GEMINI_response(user_msg, user_id)
    print(f"[AI]: {reply_text}", flush=True)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
