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
from datetime import datetime, timedelta

# ======================= Firestore 引入修正 =======================
from google.cloud import firestore
import google.cloud.firestore_v1 as firestore_module 
from google.cloud.firestore_v1.base_query import FieldFilter 

# 引入 Google GenAI
from google import genai
from google.genai import types

# ======================= 設定區域 =======================
REMINDER_COLLECTION = "reminders"

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化服務
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務啟動成功 (使用 Gemini 2.0 Flash)", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具與資料庫函式 =======================

def save_reminder_tool(content: str):
    """用於儲存使用者的行程或備忘錄。當使用者提到時間與動作時調用。"""
    return {"status": "intent_detected", "content": content}

def record_reminder(user_id, raw_text):
    """執行 Firestore 寫入"""
    print(f"DEBUG: [資料庫寫入中...] 內容: {raw_text}", flush=True)
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        return True, f"✅ 好的，已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        print(f"❌ Firestore 報錯: {e}", flush=True)
        return False, f"❌ 資料庫寫入失敗: {e}"

def get_user_reminders(user_id):
    """讀取現有行程"""
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .order_by('recorded_at', direction=firestore_module.Query.DESCENDING)\
                 .limit(5).stream()
        return [d.to_dict().get('raw_text', '') for d in docs]
    except Exception as e:
        print(f"⚠️ 讀取警告: {e}", flush=True)
        return []

# ======================= 核心 AI 回應邏輯 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ AI 模組未就緒"
    
    # 1. 準備時間與上下文
    now = datetime.now() + timedelta(hours=8)
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 判斷是否為詢問 (簡單預過濾)
    is_query = any(q in user_text for q in ["去哪", "做什麼", "有事嗎", "行程"])
    reminders = get_user_reminders(user_id) if is_query else []
    personal_context = "\n".join([f"- {r}" for r in reminders]) if reminders else "無記錄。"

    # 2. 系統指令
    system_instruction = (
        f"現在時間：{current_time_str}。\n"
        "你是行程秘書。規則：\n"
        "1. 使用者描述新計畫：立刻呼叫 save_reminder_tool，不准多問。\n"
        "2. 使用者查詢行程：禁止用工具！直接根據【 context 】回答。\n"
        f"【 context 】:\n{personal_context}"
    )
    
    try:
        # 關鍵設定：關閉自動執行 (Automatic Function Calling)，由我們手動解析
        config = types.GenerateContentConfig(
            temperature=0,
            tools=[save_reminder_tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disabled=True),
            system_instruction=system_instruction,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
        )
        
        # 換回 2.0 版本，因為它在你的環境中是存在的
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=user_text,
            config=config
        )

        # 3. 手動解析回應內容
        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts
            
            # 優先搜尋 Function Call (記錄意圖)
            for part in parts:
                if hasattr(part, 'function_call') and part.function_call:
                    fn = part.function_call
                    print(f"DEBUG: [AI 指令] 呼叫工具 {fn.name}", flush=True)
                    # 避免問句被誤記
                    if is_query: return "目前您的行程有：\n" + personal_context
                    
                    extracted_text = fn.args.get("content", user_text)
                    _, msg = record_reminder(user_id, extracted_text)
                    return msg
            
            # 搜尋文字 (查詢意圖)
            for part in parts:
                if hasattr(part, 'text') and part.text:
                    return part.text.strip()

        return "我收到您的訊息了，請問有什麼需要幫您記錄的嗎？"
        
    except Exception as e:
        print(f"❌ 發生錯誤:\n{traceback.format_exc()}", flush=True)
        return "⚠️ 系統暫時無法回應，請稍後再試。"

# ======================= Flask & LINE 路由 =======================

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    user_id = event.source.user_id
    
    print(f"\n[User]: {user_msg}", flush=True)
    start_time = time.time()
    
    reply_text = GEMINI_response(user_msg, user_id)
    
    # 監控回應時間
    duration = round(time.time() - start_time, 2)
    print(f"[AI Reply ({duration}s)]: {reply_text}", flush=True)
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route('/')
def index(): return "✅ Service Alive"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
