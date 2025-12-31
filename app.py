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

# Firestore 引入
from google.cloud import firestore
import google.cloud.firestore_v1 as firestore_module 
from google.cloud.firestore_v1.base_query import FieldFilter 

# Google GenAI
from google import genai
from google.genai import types

# ======================= 設定區域 =======================
REMINDER_COLLECTION = "reminders"

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務連線成功 (時間強化版)", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具與資料庫函式 =======================

def save_reminder_tool(content: str):
    """用於紀錄行程。"""
    return {"status": "intent_confirmed", "content": content}

def record_reminder(user_id, raw_text):
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        return True, f"✅ 已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        return False, f"❌ 儲存失敗"

def get_user_reminders(user_id):
    """讀取行程並包含紀錄日期"""
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .order_by('recorded_at', direction=firestore_module.Query.DESCENDING)\
                 .limit(10).stream()
        
        results = []
        for d in docs:
            data = d.to_dict()
            recorded_time = data.get('recorded_at')
            if recorded_time:
                # 這裡的 recorded_time 是伺服器時間 (UTC)，轉為台灣時間顯示給 AI 看
                ts = recorded_time + timedelta(hours=8)
                results.append(f"(紀錄日期: {ts.strftime('%Y-%m-%d')}) - {data.get('raw_text')}")
        return results
    except:
        return []

# ======================= 核心 AI 邏輯 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ AI 未就緒"
    
    # 1. 注入台灣時間 (UTC+8)
    now = datetime.now() + timedelta(hours=8)
    current_time_str = now.strftime("%Y-%m-%d")
    
    # 判斷是否為查詢
    is_asking = any(q in user_text for q in ["去哪", "什麼", "行程", "要做"])
    reminders = get_user_reminders(user_id) if is_asking else []
    personal_context = "\n".join(reminders) if reminders else "無任何記錄。"

    # 2. 系統指令 (教導 AI 根據紀錄日期做加減法)
    system_instruction = (
        f"今天是 {current_time_str}。\n"
        "你是專業秘書。請嚴格執行以下邏輯：\n"
        "1. 使用者描述新計畫：立刻呼叫 save_reminder_tool 直接儲存。\n"
        "2. 使用者查詢行程：禁止使用工具。請查閱【 行程記錄 】。\n"
        "   - 重要：請根據每筆紀錄前的 (紀錄日期) 來換算相對時間 (如：明天、後天)。\n"
        "   - 若推算後日期不符，請回答『當天沒有行程記錄』。\n"
        "   - 若日期符合，請將行程中的相對時間轉為絕對日期回答使用者。\n\n"
        f"【 行程記錄 】:\n{personal_context}"
    )
    
    try:
        config = types.GenerateContentConfig(
            temperature=0,
            tools=[save_reminder_tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            system_instruction=system_instruction
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=user_text,
            config=config
        )

        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts
            for part in parts:
                if hasattr(part, 'function_call') and part.function_call:
                    if is_asking: return "這似乎是查詢，目前的行程有：\n" + personal_context
                    fn = part.function_call
                    extracted = fn.args.get("content", user_text)
                    _, msg = record_reminder(user_id, extracted)
                    return msg
            for part in parts:
                if hasattr(part, 'text') and part.text:
                    return part.text.strip()
        return "我收到訊息了。"
        
    except Exception as e:
        print(f"❌ 錯誤:\n{traceback.format_exc()}", flush=True)
        return "⚠️ 系統錯誤"

# ======================= 其餘 Flask 邏輯保持不變 =======================
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try: handler.handle(body, signature)
    except: abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    user_id = event.source.user_id
    reply_text = GEMINI_response(user_msg, user_id)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route('/')
def index(): return "✅ Online"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
