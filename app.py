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

# Google GenAI SDK
from google import genai
from google.genai import types

# ======================= 設定區域 =======================
REMINDER_COLLECTION = "reminders"

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 穩定版系統初始化成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具與資料庫 =======================

def save_reminder_tool(content: str):
    """用於紀錄行程。當使用者提到未來的計畫、要去哪裡、做什麼事時調用。"""
    return {"status": "intent_detected", "content": content}

def record_reminder(user_id, raw_text):
    print(f"DEBUG: [寫入動作] {raw_text}", flush=True)
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        return True, f"✅ 已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        print(f"❌ 資料庫錯誤: {e}", flush=True)
        return False, f"❌ 儲存失敗，請檢查規則。"

def get_user_reminders(user_id):
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .order_by('recorded_at', direction=firestore_module.Query.DESCENDING)\
                 .limit(5).stream()
        return [d.to_dict().get('raw_text', '') for d in docs]
    except:
        return []

# ======================= 核心 AI 邏輯 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ 系統未就緒"
    
    # 1. 注入時間
    now = datetime.now() + timedelta(hours=8)
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. 判斷是否為查詢意圖
    is_query = any(q in user_text for q in ["去哪", "做什麼", "有沒有", "行程"])
    reminders = get_user_reminders(user_id) if is_query else []
    personal_context = "\n".join([f"- {r}" for r in reminders]) if reminders else "目前無記錄。"

    # 3. 指令強化
    system_instruction = (
        f"現在時間：{current_time_str}。\n"
        "你是行程秘書。規則：\n"
        "1. 使用者描述新計畫：立刻呼叫 save_reminder_tool，不准廢話。\n"
        "2. 使用者問行程（去哪、做什麼）：禁調工具！直接根據 context 回答。\n"
        f"【 context 】:\n{personal_context}"
    )
    
    try:
        # 使用更穩定的 1.5-flash
        config = types.GenerateContentConfig(
            temperature=0,
            tools=[save_reminder_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
            system_instruction=system_instruction,
            # 關閉安全過濾
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
        )
        
        # 呼叫 API
        response = client.models.generate_content(
            model="gemini-1.5-flash", # 切換回穩定版
            contents=user_text,
            config=config
        )

        # DEBUG: 印出完整回應，觀察是否還有迴圈或過濾
        print(f"DEBUG Response: {response.candidates[0].finish_reason if response.candidates else 'Empty'}", flush=True)

        if not response.candidates:
            return "抱歉，系統目前無法處理這段訊息，請試著換個說法。"

        parts = response.candidates[0].content.parts
        if not parts:
            return "AI 沒有產生任何回覆。"

        # 解析零件
        for part in parts:
            # 優先處理工具呼叫
            if hasattr(part, 'function_call') and part.function_call:
                fn = part.function_call
                # 如果是問句卻觸發工具，代表 AI 判斷失誤，改為提示使用者
                if is_query: return "請問您是要記錄還是查詢行程呢？"
                
                print(f"DEBUG: 觸發功能調用 {fn.name}", flush=True)
                content = fn.args.get("content", user_text)
                _, msg = record_reminder(user_id, content)
                return msg
            
            # 處理文字
            if hasattr(part, 'text') and part.text:
                return part.text.strip()

        return "我收到訊息了，請問有什麼需要幫您記錄的嗎？"
        
    except Exception as e:
        print(f"❌ 錯誤詳情:\n{traceback.format_exc()}", flush=True)
        return "⚠️ 服務忙碌，請稍後再試。"

# ======================= 路由處理 =======================

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
    
    print(f"[AI Reply ({round(time.time() - start_time, 2)}s)]: {reply_text}", flush=True)
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route('/')
def index(): return "✅ Alive"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
