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
KNOWLEDGE_COLLECTION = "knowledge_base"
REMINDER_COLLECTION = "reminders"

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化服務
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務初始化成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具與資料庫函式 =======================

def save_reminder_tool(content: str):
    """
    Action: 儲存使用者的行程。
    Requirement: 只要使用者提到計畫要做的事，請調用此工具。
    """
    return {"status": "intent_confirmed", "content": content}

def record_reminder(user_id, raw_text):
    print(f"DEBUG: [資料庫動作] 準備寫入 -> {raw_text}", flush=True)
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        print("DEBUG: [資料庫動作] 寫入完成！", flush=True)
        return True, f"✅ 好的，我已經幫您記下了：\n「{raw_text}」"
    except Exception as e:
        print(f"❌ 資料庫錯誤: {e}", flush=True)
        return False, f"❌ 儲存失敗，請檢查資料庫權限。"

def get_user_reminders(user_id):
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .order_by('recorded_at', direction=firestore_module.Query.DESCENDING)\
                 .limit(10).stream()
        return [d.to_dict().get('raw_text', '') for d in docs]
    except:
        return []

# ======================= 核心 AI 邏輯 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ AI 模組未就緒"
    
    # 1. 注入台灣時間
    now = datetime.now() + timedelta(hours=8)
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # 2. 獲取行程上下文 (Context)
    reminders = get_user_reminders(user_id)
    personal_context = "\n".join([f"- {r}" for r in reminders]) if reminders else "目前無記錄。"

    # 3. 系統指令 (極簡化，避免 AI 混淆)
    system_instruction = (
        f"現在時間：{current_time_str}。\n"
        "你是秘書。規則：\n"
        "1. 使用者描述行程或要求記下：立刻調用 save_reminder_tool。\n"
        "2. 使用者『詢問』他要去哪或有什麼事：直接根據 Context 回答。\n"
        f"【現有行程】:\n{personal_context}"
    )
    
    try:
        # 設定安全設定，防止 AI 因為敏感字眼拒絕回覆
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        ]

        config = types.GenerateContentConfig(
            temperature=0,
            tools=[save_reminder_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
            system_instruction=system_instruction,
            safety_settings=safety_settings # 注入安全設定
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=user_text,
            config=config
        )

        # 4. 強化解析邏輯
        if not response.candidates or len(response.candidates) == 0:
            return "AI 目前無法回應（可能被過濾），請試著簡短說明您的行程。"

        content = response.candidates[0].content
        if not content or not content.parts:
            return "AI 回傳了空的內容，請稍後再試。"

        # 優先處理 Function Call
        for part in content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                # 如果是記錄動作
                fn = part.function_call
                print(f"DEBUG: [AI 指令] 觸發 save_reminder_tool", flush=True)
                # 提取參數，如果提取失敗則用原文
                extracted = fn.args.get("content", user_text) if fn.args else user_text
                _, msg = record_reminder(user_id, extracted)
                return msg
        
        # 處理文字回覆
        for part in content.parts:
            if hasattr(part, 'text') and part.text:
                return part.text.strip()

        return "我收到了您的訊息，請問需要幫您記錄下來嗎？"
        
    except Exception as e:
        print(f"❌ Gemini Error:\n{traceback.format_exc()}", flush=True)
        return "⚠️ 發生未知錯誤，請稍後再試。"

# ======================= Flask 路由 =======================

@app.route('/')
def index(): return "✅ Bot is active"

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
    duration = time.time() - start_time
    
    print(f"[AI Reply ({round(duration, 2)}s)]: {reply_text}\n", flush=True)
    
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
