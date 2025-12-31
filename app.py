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

# ======================= Firestore 名稱衝突修正 =======================
from google.cloud import firestore
import google.cloud.firestore_v1 as firestore_module 
from google.cloud.firestore_v1.base_query import FieldFilter 

# 引入 Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務啟動成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具定義 =======================

def save_reminder_tool(content: str):
    """
    用於記錄行程。當使用者提到時間、地點或要做的事情時，調用此工具。
    """
    return {"status": "intent_detected", "content": content}

# ======================= 資料庫寫入 (加強 Log) =======================

def record_reminder(user_id, raw_text):
    # 如果看到這行，代表 AI 終於肯呼叫工具了
    print(f"DEBUG: [Step 1] 收到 AI 指令，寫入內容 -> {raw_text}", flush=True)
    try:
        db.collection("reminders").add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        print("DEBUG: [Step 2] Firestore 寫入成功！", flush=True)
        return True, f"✅ 已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        # 如果這裡報 Permission Denied，請檢查你的 Firestore Rules 
        print(f"❌ [Step 2 Error] 資料庫寫入失敗: {e}", flush=True)
        return False, f"❌ 資料庫拒絕寫入。請確認 Firestore 規則已發佈為 true。"

# ======================= 核心回應邏輯 (強制模式) =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ API 未就緒"
    
    try:
        # 1. 設定強制調用模式
        # mode="REQUIRED" 會逼 AI 即使你說「你好」，它也得硬塞進 save_reminder_tool 裡
        config = types.GenerateContentConfig(
            temperature=0, 
            tools=[save_reminder_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="REQUIRED", # 強制執行工具
                )
            ),
            system_instruction="你是一個機器人，你的唯一任務就是把看到的所有文字丟進 save_reminder_tool 的 content 參數裡。"
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=user_text, 
            config=config
        )

        # 2. 【最重要】印出完整回應結構，我們在 Render Logs 裡抓兇手
        print(f"DEBUG - Full Response: {response}", flush=True)

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    # 檢查 Function Call
                    # 在某些 SDK 版本中，這可能在 part.function_call
                    fn = getattr(part, 'function_call', None)
                    if fn:
                        print(f"DEBUG: [成功] AI 呼叫了工具: {fn.name}", flush=True)
                        extracted_text = fn.args.get("content", user_text)
                        success, msg = record_reminder(user_id, extracted_text)
                        return msg
                    
                    # 檢查文字
                    txt = getattr(part, 'text', None)
                    if txt:
                        return txt

        return "DEBUG: AI 既沒給工具指令也沒給文字，請檢查 Logs 中的 Full Response。"
        
    except Exception as e:
        print(f"❌ Gemini 錯誤詳情:\n{traceback.format_exc()}", flush=True)
        return f"⚠️ 發生錯誤: {str(e)}"

# ======================= Flask 路由 =======================

@app.route('/')
def index(): return "✅ Alive"

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
    print(f"[AI]: {reply_text}", flush=True)
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
