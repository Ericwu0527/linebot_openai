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
        # 修正後的模式：ANY 代表「強制模型必須選擇一個工具執行」
        config = types.GenerateContentConfig(
            temperature=0, 
            tools=[save_reminder_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY", # 這裡是關鍵修正
                )
            ),
            system_instruction="你是一個專用的行程提取器。將使用者的所有計畫或行程轉化為 save_reminder_tool 的 content。"
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=user_text, 
            config=config
        )

        # 輸出完整結構以便日誌追蹤
        print(f"DEBUG - Full Response: {response}", flush=True)

        if response.candidates:
            parts = response.candidates[0].content.parts
            if parts:
                for part in parts:
                    # 1. 檢查 Function Call (優先)
                    if part.function_call:
                        print(f"DEBUG: [成功觸發] AI 執行工具: {part.function_call.name}", flush=True)
                        extracted_text = part.function_call.args.get("content", user_text)
                        success, msg = record_reminder(user_id, extracted_text)
                        return msg
                    
                    # 2. 檢查文字 (只有在沒有工具時才會跑到這)
                    if part.text:
                        return part.text

        return "我不確定如何記錄這項訊息，請嘗試更具體的描述。"
        
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
