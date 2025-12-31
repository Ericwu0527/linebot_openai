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
    
    # 1. 先抓取現有行程（這很重要，AI 才能回答你「要去哪」）
    reminders_list = get_user_reminders(user_id)
    personal_context = "\n".join(reminders_list) if reminders_list else "目前沒有任何行程記錄。"
    
    try:
        # 模式改回 AUTO，讓 AI 決定要「講話」還是「用工具」
        config = types.GenerateContentConfig(
            temperature=0, 
            tools=[save_reminder_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="AUTO", # 恢復智慧模式
                )
            ),
            system_instruction=(
                "你是一位行程秘書。請根據以下邏輯處理訊息：\n"
                "【場景 A：記錄行程】\n"
                "當使用者提到未來的計畫（如：明天要去...、幫我記下...）時，請『務必調用』save_reminder_tool。\n\n"
                "【場景 B：回答問題】\n"
                "當使用者詢問自己的行程（如：我明天要去哪？、我有什麼行程？）時，請『絕對不要』調用工具，"
                "請直接根據下方的【個人行程 context】回答使用者。\n\n"
                f"【個人行程 context】:\n{personal_context}"
            )
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=user_text, 
            config=config
        )

        # 這裡的解析邏輯要非常清楚
        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts
            
            # 先檢查是否有工具呼叫（優先處理記錄意圖）
            for part in parts:
                if part.function_call:
                    print(f"DEBUG: [觸發記錄] AI 判定為新行程", flush=True)
                    fn = part.function_call
                    extracted_text = fn.args.get("content", user_text)
                    success, msg = record_reminder(user_id, extracted_text)
                    return msg
            
            # 如果沒有工具呼叫，則處理 AI 的文字回覆（回答問題）
            for part in parts:
                if part.text:
                    print(f"DEBUG: [觸發對話] AI 判定為查詢或聊天", flush=True)
                    return part.text

        return "我不確定如何處理這項訊息，您可以試著說『幫我記下...』或詢問『我明天有什麼行程？』"
        
    except Exception as e:
        print(f"❌ Gemini 錯誤:\n{traceback.format_exc()}", flush=True)
        return "⚠️ 服務忙碌中，請稍後再試。"
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
