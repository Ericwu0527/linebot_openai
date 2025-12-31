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

# ======================= Firestore & 語法修正 =======================
from google.cloud import firestore
import google.cloud.firestore_v1 as firestore_module 
from google.cloud.firestore_v1.base_query import FieldFilter 

# 引入 Google GenAI
from google import genai
from google.genai import types

# ======================= 設定區域 =======================
KNOWLEDGE_COLLECTION = "knowledge_base"
REMINDER_COLLECTION = "reminders"

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務初始化成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具與資料庫函式 =======================

def save_reminder_tool(content: str):
    """用於儲存行程。僅在使用者要求『記下』或『要去哪裡』的『宣告』時使用。"""
    return {"status": "intent_detected", "content": content}

def record_reminder(user_id, raw_text):
    """快速寫入資料庫"""
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        return True, f"✅ 好的，我已經幫您記下了：\n「{raw_text}」"
    except Exception as e:
        return False, f"❌ 儲存失敗: {e}"

def get_user_reminders(user_id):
    """快速讀取行程 (限 5 筆以提升速度)"""
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .order_by('recorded_at', direction=firestore_module.Query.DESCENDING)\
                 .limit(5).stream()
        return [d.to_dict().get('raw_text', '') for d in docs]
    except:
        return []

# ======================= 核心 AI 回應邏輯 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ API 未就緒"
    
    # 1. 抓取行程 (Context)
    reminders = get_user_reminders(user_id)
    # 將列表轉為乾淨的字串，供 AI 閱讀
    personal_context = "\n".join([f"- {r}" for r in reminders]) if reminders else "您目前沒有任何行程記錄。"
    
    # 2. 強化的系統指令：解決「主詞怪異」與「回答內容重複」
    system_instruction = (
        "你是一位親切且精確的私人秘書。\n"
        "【你的任務分流】\n"
        "1. **記錄任務**：如果使用者說『我要去...』或『幫我記下...』，請調用 save_reminder_tool，並簡短回覆已記下。\n"
        "2. **查詢任務**：如果使用者問『我要去哪？』或『我有什麼行程？』，請『絕對禁止』調用工具！請根據【行程記錄】用自然語言回答。\n"
        "3. **說話語氣**：請稱呼使用者為『您』。不要重複列出所有記錄，請幫使用者整理好。例如：『您明天下午 2 點要去松山運動。』\n\n"
        f"【行程記錄Context】:\n{personal_context}"
    )
    
    try:
        config = types.GenerateContentConfig(
            temperature=0, # 設為 0 確保回覆不囉唆且精準
            tools=[save_reminder_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
            system_instruction=system_instruction
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", # 確保使用 2.0-flash 速度最快
            contents=user_text,
            config=config
        )

        # 3. 解析與分流
        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts
            
            # 優先處理工具呼叫 (記錄)
            for part in parts:
                if part.function_call:
                    print(f"DEBUG: AI 觸發記錄功能", flush=True)
                    content = part.function_call.args.get("content", user_text)
                    _, msg = record_reminder(user_id, content)
                    return msg
            
            # 處理文字回覆 (查詢)
            for part in parts:
                if part.text:
                    return part.text.strip()

        return "抱歉，我不太確定該如何處理這項訊息。"
        
    except Exception as e:
        print(f"❌ Gemini Error: {e}", flush=True)
        return "⚠️ 處理訊息時發生錯誤，請稍後再試。"

# ======================= Flask & LINE 路由 =======================

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except Exception as e:
        print(f"Callback Error: {e}")
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    user_id = event.source.user_id
    
    print(f"\n[User]: {user_msg}", flush=True)
    
    # 計算處理時間，若太久會被 LINE 斷線
    start_time = time.time()
    reply_text = GEMINI_response(user_msg, user_id)
    end_time = time.time()
    
    print(f"[AI Reply ({round(end_time - start_time, 2)}s)]: {reply_text}", flush=True)
    
    # 回覆訊息給 LINE
    try:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )
    except Exception as e:
        print(f"❌ LINE 回覆失敗 (可能已逾時): {e}", flush=True)

@app.route('/')
def index(): return "✅ Bot Alive"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
