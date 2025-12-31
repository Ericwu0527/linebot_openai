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
from google.genai.errors import APIError

# ======================= 設定區域 =======================
KNOWLEDGE_COLLECTION = "knowledge_base"
REMINDER_COLLECTION = "reminders"

app = Flask(__name__)

# LINE Bot 設定 (從環境變數讀取)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化 Gemini & Firestore
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務初始化成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具與資料庫函式 =======================

def save_reminder_tool(content: str):
    """
    Action: 儲存使用者的行程或備忘錄。
    When to call: 只要使用者提到『未來時間』加上『要做的事情』，請務必調用此工具。
    Example: '明天下午2點去運動' -> content='明天下午2點運動'
    """
    return {"status": "intent_confirmed", "content": content}

def record_reminder(user_id, raw_text):
    """將行程寫入 Firestore"""
    print(f"DEBUG: [Step 1] AI 下達儲存指令 -> {raw_text}", flush=True)
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        print("DEBUG: [Step 2] Firestore 寫入成功！", flush=True)
        return True, f"✅ 好的，已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        print(f"❌ Firestore 寫入失敗: {e}", flush=True)
        return False, f"❌ 儲存失敗，請檢查資料庫權限。{e}"

def get_user_reminders(user_id):
    """從 Firestore 讀取行程 (限 5 筆以確保速度)"""
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .order_by('recorded_at', direction=firestore_module.Query.DESCENDING)\
                 .limit(5).stream()
        return [d.to_dict().get('raw_text', '') for d in docs]
    except Exception as e:
        print(f"⚠️ 讀取行程警告: {e}", flush=True)
        return []

# ======================= 核心 AI 回應邏輯 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ AI 模組未就緒"
    
    # 1. 時間處理 (設定台灣時區 UTC+8)
    now = datetime.now() + timedelta(hours=8)
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    minguo_year = now.year - 1911
    
    # 2. 只有在問問題時才抓取資料庫 (節省反應時間)
    is_asking = any(q in user_text for q in ["去哪", "做什麼", "行程", "幾點", "什麼事", "有沒有"])
    personal_context = ""
    if is_asking:
        reminders = get_user_reminders(user_id)
        personal_context = "\n".join([f"- {r}" for r in reminders]) if reminders else "目前無行程記錄。"

    # 3. 系統指令 (注入時間感、年份換算、強硬分流)
    system_instruction = (
        f"現在時間：{current_time_str} (民國 {minguo_year} 年)。\n"
        "你是秘書。規則如下：\n"
        "1. 使用者說『我要去...』、『下午要...』、『幫我記...』：立刻調用 save_reminder_tool，不准廢話問問題。\n"
        "2. 使用者問『要去哪』、『有什麼行程』：禁止調用工具。請查閱下方的【行程記錄】用『您』來回答。\n"
        "3. 時間換算：『明天』指 2026-01-01 (民國 115 年)。\n\n"
        f"【行程記錄Context】: {personal_context if is_asking else '非查詢模式'}"
    )
    
    try:
        config = types.GenerateContentConfig(
            temperature=0,
            tools=[save_reminder_tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            ),
            system_instruction=system_instruction
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents=user_text,
            config=config
        )

        # 4. 強化解析 (確保無論如何都有回覆)
        if not response.candidates or not response.candidates[0].content.parts:
            return "收到，但我現在無法處理這項訊息，請再試一次。"

        parts = response.candidates[0].content.parts
        
        # A. 優先檢查是否要記錄 (Function Call)
        for part in parts:
            if hasattr(part, 'function_call') and part.function_call:
                # 若是問句卻誤觸工具，則跳過工具邏輯
                if is_asking: continue 
                
                print(f"DEBUG: AI 觸發 [記錄] 工具", flush=True)
                content = part.function_call.args.get("content", user_text)
                _, msg = record_reminder(user_id, content)
                return msg
        
        # B. 檢查是否要回答 (Text Response)
        for part in parts:
            if hasattr(part, 'text') and part.text:
                return part.text.strip()

        return "我已收到您的訊息，請問需要幫您記錄下來嗎？"
        
    except Exception as e:
        print(f"❌ Gemini Error:\n{traceback.format_exc()}", flush=True)
        return "⚠️ 系統忙碌中，請稍後再試。"

# ======================= Flask 路由 =======================

@app.route('/')
def index():
    return "✅ LINE Bot is active and time-aware!"

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
    
    print(f"\n[User Message]: {user_msg}", flush=True)
    
    # 計算處理秒數 (若 > 2s 則 LINE 可能逾時)
    start_time = time.time()
    reply_text = GEMINI_response(user_msg, user_id)
    duration = time.time() - start_time
    
    print(f"[AI Reply ({round(duration, 2)}s)]: {reply_text}\n", flush=True)
    
    try:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )
    except Exception as e:
        print(f"❌ LINE 回覆失敗: {e}", flush=True)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
