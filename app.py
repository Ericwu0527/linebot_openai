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

# ======================= Firestore 引入區 =======================
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

# LINE Bot 設定
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化服務
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務初始化成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具與輔助函式 =======================

def save_reminder_tool(content: str):
    """
    用於儲存行程。僅在使用者描述未來計畫或要求『記下』時調用。
    參數 content 應包含時間與事件細節。
    """
    return {"status": "intent_detected", "content": content}

def record_reminder(user_id, raw_text):
    """將行程寫入 Firestore"""
    print(f"DEBUG: 準備寫入資料庫 -> {raw_text}", flush=True)
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        print("DEBUG: 資料庫寫入成功", flush=True)
        return True, f"✅ 好的，我已經幫您記下了：\n「{raw_text}」"
    except Exception as e:
        print(f"❌ 資料庫寫入失敗: {e}", flush=True)
        return False, f"❌ 記錄失敗：{e}"

def get_user_reminders(user_id):
    """從 Firestore 讀取未完成行程 (限制 5 筆以確保速度)"""
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
    
    # 1. 注入時間感 (處理台灣時區 UTC+8)
    now = datetime.now() + timedelta(hours=8)
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    minguo_year = now.year - 1911
    
    # 2. 抓取行程上下文
    reminders = get_user_reminders(user_id)
    personal_context = "\n".join([f"- {r}" for r in reminders]) if reminders else "目前無行程記錄。"
    
    # 3. 系統指令 (注入時間感、年份換算、意圖分流)
    system_instruction = (
        f"今天是 {current_time_str} (民國 {minguo_year} 年)。\n"
        "你是使用者的專業私人秘書，請遵守以下原則：\n"
        "【時間處理】\n"
        "- 台灣使用者說『115年』= 2026年，『116年』= 2027年。\n"
        "- 使用者說『明天』，指的就是資料庫中 2026-01-01 (115年1月1日) 的行程。\n\n"
        "【意圖判斷】\n"
        "1. **查詢行程**：只要句子包含『去哪』、『有什麼事』、『做什麼』、『行程』等問句，"
        "請禁止調用工具。請查閱【行程記錄】用親切的語氣告訴使用者答案。\n"
        "2. **記錄行程**：使用者提到未來計畫（例如：我明天要...、幫我記下...）時，才調用 save_reminder_tool。\n"
        "3. **主詞規範**：稱呼使用者為『您』，回答要自然且簡短。\n\n"
        f"【行程記錄Context】:\n{personal_context}"
    )
    
    try:
        config = types.GenerateContentConfig(
            temperature=0, # 設為 0 確保最穩定且不囉嗦
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

        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts
            
            # 優先檢查是否有工具調用 (意圖記錄)
            for part in parts:
                if part.function_call:
                    # 雙重檢查：如果是問句，則不執行寫入動作
                    if any(q in user_text for q in ["去哪", "做什麼", "有沒有", "幾點"]):
                        continue 
                    print(f"DEBUG: AI 觸發 [記錄] 功能", flush=True)
                    content = part.function_call.args.get("content", user_text)
                    _, msg = record_reminder(user_id, content)
                    return msg
            
            # 處理文字回覆 (意圖查詢/聊天)
            for part in parts:
                if part.text:
                    return part.text.strip()

        return "抱歉，我現在無法確認這項行程資訊。"
        
    except Exception as e:
        print(f"❌ Gemini Error:\n{traceback.format_exc()}", flush=True)
        return "⚠️ AI 服務忙碌中，請稍後再試。"

# ======================= Flask 路由與處理 =======================

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
    
    print(f"\n[User]: {user_msg}", flush=True)
    
    # 紀錄處理時間
    start_time = time.time()
    reply_text = GEMINI_response(user_msg, user_id)
    duration = time.time() - start_time
    
    print(f"[AI Reply ({round(duration, 2)}s)]: {reply_text}\n", flush=True)
    
    # 回覆訊息給 LINE
    try:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )
    except Exception as e:
        print(f"❌ LINE 回覆失敗 (逾時或 token 失效): {e}", flush=True)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
