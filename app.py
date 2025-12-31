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
RAG_CONFIDENCE_THRESHOLD = 0.5

app = Flask(__name__)

# 從環境變數獲取 LINE 資訊
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化 Gemini & Firestore
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務初始化成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具定義 (意圖開關) =======================

def save_reminder_tool(content: str):
    """
    Action: 儲存使用者的行程或備忘錄。
    When to call: 只要使用者提到『時間』(如:明天、後天、幾點) 加上 『動作』，必須調用此工具。
    """
    return {"status": "intent_confirmed", "content": content}

# ======================= 資料庫寫入函式 =======================

def record_reminder(user_id, raw_text):
    # 這是判斷有沒有進入寫入流程的關鍵日誌
    print(f"DEBUG: [Step 2] AI 已下達指令，準備寫入 Firestore -> {raw_text}", flush=True)
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        print("DEBUG: [Step 3] Firestore 寫入成功！", flush=True)
        return True, f"✅ 已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        print(f"❌ [Step 3 Error] Firestore 寫入報錯: {e}", flush=True)
        return False, f"❌ 寫入資料庫失敗。請檢查 Firestore Rules。錯誤訊息：{e}"

# ======================= AI 回應邏輯 (穩定優先) =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ API 未就緒"
    
    # 讀取現有行程 (提供上下文給 AI)
    reminders_list = []
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .order_by('recorded_at', direction=firestore_module.Query.DESCENDING)\
                 .limit(5).stream()
        reminders_list = [f"• {d.to_dict()['raw_text']}" for d in docs]
    except Exception as e:
        print(f"查詢行程報錯: {e}")

    personal_context = "\n".join(reminders_list) if reminders_list else "目前無未完成行程。"
    
    # 極度強制的指令，不准 AI 說廢話
    system_instruction = (
        "你是一個專業行程秘書。請嚴格遵守：\n"
        "1. 只要使用者說出任何『計畫』、『預約』或『明天/後天要去哪』，必須『立刻』呼叫 save_reminder_tool。\n"
        "2. 禁止回答『我收到了』或『需要幫你記下來嗎？』，直接記錄就對了。\n"
        "3. 如果使用者問『我明天要去哪？』，請從下面的【個人行程】中回答他，不要重複記錄。\n"
        f"【個人行程】如下：\n{personal_context}"
    )
    
    try:
        # 使用 1.5-flash 版本，穩定度最高
        config = types.GenerateContentConfig(
            temperature=0, 
            tools=[save_reminder_tool], 
            system_instruction=system_instruction
        )
        
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=user_text, 
            config=config
        )

        # 優先處理功能調用
        if response.candidates and response.candidates[0].content.parts:
            parts = response.candidates[0].content.parts
            
            # 檢查 AI 是否有想調用工具
            for part in parts:
                if part.function_call:
                    print(f"DEBUG: [Step 1] AI 識別意圖成功，準備提取參數", flush=True)
                    fn = part.function_call
                    extracted_text = fn.args.get("content", user_text)
                    success, msg = record_reminder(user_id, extracted_text)
                    return msg
            
            # 如果沒有工具調用，才回傳文字（例如：詢問明天行程時）
            for part in parts:
                if part.text and part.text.strip():
                    return part.text

        return "我不確定這是否為一個行程，若要記錄請說『幫我記下：[內容]』"
        
    except Exception as e:
        print(f"❌ Gemini API 錯誤: {e}", flush=True)
        return "⚠️ AI 服務忙碌中，請稍後再試。"

# ======================= Flask & LINE 路由 (保持不變) =======================

@app.route('/')
def index():
    return "✅ LINE Bot is running and waiting for UptimeRobot!"

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
    
    print(f"\n[LINE User Message]: {user_msg}", flush=True)
    
    # 進入 AI 分流邏輯
    reply_text = GEMINI_response(user_msg, user_id)
    
    print(f"[AI Response]: {reply_text}\n", flush=True)
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
