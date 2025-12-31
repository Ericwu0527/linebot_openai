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

# ======================= 設定區域 =======================
REMINDER_COLLECTION = "reminders"

app = Flask(__name__)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化服務
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務連線成功 (CRUD 強化版)", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具定義區 =======================

def save_reminder_tool(content: str):
    """用於紀錄行程。當使用者提到時間與要做的事情時調用。"""
    return {"status": "intent_save", "content": content}

def delete_reminder_tool(keywords: str):
    """用於刪除、取消、移除行程。參數 keywords 為使用者提到的關鍵字（如：松山、醫生）。"""
    return {"status": "intent_delete", "keywords": keywords}

# ======================= 資料庫操作函式 =======================

def record_reminder(user_id, raw_text):
    """【新增】寫入資料庫"""
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id,
            'raw_text': raw_text,
            'recorded_at': firestore_module.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        return True, f"✅ 已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        return False, f"❌ 儲存失敗: {e}"

def delete_reminders_from_db(user_id, keywords):
    """【刪除】符合關鍵字的行程"""
    print(f"DEBUG: 準備刪除包含 '{keywords}' 的行程", flush=True)
    try:
        # 撈出該使用者所有未完成行程
        docs = db.collection(REMINDER_COLLECTION)\
                 .where(filter=FieldFilter('user_id', '==', user_id))\
                 .where(filter=FieldFilter('is_completed', '==', False))\
                 .stream()

        deleted_count = 0
        for doc in docs:
            content = doc.to_dict().get('raw_text', '')
            # 如果行程內容包含關鍵字，則刪除
            if keywords in content:
                doc.reference.delete()
                deleted_count += 1

        if deleted_count > 0:
            return True, f"🗑️ 已成功刪除 {deleted_count} 筆包含「{keywords}」的行程。"
        else:
            return False, f"⚠️ 找不到包含「{keywords}」的行程，請確認關鍵字是否正確。"
    except Exception as e:
        print(f"❌ 刪除出錯: {e}", flush=True)
        return False, f"❌ 刪除過程發生錯誤。"

def get_user_reminders(user_id):
    """【讀取】行程記錄"""
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
                ts = recorded_time + timedelta(hours=8)
                results.append(f"(紀錄日期: {ts.strftime('%Y-%m-%d')}) - {data.get('raw_text')}")
        return results
    except:
        return []

# ======================= 核心 AI 回應邏輯 =======================

def GEMINI_response(user_text, user_id):
    if not client: return "⚠️ AI 未就緒"
    
    # 1. 注入台灣時間
    now = datetime.now() + timedelta(hours=8)
    current_time_str = now.strftime("%Y-%m-%d")
    
    # 2. 準備上下文
    is_query = any(q in user_text for q in ["去哪", "什麼", "行程", "要做", "有哪些"])
    reminders = get_user_reminders(user_id)
    personal_context = "\n".join(reminders) if reminders else "無任何記錄。"

    # 3. 系統指令 (加入刪除功能的邏輯)
    system_instruction = (
        f"今天是 {current_time_str}。\n"
        "你是專業秘書。請執行以下分流邏輯：\n"
        "1. **新增行程**：提到新計畫時，呼叫 save_reminder_tool。\n"
        "2. **刪除行程**：當使用者說『取消』、『刪除』、『不用去了』時，呼叫 delete_reminder_tool 並提取關鍵字。\n"
        "3. **查詢行程**：使用者問問題時，禁止使用工具，直接根據【 行程記錄 】回答。\n\n"
        f"【 行程記錄 】:\n{personal_context}"
    )
    
    try:
        # 將兩個工具都放入清單
        config = types.GenerateContentConfig(
            temperature=0,
            tools=[save_reminder_tool, delete_reminder_tool],
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
                    fn = part.function_call
                    
                    # 處理【儲存工具】
                    if fn.name == "save_reminder_tool":
                        content = fn.args.get("content", user_text)
                        _, msg = record_reminder(user_id, content)
                        return msg
                    
                    # 處理【刪除工具】
                    if fn.name == "delete_reminder_tool":
                        kw = fn.args.get("keywords", "")
                        _, msg = delete_reminders_from_db(user_id, kw)
                        return msg
            
            # 處理文字回覆
            for part in parts:
                if hasattr(part, 'text') and part.text:
                    return part.text.strip()
        
        return "我收到訊息了，請問有什麼需要幫忙的嗎？"
        
    except Exception as e:
        print(f"❌ 錯誤詳情:\n{traceback.format_exc()}", flush=True)
        return "⚠️ 處理時發生錯誤。"

# ======================= Flask 路由 =======================

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
    
    # 獲取 AI 回應
    reply_text = GEMINI_response(user_msg, user_id)
    
    # 回覆給使用者
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))

@app.route('/')
def index(): return "✅ Online with Delete Function"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
