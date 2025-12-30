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

# 引入 Firestore 與 GenAI
from google.cloud import firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ======================= 設定區域 =======================
KNOWLEDGE_COLLECTION = "knowledge_base"
REMINDER_COLLECTION = "reminders"
RAG_CONFIDENCE_THRESHOLD = 0.5

# 初始企業知識
initial_knowledge_data = [
    {"content": "本公司的營業時間是週一至週五，早上九點到下午六點。"},
    {"content": "退貨政策：非特價商品可在購買後30天內憑發票退票。"},
    {"content": "技術支援請發送電子郵件至 support@mycompany.com。"},
    {"content": "114年工作考成分數(立法院提刪通過)為 6.91 分。"},
    {"content": "114年工作考成分數(立法院提刪未通過)為 6.04 分。"},
    {"content": "114年工作考成分數(含不可抗力因素)為 6.46 分。"},
]

app = Flask(__name__)

# Channel Access Token / Secret (從環境變數讀取)
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化 Gemini & Firestore
try:
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
    db = firestore.Client()
    print("✅ 系統服務初始化成功", flush=True)
except Exception as e:
    print(f"❌ 初始化失敗: {e}", flush=True)

# ======================= 工具定義 (Tools) =======================

def save_reminder_tool(content: str):
    """
    當使用者提到未來的行程、計畫、動作、預約或想記下的事情時（例如：『明天我要去...』、『幫我記下...』），請務必調用此工具。
    參數 content：應包含時間與行程的簡短描述。
    """
    return {"status": "intent_detected", "content": content}

# =======================
