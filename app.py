from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

import os
import time
import traceback
import math 
import json 
from datetime import datetime # 新增：用於記錄行程

# 【變更 1】引入 Firestore 函式庫
from google.cloud import firestore

# 引入 Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ======================= RAG 知識庫設定 (使用 Firestore) =======================
# 設定 Firestore 集合名稱 (企業知識)
KNOWLEDGE_COLLECTION = "knowledge_base" 
# 設定 Firestore 集合名稱 (個人行程)
REMINDER_COLLECTION = "reminders" 

# 初始資料 (只在資料庫第一次建立時使用)
initial_knowledge_data = [
    {"content": "本公司的營業時間是週一至週五，早上九點到下午六點。"},
    {"content": "退貨政策：非特價商品可在購買後30天內憑發票退票。"},
    {"content": "技術支援請發送電子郵件至 support@mycompany.com。"},
    {"content": "114年工作考成分數(立法院提刪通過)為 6.91 分。"}, 
    {"content": "114年工作考成分數(立法院提刪未通過)為 6.04 分。"}, 
    {"content": "114年工作考成分數(含不可抗力因素)為 6.46 分。"},
]

# RAG 信心門檻：使用餘弦距離 (Cosine Distance)，距離 0.5 表示相似度為 0.5
RAG_CONFIDENCE_THRESHOLD = 0.5 
# =============================================================


# 初始化 Flask
app = Flask(__name__)

# Channel Access Token / Secret
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 從環境變數獲取 Gemini API Key (請確保您的環境變數名稱為 GEMINI_API_KEY)
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    print("警告：未設定 GEMINI_API_KEY 環境變數！API 呼叫將會失敗。")

# 初始化 Gemini Client
try:
    client = genai.Client()
except Exception as e:
    print(f"初始化 Gemini 客戶端失敗: {e}")
    client = None

# 初始化 Firestore 客戶端
try:
    db = firestore.Client()
    print("Firestore 客戶端初始化成功。")
except Exception as e:
    print(f"初始化 Firestore 客戶端失敗: {e}")
    db = None


def cosine_distance(vec1, vec2):
    """計算兩個向量之間的餘弦距離 (1 - 餘弦相似度) (距離越小，相似度越高)。"""
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude_v1 = math.sqrt(sum(v1 * v1 for v1 in vec1))
    magnitude_v2 = math.sqrt(sum(v2 * v2 for v2 in vec2))

    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 1.0 

    cosine_similarity = dot_product / (magnitude_v1 * magnitude_v2)
    return 1.0 - cosine_similarity


def get_embedding(text):
    """呼叫 Gemini API 取得文字的向量表示 (Embedding)。"""
    if not client:
        return None
    try:
        result = client.models.embed_content(
            model='text-embedding-004',
            contents=[text],
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"[Embedding Error] 無法生成向量: {e}")
        return None


def initialize_knowledge_base():
    """檢查 Firestore 資料庫，如果沒有資料則插入初始資料並生成向量。"""
    if not client or not db:
        print("警告：LLM 或 Firestore 客戶端未初始化，跳過知識庫初始化。")
        return
    
    doc_count = 0
    try:
        docs = db.collection(KNOWLEDGE_COLLECTION).limit(1).stream() 
        doc_count = sum(1 for _ in docs)
    except Exception as e:
        print(f"檢查 Firestore 集合失敗: {e}")
        return

    if doc_count == 0:
        print("正在初始化 RAG 知識庫 (生成 embeddings 並寫入 Firestore)...")
        for item in initial_knowledge_data:
            content = item['content']
            
            embedding = get_embedding(content)
            
            if embedding:
                embedding_json = json.dumps(embedding)
