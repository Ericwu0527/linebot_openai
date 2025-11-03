from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

import os
import time
import traceback
import math 
import sqlite3 # 引入 SQLite 函式庫
import json # 引入 json 函式庫用於序列化向量

# 引入 Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ======================= RAG 知識庫設定 =======================
# 【變更】使用 SQLite 檔案來持久化儲存資料
DB_FILE = "knowledge_base.db" 

# 初始資料 (只在資料庫第一次建立時使用)
initial_knowledge_data = [
    {"content": "本公司的營業時間是週一至週五，早上九點到下午六點。"},
    {"content": "退貨政策：非特價商品可在購買後30天內憑發票退貨。"},
    {"content": "技術支援請發送電子郵件至 support@mycompany.com。"},
    # ... 您可以加入更多自訂資料
]

# 【新增】RAG 信心門檻：歐幾里得距離小於此值，視為高相關度
RAG_CONFIDENCE_THRESHOLD = 1.0 
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


def get_db_connection():
    """建立並返回 SQLite 資料庫連線。"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row # 讓資料以字典形式返回
    return conn

def setup_db():
    """建立知識庫表格，如果它不存在。"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # 建立表格：content 儲存原始文本, embedding_json 儲存向量的 JSON 格式
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                embedding_json TEXT
            );
        """)
        conn.commit()
        conn.close()
        print("SQLite 資料庫設定完成。")
    except Exception as e:
        print(f"SQLite 資料庫設定失敗: {e}")


def euclidean_distance(vec1, vec2):
    """計算兩個向量之間的歐幾里得距離 (距離越小，相似度越高)。"""
    if len(vec1) != len(vec2):
        return float('inf')
    return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vec1, vec2)))


def get_embedding(text):
    """呼叫 Gemini API 取得文字的向量表示 (Embedding)。"""
    if not client:
        return None
    try:
        # 【修正 1: 將 'content' 改為 'contents'，並將 text 放入列表中】
        result = client.models.embed_content(
            model='text-embedding-004',
            contents=[text], # 這裡需要傳遞一個包含文本的列表
            task_type='RETRIEVAL_DOCUMENT'
        )
        # result['embedding'] 包含單個文本的向量
        return result['embedding']
    except Exception as e:
        print(f"[Embedding Error] 無法生成向量: {e}")
        return None


def initialize_knowledge_base():
    """檢查資料庫，如果沒有資料則插入初始資料並生成向量。"""
    if not client:
        return
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM knowledge_base")
    count = cursor.fetchone()[0]

    if count == 0:
        print("正在初始化 RAG 知識庫 (生成 embeddings 並寫入資料庫)...")
        for item in initial_knowledge_data:
            content = item['content']
            # 生成向量
            embedding = get_embedding(content)
            
            if embedding:
                # 將向量轉換為 JSON 字符串以便儲存在 SQLite
                embedding_json = json.dumps(embedding)
                cursor.execute(
                    "INSERT INTO knowledge_base (content, embedding_json) VALUES (?, ?)",
                    (content, embedding_json)
                )
        conn.commit()
        print("RAG 知識庫初始化完成，資料已儲存到 knowledge_base.db。")
    
    conn.close()


def add_new_knowledge(content):
    """
    將新的內容添加到知識庫資料庫，並自動生成向量。
    """
    if not client:
        print("無法新增知識：Gemini 客戶端未初始化。")
        return
        
    embedding = get_embedding(content)
    
    if embedding:
        conn = get_db_connection()
        cursor = conn.cursor()
        embedding_json = json.dumps(embedding)
        
        # 這裡直接插入新資料，您也可以加入邏輯檢查內容是否重複
        try:
            cursor.execute(
                "INSERT INTO knowledge_base (content, embedding_json) VALUES (?, ?)",
                (content, embedding_json)
            )
            conn.commit()
            print(f"[Success] 成功新增知識到資料庫: {content[:30]}...")
        except Exception as e:
            print(f"[Error] 新增知識失敗: {e}")
        finally:
            conn.close()
    else:
        print(f"[Error] 無法為內容生成 Embedding: {content[:30]}...")


def query_knowledge_base(query_text, top_k=3):
    """
    從 SQLite 資料庫中檢索與查詢最相關的文檔。
    """
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        # 【修正】現在返回兩個值：(上下文, 是否高相關度)
        return "", False

    results = []
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT content, embedding_json FROM knowledge_base")
    rows = cursor.fetchall()
    conn.close()

    is_high_confidence = False

    for row in rows:
        content = row['content']
        embedding_json = row['embedding_json']
        
        if embedding_json:
            # 從 JSON 字符串還原為 Python 列表/向量
            item_embedding = json.loads(embedding_json)
            
            # 計算相似度
            distance = euclidean_distance(query_embedding, item_embedding)
            results.append((distance, content))

    # 依距離排序 (距離小的排前面)
    results.sort(key=lambda x: x[0])

    # 【新增邏輯】檢查最佳匹配的距離是否低於信心門檻
    if results and results[0][0] < RAG_CONFIDENCE_THRESHOLD:
        is_high_confidence = True

    # 選擇前 top_k 個結果，並組成上下文
    context = []
    for distance, content in results[:top_k]:
        context.append(content)

    return "\n".join(context), is_high_confidence # 增加返回高相關度標記


# Gemini 回覆函數
def GEMINI_response(user_text):
    """
    呼叫 Google Gemini API，先進行 RAG 檢索，再將上下文與問題一起傳給模型。
    """
    if not client:
        return "⚠️ Gemini 客戶端未成功初始化，請檢查您的 GEMINI_API_KEY 。"

    # 1. RAG 檢索步驟：從您的知識庫中獲取相關上下文
    # 【修正】現在接收兩個值：rag_context 和 is_high_confidence 標記
    rag_context, is_high_confidence = query_knowledge_base(user_text, top_k=3)
    
    # 2. 組合提示詞 (Prompt Augmentation)
    tools_config = [] # 預設不啟用 Google Search

    if rag_context:
        print(f"[RAG] 檢索到上下文:\n{rag_context[:50]}...")
        
        if is_high_confidence:
            # 【高相關度邏輯】強制模型優先使用 RAG 內容，並關閉 Google Search
            print("[RAG] 檢索到高相關度知識，將優先使用 RAG 內容並禁用 Google Search。")
            system_instruction = (
                "你是一位專業的客服助理。你必須且只能根據提供的「CONTEXT」來回答問題，不得使用外部資訊。 "
                "如果 CONTEXT 無法回答問題，請明確告知使用者資訊不足。 "
                f"CONTEXT:\n---\n{rag_context}\n---"
            )
            tools_config = [] # 移除 Google Search
        else:
            # 【一般相關度邏輯】可以結合 Google Search
            tools_config = [{"google_search": {}}] # 啟用 Google Search
            system_instruction = (
                "你是一位專業的客服助理。你的首要任務是根據使用者提問和提供的「CONTEXT」來回答問題。 "
                "如果 CONTEXT 包含相關資訊，請使用它。如果 CONTEXT 不相關或無法回答，則使用你的通用知識 (Google Search) 回答。 "
                f"CONTEXT:\n---\n{rag_context}\n---"
            )
        final_prompt = user_text
    else:
        # 沒有檢索到任何自訂資料，使用 Google Search
        tools_config = [{"google_search": {}}]
        system_instruction = "你是一位樂於助人的助理，請使用最新資訊來回答問題。"
        final_prompt = user_text


    max_retries = 3
    delay = 2

    for attempt in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                temperature=0.5, 
                max_output_tokens=500,
                # 【修正】動態設定 tools
                tools=tools_config,
                # 傳入系統指令
                system_instruction=system_instruction, 
            )

            # 呼叫 Gemini API
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=final_prompt,
                config=config,
            )

            # 【內容檢查】
            if not response.text:
                error_detail = "API 回應中無文字內容。"
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason.name
                    error_detail = f"模型完成原因: {finish_reason}。"
                print(f"[Gemini Error] Generation blocked or empty. Detail: {error_detail}")
                return f"⚠️ 內容生成失敗：{error_detail}"


            # 取出回答文字
            answer = response.text.strip()

            if len(answer) > 2000:
                answer = answer[:2000] + "…（回覆過長，已截斷）"

            return answer

        except APIError as e:
            print(f"[Gemini API Error] {e}")
            if attempt < max_retries - 1:
                print(f"等待 {delay} 秒後重試...")
                time.sleep(delay)
                delay *= 2
                continue
            return "⚠️ 目前系統忙碌或 Gemini API 無法回應，請稍後再試。"

        except Exception as e:
            print(traceback.format_exc())
            return "⚠️ 發生未知錯誤，請稍後再試。"

# ========= LINE Webhook =========
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

# ========= 處理文字訊息 =========
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    print(f"[User Message]: {user_msg}")

    # 改為呼叫 Gemini 回覆函數
    reply_text = GEMINI_response(user_msg)
    print(f"[Gemini Reply]: {reply_text}")

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

# ========= 處理 Postback (維持原樣) =========
@handler.add(PostbackEvent)
def handle_postback(event):
    print(f"[Postback Data]: {event.postback.data}")

# ========= 處理加入群組事件 (微調歡迎訊息) =========
@handler.add(MemberJoinedEvent)
def welcome_new_member(event):
    try:
        uid = event.joined.members[0].user_id
        if event.source.type == 'group':
            gid = event.source.group_id
            profile = line_bot_api.get_group_member_profile(gid, uid)
            name = profile.display_name
        else:
            name = "新朋友"
            
        message = TextSendMessage(text=f"👋 歡迎 {name} 加入！我是由 Gemini 驅動的 AI 助手。")
        line_bot_api.reply_message(event.reply_token, message)
    except Exception as e:
        print(f"發送歡迎訊息失敗: {e}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"👋 歡迎新成員加入！"))


# ========= 啟動 Flask =========
if __name__ == "__main__":
    # 【新增】應用程式啟動時先設定資料庫
    setup_db()
    # 【修正】在資料庫設定完成後，再初始化知識庫 (寫入初始資料)
    initialize_knowledge_base() 
    
    # 【範例：寫入您的新知識】
    # 1. 寫入具體的考成分數資訊
    add_new_knowledge("工作考成分數為 6.5 分。")
    # 2. 寫入另一個範例，例如：誰負責考評
    add_new_knowledge("績效考評由部門主管負責，每年進行兩次。")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
