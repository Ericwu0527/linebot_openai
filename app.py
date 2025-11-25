from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

import os
import time
import traceback
import math 
import json 

# 【變更 1】引入 Firestore 函式庫
from google.cloud import firestore

# 引入 Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

# ======================= RAG 知識庫設定 (使用 Firestore) =======================
# 設定 Firestore 集合名稱
KNOWLEDGE_COLLECTION = "knowledge_base" 

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
                
                try:
                    db.collection(KNOWLEDGE_COLLECTION).add({
                        'content': content,
                        'embedding_json': embedding_json,
                        'created_at': firestore.SERVER_TIMESTAMP 
                    })
                except Exception as e:
                    print(f"寫入 Firestore 失敗: {e}")
                    
        print("RAG 知識庫初始化完成，資料已儲存到 Firestore。")
    else:
        print("RAG 知識庫已包含資料，跳過初始化。")

# 【移除】add_new_knowledge 函式已被移除
# 【移除】delete_knowledge 函式已被移除

def query_knowledge_base(query_text, top_k=5):
    """從 Firestore 資料庫中檢索與查詢最相關的文檔。"""
    if not db:
        return "", False

    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return "", False

    results = []
    is_high_confidence = False

    try:
        # 從 Firestore 讀取所有文檔
        docs = db.collection(KNOWLEDGE_COLLECTION).stream()
        
        for doc in docs:
            data = doc.to_dict()
            content = data.get('content')
            embedding_json = data.get('embedding_json')
            
            if content and embedding_json:
                item_embedding = json.loads(embedding_json)
                
                # 計算餘弦距離
                distance = cosine_distance(query_embedding, item_embedding)
                results.append((distance, content))

    except Exception as e:
        print(f"[Firestore Query Error] 無法查詢知識庫: {e}") 
        return "", False 

    # 依距離排序 (距離小的排前面)
    results.sort(key=lambda x: x[0])

    # 檢查最佳匹配的距離是否低於信心門檻
    if results and results[0][0] < RAG_CONFIDENCE_THRESHOLD:
        is_high_confidence = True

    # 選擇前 top_k 個結果，並組成上下文
    context = []
    for distance, content in results[:top_k]:
        context.append(content)

    return "\n".join(context), is_high_confidence


def GEMINI_response(user_text):
    """
    呼叫 Google Gemini API，先進行 RAG 檢索，再將上下文與問題一起傳給模型。
    """
    if not client:
        return "⚠️ Gemini 客戶端未成功初始化，請檢查您的 GEMINI_API_KEY 。"

    # 1. RAG 檢索步驟：從您的知識庫中獲取相關上下文
    rag_context, is_high_confidence = query_knowledge_base(user_text, top_k=5)
    
    # 2. 組合提示詞 (Prompt Augmentation)
    tools_config = [] 

    if rag_context:
        tools_config = [{"google_search": {}}]
        
        if is_high_confidence:
            print("[RAG] 檢索到高相關度知識，將優先使用 RAG 內容，但同時允許 Google Search 處理非業務問題。")
            system_instruction = (
                "你是一位企業內部客服助理。請**優先且嚴格**根據下列 CONTEXT 來回答**與內部業務相關**的問題。 "
                "請將 CONTEXT 中的資訊直接轉換為自然語言回答。 "
                "**如果問題明顯是外部知識、數學計算或通用查詢，則請忽略 CONTEXT 的限制，使用 Google Search 或你的通用知識來回答。** "
                "如果 CONTEXT 相關但不足以回答，則可結合 Google Search。 "
                f"CONTEXT:\n---\n{rag_context}\n---"
            )
        else:
            system_instruction = (
                "你是一位樂於助人的助理。請根據使用者的問題回答。 "
                "**優先**使用 Google Search 獲取最新資訊，並同時參考提供的 CONTEXT。 "
                "如果 CONTEXT 相關，請結合；如果 CONTEXT 不相關，請忽略並僅使用 Google Search 的資訊來回答。\n\n"
                f"CONTEXT:\n---\n{rag_context}\n---"
            )
        final_prompt = user_text
    else:
        # 沒有檢索到任何自訂資料，只使用 Google Search
        tools_config = [{"google_search": {}}]
        system_instruction = "你是一位樂於助人的助理，請使用最新資訊來回答問題。"
        final_prompt = user_text


    max_retries = 3
    delay = 2

    for attempt in range(max_retries):
        try:
            config = types.GenerateContentConfig(
                temperature=0.5, 
                max_output_tokens=1500,
                tools=tools_config,
                system_instruction=system_instruction, 
            )

            # 呼叫 Gemini API
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=final_prompt,
                config=config,
            )

            if not response.text:
                error_detail = "API 回應中無文字內容。"
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason.name
                    error_detail = f"模型完成原因: {finish_reason}。"
                print(f"[Gemini Error] Generation blocked or empty. Detail: {error_detail}")
                return f"⚠️ 內容生成失敗：{error_detail}"

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
@app.route('/')
def index():
    return "✅ LINE Bot Flask App is running on Render!"


@app.route("/callback", methods=['POST'])
def callback():
    initialize_knowledge_base() 
    
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info(f"Request body: {body}")

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


@app.route("/resetdb")
def reset_db():
    """手動清除 Firestore 知識庫集合並重建初始資料。"""
    if not db:
        return "❌ Firestore 客戶端未初始化，無法執行重設。"
    
    try:
        docs = db.collection(KNOWLEDGE_COLLECTION).list_documents()
        deleted_count = 0
        batch = db.batch()
        for doc in docs:
            batch.delete(doc)
            deleted_count += 1
        
        if deleted_count > 0:
             batch.commit()
             print(f"舊的知識庫集合 ({KNOWLEDGE_COLLECTION}) 中 {deleted_count} 筆資料已移除。")
        else:
             print("知識庫集合為空，無需刪除。")

        initialize_knowledge_base() 
        
        return "✅ Firestore 資料庫已清除並重新初始化完成。"
    except Exception as e:
        print(f"❌ Firestore 資料庫重設失敗: {e}")
        return f"❌ 資料庫重設失敗: {e}"


# ========= 處理文字訊息 【修改】只保留正常的問答流程 =========
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    print(f"[User Message]: {user_msg}")

    # 3. 正常的問答流程
    reply_text = GEMINI_response(user_msg)
    print(f"[Gemini Reply]: {reply_text}")

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

# Postback 和 MemberJoinedEvent 處理保持不變
@handler.add(PostbackEvent)
def handle_postback(event):
    print(f"[Postback Data]: {event.postback.data}")

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
    initialize_knowledge_base() 
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
