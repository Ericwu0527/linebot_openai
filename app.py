from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

import os
import time
import traceback
import math 
import json 
from datetime import datetime 

# 引入 Firestore 函式庫
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

# 從環境變數獲取 Gemini API Key
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
    """計算兩個向量之間的餘弦距離 (1 - 餘弦相似度)。"""
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
    """
    檢查 Firestore 資料庫，如果沒有資料則插入初始資料並生成向量。
    【使用自定義 ID】
    """
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
        for i, item in enumerate(initial_knowledge_data):
            content = item['content']
            
            doc_id = f"knowledge_{i+1:02d}"  
            embedding = get_embedding(content)
            
            if embedding:
                embedding_json = json.dumps(embedding) 
                
                try:
                    # 使用 document(doc_id).set() 自定義 ID
                    db.collection(KNOWLEDGE_COLLECTION).document(doc_id).set({
                        'content': content,
                        'embedding_json': embedding_json,
                        'created_at': firestore.SERVER_TIMESTAMP 
                    })
                except Exception as e:
                    print(f"寫入 Firestore 失敗: {e}")
                    
        print("RAG 知識庫初始化完成，資料已儲存到 Firestore。")
    else:
        print("RAG 知識庫已包含資料，跳過初始化。")


def query_knowledge_base(query_text, top_k=5):
    """從 Firestore 資料庫中檢索與查詢最相關的文檔 (企業知識)。"""
    # 邏輯與之前版本相同，負責檢索企業知識
    if not db:
        return "", False

    query_embedding = get_embedding(query_text)
    if not query_embedding:
        return "", False

    results = []
    is_high_confidence = False

    try:
        docs = db.collection(KNOWLEDGE_COLLECTION).stream()
        
        for doc in docs:
            data = doc.to_dict()
            content = data.get('content')
            embedding_json = data.get('embedding_json')
            
            if content and embedding_json:
                item_embedding = json.loads(embedding_json)
                
                distance = cosine_distance(query_embedding, item_embedding)
                results.append((distance, content))

    except Exception as e:
        print(f"[Firestore Query Error] 無法查詢知識庫: {e}") 
        return "", False 

    results.sort(key=lambda x: x[0])

    if results and results[0][0] < RAG_CONFIDENCE_THRESHOLD:
        is_high_confidence = True

    context = []
    for distance, content in results[:top_k]:
        context.append(content)

    return "\n".join(context), is_high_confidence


def record_reminder(user_id, raw_text):
    """將用戶輸入的原始行程文字寫入 Firestore 的 'reminders' 集合。"""
    if not db:
        return False, "Firestore 客戶端未初始化，無法記錄。"
    
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id, 
            'raw_text': raw_text,
            'recorded_at': firestore.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        
        return True, f"✅ 已為您記下行程：\n「{raw_text}」"
    except Exception as e:
        print(f"[Firestore Record Error] 無法記錄行程: {e}")
        return False, f"❌ 記錄行程失敗：{e}"


def get_user_reminders(user_id):
    """
    從 Firestore 的 'reminders' 集合中讀取特定用戶的所有未完成行程。
    【注意】：此函式需要 Firestore 複合索引才能運行。
    """
    if not db:
        return []
    
    reminders_list = []
    try:
        docs = db.collection(REMINDER_COLLECTION)\
                 .where('user_id', '==', user_id)\
                 .where('is_completed', '==', False)\
                 .order_by('recorded_at', direction=firestore.Query.DESCENDING)\
                 .limit(10)\
                 .stream()
        
        for i, doc in enumerate(docs):
            data = doc.to_dict()
            recorded_time = data.get('recorded_at')
            
            time_str = recorded_time.strftime('%m/%d %H:%M') if recorded_time else "未知時間"
            
            reminders_list.append(f"行程 {i+1}. 內容: {data['raw_text']} (記錄於: {time_str})")
            
    except Exception as e:
        # 請務必在 Firebase Console 中建立複合索引！
        print(f"[Firestore Read Error] 無法讀取行程: {e}")
        return []

    return reminders_list


def GEMINI_response(user_text, user_id):
    """
    所有非指令的輸入都進入 RAG 檢索流程 (企業知識 + 個人行程)，然後交給模型。
    """
    if not client:
        return "⚠️ Gemini 客戶端未成功初始化，請檢查您的 GEMINI_API_KEY 。"

    # 1. RAG 檢索企業知識
    rag_context, is_high_confidence = query_knowledge_base(user_text, top_k=5)
    
    # 2. 讀取使用者個人行程
    user_reminders = get_user_reminders(user_id)
    personal_context = "\n".join(user_reminders)
    
    # 3. 組合提示詞 (Prompt Augmentation)
    tools_config = []
    full_context_parts = []

    if rag_context:
        full_context_parts.append(f"【企業知識】:\n{rag_context}")
    if personal_context:
        full_context_parts.append(f"【您的個人行程（未完成事項）】:\n{personal_context}")

    full_context = "\n---\n".join(full_context_parts)

    if full_context:
        # 只要有任何上下文，就啟用 Google Search 進行補充
        tools_config = [{"google_search": {}}]
        
        # 調整 System Instruction，強調基於 Context 回答 (符合 HackMD 的意圖驅動)
        system_instruction = (
            "你是一位專業且樂於助人的助理。你擁有一份【企業知識】和一份【您的個人行程】。 "
            "請嚴格根據以下規則回答：\n"
            "1. 當用戶詢問與 CONTEXT 中任一部分相關的問題時，請**直接且完整地**使用 CONTEXT 中的資訊來回答。\n"
            "2. 當用戶詢問『我的名字是什麼？』，請**特別注意**：如果【您的個人行程】CONTEXT 中有明確的提示（例如：『我叫Eric』），請基於該記錄回答：『根據您的行程記錄，您曾記下您叫Eric。』，避免直接聲稱知道您的名字。\n"
            "3. 對於其他通用問題或 CONTEXT 不足時，使用 Google Search。\n"
            f"所有可用的 CONTEXT:\n===\n{full_context}\n==="
        )
        final_prompt = user_text
    else:
        # 沒有任何上下文，只使用 Google Search
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

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=final_prompt,
                config=config,
            )

            if not response.text:
                # ... (錯誤處理保持不變)
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
            # ... (重試邏輯保持不變)
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

# ========= LINE Webhook / Flask Routes / Handler (保持不變) =========
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
        # 清除企業知識庫 (KNOWLEDGE_COLLECTION)
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

        # 重新初始化，使用新的可讀ID寫入
        initialize_knowledge_base() 
        
        return "✅ Firestore 企業知識庫已清除並重新初始化完成。"
    except Exception as e:
        print(f"❌ Firestore 資料庫重設失敗: {e}")
        return f"❌ 資料庫重設失敗: {e}"


# ========= 處理文字訊息 (核心邏輯：區分指令與查詢) =========
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    user_id = event.source.user_id 
    print(f"[User Message]: {user_msg}")

    # 定義行程記錄指令前綴
    REMINDER_COMMAND_PREFIXES = ["/記下:", "/記下："]
    
    command_found = False
    reply_text = ""
    
    # 1. 檢查並處理明確的寫入指令
    for prefix in REMINDER_COMMAND_PREFIXES:
        if user_msg.startswith(prefix):
            reminder_content = user_msg[len(prefix):].strip()
            command_found = True
            if reminder_content:
                success, message = record_reminder(user_id, reminder_content)
                reply_text = message
            else:
                reply_text = f"請在指令後提供要記下的內容。格式範例：/記下: 11/26 13:30 去大潤發"
            break

    if command_found:
        # 如果是指令 (寫入資料庫)，直接回覆指令結果
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply_text)
        )
    else:
        # 2. 如果不是指令，就走 RAG 查詢流程 (意圖驅動)
        # 任何非指令的輸入都視為查詢，讓 RAG 和 Gemini 處理
        reply_text = GEMINI_response(user_msg, user_id) 
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
