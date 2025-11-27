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

# -------------------------------------------------------------
# 【Function Calling 工具函式】
# -------------------------------------------------------------
def record_reminder(user_id: str, raw_text: str) -> str:
    """
    將用戶輸入的原始行程文字寫入 Firestore 的 'reminders' 集合。
    
    Args:
        user_id: 唯一識別用戶的 LINE ID。
        raw_text: 用戶要求記下的行程內容。
        
    Returns:
        JSON 格式的字串，包含 status 和 message。
    """
    if not db:
        return json.dumps({"status": "error", "message": "Firestore 客戶端未初始化，無法記錄。"})
    
    try:
        db.collection(REMINDER_COLLECTION).add({
            'user_id': user_id, 
            'raw_text': raw_text,
            'recorded_at': firestore.SERVER_TIMESTAMP,
            'is_completed': False,
        })
        
        return json.dumps({"status": "success", "message": f"已成功記錄行程：『{raw_text}』"})
    except Exception as e:
        print(f"[Firestore Record Error] 無法記錄行程: {e}")
        return json.dumps({"status": "error", "message": f"記錄行程失敗：{e}"})


def get_user_reminders(user_id):
    """
    從 Firestore 的 'reminders' 集合中讀取特定用戶的所有未完成行程。
    【注意】：此函式需要 Firestore 複合索引才能運行。
    """
    if not db:
        return []
    
    reminders_list = []
    try:
        # 查詢需要複合索引的語句
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
        # 捕獲並打印索引錯誤
        print(f"[Firestore Read Error] 無法讀取行程: {e}")
        return []

    return reminders_list


def GEMINI_response_with_tools(user_text, user_id):
    """
    使用 Function Calling (工具呼叫) 來處理行程記錄和 RAG 查詢。
    """
    if not client:
        return "⚠️ Gemini 客戶端未成功初始化，請檢查您的 GEMINI_API_KEY 。"
    
    # 步驟 1: 建立工具配置，並準備 RAG 上下文
    available_tools = [record_reminder] 
    
    rag_context, _ = query_knowledge_base(user_text, top_k=5)
    user_reminders = get_user_reminders(user_id)
    personal_context = "\n".join(user_reminders)
    
    full_context_parts = []
    if rag_context:
        full_context_parts.append(f"【企業知識】:\n{rag_context}")
    if personal_context:
        full_context_parts.append(f"【您的個人行程（未完成事項）】:\n{personal_context}")
    full_context = "\n---\n".join(full_context_parts)
    
    
    # 1.3 設置 System Instruction
    system_instruction = (
        "你是一位專業且樂於助人的助理。你擁有一份【企業知識】和一份【您的個人行程】。"
        "請嚴格遵守以下規則：\n"
        "1. 如果用戶要求『記下』、『幫我記錄』或『提醒我』某件事，你必須呼叫 `record_reminder` 工具，並將用戶要求的內容作為 `raw_text` 參數。\n"
        "2. 當用戶詢問與 CONTEXT 中任一部分相關的問題時，請直接使用 CONTEXT 中的資訊回答。\n"
        "3. 當用戶詢問名字時，請參考個人行程記錄回答，例如：『根據您的記錄，您曾記下您叫Eric。』\n"
        "4. 對於其他通用問題或 CONTEXT 不足時，使用 Google Search。\n"
        f"RAG CONTEXT:\n===\n{full_context}\n==="
    )
    
    # 2. 第一次 API 呼叫 (讓模型決定是回答還是呼叫工具)
    # 【修正點】: 使用標準的 types.Part 建立方式，避免 TypeError
    history = [
        types.Content(
            role="user", 
            parts=[types.Part(text=user_text)]
        )
    ]
    
    config = types.GenerateContentConfig(
        temperature=0.3,
        tools=available_tools + [{"google_search": {}}], 
        system_instruction=system_instruction
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history,
            config=config,
        )

        # 3. 處理 Function Calling (工具呼叫)
        if response.function_calls:
            call = response.function_calls[0] 
            
            # 呼叫 Python 函式，傳入 user_id 和模型解析出來的參數
            tool_response = globals()[call.name](user_id, **dict(call.args))
            
            print(f"[Function Calling] 呼叫 {call.name}, 參數: {call.args}, 結果: {tool_response}")
            
            # 將工具結果回傳給 Gemini 進行第二次呼叫，生成最終自然語言回覆
            history.append(response.candidates[0].content)
            history.append(types.Content(
                role="tool",
                parts=[types.Part.from_function_response(
                    name=call.name,
                    response=json.loads(tool_response)
                )]
            ))
            
            # 第二次 API 呼叫
            final_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=history,
                config=config,
            )
            return final_response.text.strip()
            
        else:
            # 如果沒有呼叫工具，直接返回模型的第一輪文字回覆 (RAG 查詢或通用回答)
            return response.text.strip()

    except Exception as e:
        print(traceback.format_exc())
        return "⚠️ 服務發生錯誤，無法處理您的請求。"


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


# ========= 處理文字訊息 (簡化為 Function Calling 流程) =========
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_msg = event.message.text
    user_id = event.source.user_id 
    print(f"[User Message]: {user_msg}")

    # 現在所有輸入都直接進入 Function Calling 流程
    reply_text = GEMINI_response_with_tools(user_msg, user_id) 
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
