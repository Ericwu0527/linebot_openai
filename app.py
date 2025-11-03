from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

import os
import time
import traceback
# 引入 Google GenAI SDK
from google import genai
from google.genai import types
from google.genai.errors import APIError

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
# 客戶端會自動從環境變數 GEMINI_API_KEY 讀取金鑰
try:
    client = genai.Client()
except Exception as e:
    print(f"初始化 Gemini 客戶端失敗: {e}")
    client = None

# Gemini 回覆函數
def GEMINI_response(user_text):
    """
    呼叫 Google Gemini API (gemini-2.5-flash) 生成回覆，內含重試機制與錯誤處理。
    同時啟用 Google Search 工具以處理需要即時資訊的問題。
    """
    if not client:
        return "⚠️ Gemini 客戶端未成功初始化，請檢查您的 GEMINI_API_KEY。"

    max_retries = 3
    delay = 2

    for attempt in range(max_retries):
        try:
            # 設置生成參數
            config = types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=500, # 限制最大輸出 Token 數量
            )

            # 呼叫 Gemini API (使用最新的 gemini-2.5-flash 模型)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=user_text,
                config=config,
                # 【關鍵修復與增強】加入 Google Search 工具，讓模型可以搜尋即時資訊 (如天氣)
                tools=[{"google_search": {}}],
            )

            # 【關鍵修復】檢查是否有內容生成。如果 response.text 是 None，通常表示內容被阻擋或沒有輸出。
            if not response.text:
                error_detail = "API 回應中無文字內容。"
                
                # 嘗試從 candidates 獲取更多資訊 (檢查被阻擋的原因)
                if response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason.name
                    
                    if finish_reason == "SAFETY":
                        # 內容被安全過濾器阻擋
                        error_detail = "內容被安全過濾器阻擋，請嘗試調整提問。"
                    elif finish_reason == "RECITATION":
                        # 模型拒絕回應（例如：潛在違反使用政策，或需要外部知識但未成功獲取）
                        error_detail = "模型拒絕回應，請嘗試提供更多情境或調整提問。"
                    else:
                        error_detail = f"模型完成原因: {finish_reason}，但沒有生成文字。"

                print(f"[Gemini Error] Generation blocked or empty. Detail: {error_detail}")
                # 返回更具體的錯誤訊息
                return f"⚠️ 內容生成失敗：{error_detail}"


            # 取出回答文字 (現在確定 response.text 不為 None)
            answer = response.text.strip()

            # LINE 限制訊息長度（最多約 2000 字元）
            if len(answer) > 2000:
                answer = answer[:2000] + "…（回覆過長，已截斷）"

            return answer

        except APIError as e:
            # 處理 Gemini API 相關錯誤，例如認證失敗、配額用盡等
            print(f"[Gemini API Error] {e}")
            if attempt < max_retries - 1:
                print(f"等待 {delay} 秒後重試...")
                time.sleep(delay)
                delay *= 2  # 指數退避
                continue
            return "⚠️ 目前系統忙碌或 Gemini API 無法回應，請稍後再試。"

        except Exception as e:
            # 處理其他未知錯誤，例如網路超時或解析錯誤
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
        # 處理簽章驗證失敗
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
        # 嘗試獲取加入成員的名稱
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
    # 使用 Render 提供的 PORT 環境變數
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
