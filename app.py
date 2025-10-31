from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *

import os
import time
import traceback
from openai import OpenAI, OpenAIError

# 初始化 Flask
app = Flask(__name__)

# Channel Access Token / Secret
line_bot_api = LineBotApi(os.getenv('CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('CHANNEL_SECRET'))

# 初始化 OpenAI Client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# GPT 回覆函數
def GPT_response(user_text):
    """
    呼叫 OpenAI GPT 生成回覆，內含重試機制與錯誤處理。
    """
    max_retries = 3
    delay = 2

    for attempt in range(max_retries):
        try:
            # 使用最新 API (Responses endpoint)
            response = client.responses.create(
                model="gpt-4o-mini",  # 最新模型，效能佳
                input=user_text,
                temperature=0.5,
                max_output_tokens=500,
                timeout=15,  # 秒數：防止超時
            )

            # 取出回答文字
            answer = response.output[0].content[0].text.strip()

            # LINE 限制訊息長度（最多約 2000 字元）
            if len(answer) > 2000:
                answer = answer[:2000] + "…（回覆過長，已截斷）"

            return answer

        except OpenAIError as e:
            print(f"[OpenAI API Error] {e}")
            if attempt < max_retries - 1:
                print(f"等待 {delay} 秒後重試...")
                time.sleep(delay)
                delay *= 2  # 指數退避
                continue
            return "⚠️ 目前系統忙碌或 API 無法回應，請稍後再試。"

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

    reply_text = GPT_response(user_msg)
    print(f"[GPT Reply]: {reply_text}")

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply_text)
    )

# ========= 處理 Postback =========
@handler.add(PostbackEvent)
def handle_postback(event):
    print(f"[Postback Data]: {event.postback.data}")

# ========= 處理加入群組事件 =========
@handler.add(MemberJoinedEvent)
def welcome_new_member(event):
    uid = event.joined.members[0].user_id
    gid = event.source.group_id
    profile = line_bot_api.get_group_member_profile(gid, uid)
    name = profile.display_name
    message = TextSendMessage(text=f"👋 歡迎 {name} 加入群組！")
    line_bot_api.reply_message(event.reply_token, message)

# ========= 啟動 Flask =========
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)
