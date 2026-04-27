import os
import logging
from typing import List, Dict, Any, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from core.config import MODE_1IN

logger = logging.getLogger(__name__)

class LLMService:
    """LLM関連のロジックを管理するサービス"""
    
    def __init__(self, model_name: str = "gemini", api_key_path: str = "api_key.txt"):
        self._load_env(api_key_path)
        self.llm = self._create_model(model_name)

    def _load_env(self, filepath: str):
        if not os.path.exists(filepath):
            logger.warning(f"設定ファイルが見つかりません: {filepath}")
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        os.environ[key] = value
        except Exception as e:
            logger.error(f"環境変数の読み込み中にエラーが発生しました: {e}")

    def _create_model(self, model_name: str) -> BaseChatModel:
        if model_name == "gemini":
            return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
        elif model_name == "openai":
            return ChatOpenAI(model="gpt-4o")
        else:
            raise ValueError(f"サポートされていないモデルです: {model_name}")

    def get_response(self, messages: List[Dict[str, Any]], params: Dict[str, Any], param_history: List[Dict[str, Any]]) -> str:
        last_user_query = self._extract_last_user_query(messages)
        has_images = self._has_image(messages)
        
        system_prompt = self._build_system_prompt(params, param_history, last_user_query, has_images)
        langchain_messages = self._build_langchain_messages(system_prompt, messages)
        
        try:
            response = self.llm.invoke(langchain_messages)
            if isinstance(response.content, str):
                return response.content
            elif isinstance(response.content, list):
                return response.content[0].get("text", "") if isinstance(response.content[0], dict) else str(response.content[0])
            else:
                return str(response.content)
        except Exception as e:
            logger.error(f"LLM処理中にエラーが発生しました: {e}")
            return f"エラーが発生しました: {str(e)}"

    def _extract_last_user_query(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            return item.get("text", "")
                return str(content)
        return ""

    def _has_image(self, messages: List[Dict[str, Any]]) -> bool:
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        return True
        return False

    def _build_langchain_messages(self, system_prompt: str, raw_messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        langchain_messages = [SystemMessage(content=system_prompt)]
        for m in raw_messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            else:
                langchain_messages.append(AIMessage(content=content))
        return langchain_messages

    def _build_system_prompt(self, params: Dict[str, Any], param_history: List[Dict[str, Any]], last_user_query: str, has_images: bool) -> str:
        mode = params.get("mode", MODE_1IN)
        doc_file = "PID_1inr_doc.md" if MODE_1IN in mode else "PID_2inr_doc.md"
        doc_content = ""
        if os.path.exists(doc_file):
            with open(doc_file, "r", encoding="utf-8") as f:
                doc_content = f.read()
            if "[ユーザ入力]" in doc_content:
                doc_content = doc_content.replace("[ユーザ入力]", last_user_query)

        param_info = "\n".join([f"- {k}: {v}" for k, v in params.items()])
        history_text = ""
        if param_history:
            history_lines = ["\n\n【パラメータ調整の推移 (Change Log)】"]
            for i, entry in enumerate(param_history):
                req = entry.get("user_request", "")
                p = entry.get("parameters", {})
                p_str = ", ".join([f"{k}={v}" for k, v in p.items() if k != "mode"])
                history_lines.append(f"{i+1}. ユーザー要望: {req}\n   その時のパラメータ: [{p_str}]")
            history_text = "\n".join(history_lines)

        prompt = f"""あなたはサーボ制御調整の専門家アシスタントです。
以下のドキュメントの内容を深く理解し、ユーザーの質問に対して専門的なアドバイスを行ってください。

【参考ドキュメント: {mode}】
{doc_content}

【現在のシミュレーション環境パラメータ】
{param_info}
{history_text}

回答は技術的に正確かつ簡潔に行い、必要に応じてドキュメント内の用語や知識を引用してください。
"""
        if has_images:
            prompt += "\n重要: ユーザが現在のシミュレーション結果（波形画像）を添付しています。画像を詳しく解析し、オーバーシュート、振動、制定時間、定常偏差などの視覚的な挙動を踏まえて回答してください。\n"
        return prompt
