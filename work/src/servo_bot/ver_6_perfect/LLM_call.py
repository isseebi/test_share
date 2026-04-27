import os
import logging
from typing import List, Dict, Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# 1. ロギングの設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ConfigManager:
    """環境変数や設定値のロードを管理するクラス"""
    
    @staticmethod
    def load_env_from_file(filepath: str = "api_key.txt") -> None:
        """テキストファイルからキーを読み込み、環境変数にセットする"""
        if not os.path.exists(filepath):
            logger.warning(f"設定ファイルが見つかりません: {filepath}")
            return
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        os.environ[key] = value
            logger.info(f"{filepath} から環境変数を読み込みました。")
        except Exception as e:
            logger.error(f"環境変数の読み込み中にエラーが発生しました: {e}")


class LLMFactory:
    """LLMモデルの生成を担当するファクトリクラス (拡張性を高める)"""
    
    @staticmethod
    def create_model(model_name: str = "gemini") -> BaseChatModel:
        if model_name == "gemini":
            return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
        elif model_name == "openai":
            return ChatOpenAI(model="gpt-4o")
        else:
            raise ValueError(f"サポートされていないモデルです: {model_name}")


class MessageParser:
    """メッセージ履歴の解析・変換を担当するクラス"""
    
    @staticmethod
    def extract_last_user_query(messages: List[Dict[str, Any]]) -> str:
        """メッセージ履歴から最新のユーザーテキストを抽出する"""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            return item.get("text", "")
                return str(content)
        return ""

    @staticmethod
    def has_image(messages: List[Dict[str, Any]]) -> bool:
        """メッセージ内に画像が含まれているかを判定する"""
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image_url":
                        return True
        return False

    @staticmethod
    def build_langchain_messages(system_prompt: str, raw_messages: List[Dict[str, Any]]) -> List[BaseMessage]:
        """辞書型のメッセージリストをLangChainのMessageオブジェクトに変換する"""
        langchain_messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
        
        for m in raw_messages:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            else:
                langchain_messages.append(AIMessage(content=content))
                
        return langchain_messages


class PromptBuilder:
    """プロンプトの組み立てを担当するクラス"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir

    def _get_document_content(self, mode: str, last_user_query: str) -> str:
        """モードに応じたドキュメントを読み込み、プレースホルダーを置換する"""
        doc_file = "PID_1inr_doc.md" if "1慣性系" in mode else "PID_2inr_doc.md"
        filepath = os.path.join(self.base_dir, doc_file)
        
        if not os.path.exists(filepath):
            logger.warning(f"ドキュメントファイルが見つかりません: {filepath}")
            return ""

        with open(filepath, "r", encoding="utf-8") as f:
            doc_content = f.read()

        if "[ユーザ入力]" in doc_content:
            doc_content = doc_content.replace("[ユーザ入力]", last_user_query)
            
        return doc_content

    def _build_history_text(self, param_history: List[Dict[str, Any]]) -> str:
        """パラメータ変更履歴のテキストを生成する"""
        if not param_history:
            return ""
            
        history_lines = ["\n\n【パラメータ調整の推移 (Change Log)】"]
        for i, entry in enumerate(param_history):
            req = entry.get("user_request", "")
            p = entry.get("parameters", {})
            p_str = ", ".join([f"{k}={v}" for k, v in p.items() if k != "mode"])
            history_lines.append(f"{i+1}. ユーザー要望: {req}\n   その時のパラメータ: [{p_str}]")
            
        return "\n".join(history_lines)

    def generate_system_prompt(self, params: Dict[str, Any], param_history: List[Dict[str, Any]], last_user_query: str, has_images: bool) -> str:
        """すべての情報を統合してシステムプロンプトを生成する"""
        mode = params.get("mode", "1慣性系 (Single Inertia)")
        doc_content = self._get_document_content(mode, last_user_query)
        param_info = "\n".join([f"- {k}: {v}" for k, v in params.items()])
        history_md = self._build_history_text(param_history)

        prompt = f"""あなたはサーボ制御調整の専門家アシスタントです。
                    以下のドキュメントの内容を深く理解し、ユーザーの質問に対して専門的なアドバイスを行ってください。

                    【参考ドキュメント: {mode}】
                    {doc_content}

                    【現在のシミュレーション環境パラメータ】
                    {param_info}
                    {history_md}

                    回答は技術的に正確かつ簡潔に行い、必要に応じてドキュメント内の用語や知識を引用してください。
                """
        if has_images:
            prompt += "\n重要: ユーザが現在のシミュレーション結果（波形画像）を添付しています。画像を詳しく解析し、オーバーシュート、振動、制定時間、定常偏差などの視覚的な挙動を踏まえて回答してください。\n"

        return prompt


class ServoAssistantService:
    """一連の処理を統合するメインサービス・クラス"""
    
    def __init__(self, model_name: str = "gemini"):
        self.llm = LLMFactory.create_model(model_name)
        self.prompt_builder = PromptBuilder()

    def get_llm_response(self, messages: List[Dict[str, Any]], params: Dict[str, Any], param_history: List[Dict[str, Any]]) -> str:
        """外部から呼び出されるメインのインターフェース"""
        
        # 1. メッセージから必要な情報をパース
        last_user_query = MessageParser.extract_last_user_query(messages)
        has_images = MessageParser.has_image(messages)
        
        # 2. システムプロンプトの生成
        system_prompt = self.prompt_builder.generate_system_prompt(
            params=params,
            param_history=param_history,
            last_user_query=last_user_query,
            has_images=has_images
        )
        logger.debug(f"=== 生成されたシステムプロンプト ===\n{system_prompt}")

        # 3. LangChain用メッセージへの変換
        langchain_messages = MessageParser.build_langchain_messages(system_prompt, messages)
        print(f"=== LangChainに渡すメッセージ ===")
        print(langchain_messages)
        # 4. LLMへの問い合わせ
        try:
            logger.info("LLMへの問い合わせを開始します...")
            response = self.llm.invoke(langchain_messages)
            
            # レスポンスの抽出処理
            if isinstance(response.content, str):
                return response.content
            elif isinstance(response.content, list):
                return response.content[0].get("text", "") if isinstance(response.content[0], dict) else str(response.content[0])
            else:
                return str(response.content)
                
        except Exception as e:
            logger.error(f"LLM処理中にエラーが発生しました: {e}")
            return f"エラーが発生しました: {str(e)}"


# --- 実行用エントリポイント ---
if __name__ == "__main__":
    # モジュールとしてimportされた際はここは実行されません（副作用の防止）
    ConfigManager.load_env_from_file("api_key.txt")
    
    # テスト用のダミーデータ
    dummy_messages = [{"role": "user", "content": "ゲインを上げたら振動しました。どうすればいいですか？"}]
    dummy_params = {"mode": "1慣性系", "Kp": 10}
    dummy_history = []
    
    # サービスを初期化して実行
    service = ServoAssistantService(model_name="gemini")
    result = service.get_llm_response(dummy_messages, dummy_params, dummy_history)
    
    print("\n=== 最終回答 ===")
    print(result)